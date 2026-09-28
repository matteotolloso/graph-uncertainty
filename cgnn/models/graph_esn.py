"""Graph Echo State Network ensemble (not in the AAAI paper; exploratory).

``num_reservoirs`` fixed random graph reservoirs (``graphesn``), each with a
ridge-regression readout fitted once on the training nodes. The ensemble
yields credal (hull) and classical uncertainties.
Logged: ``{val,test}_auroc_{EU,AU,TU}_{credal,classic}``, ``{val,test}_auroc`` (= EU_credal).
Requires the optional ``graphesn`` package.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from cgnn.models.base import NodeUQModule, id_mask, ood_targets
from cgnn.uncertainty.ensemble import ensemble_uncertainties

_FAMILIES = ("credal", "classic")
_KINDS = ("EU", "AU", "TU")


class GraphEchoStateNetwork(NodeUQModule):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: int,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        ood_in_val: bool = True,
        spectral_radius: float = 0.9,
        input_scaling: float = 1.0,
        leakage: float = 1.0,
        num_reservoirs: int = 1,
        readout_regularization: float = 1e-3,
        bias: bool = False,
        pooling: str | None = None,
        fully: bool = False,
        max_iterations: int | None = None,
        epsilon: float | None = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__()
        from graphesn import Readout

        self.save_hyperparameters()
        self.automatic_optimization = False
        if num_reservoirs <= 0:
            raise ValueError("num_reservoirs must be positive.")

        self.reservoirs = nn.ModuleList(
            [
                self._build_reservoir(
                    in_channels,
                    hidden_channels,
                    num_layers,
                    spectral_radius,
                    input_scaling,
                    leakage,
                    bias,
                    pooling,
                    fully,
                    max_iterations,
                    epsilon,
                )
                for _ in range(num_reservoirs)
            ]
        )
        self.readouts = nn.ModuleList(
            [Readout(num_features=r.out_features, num_targets=out_channels) for r in self.reservoirs]
        )
        self.criterion = nn.CrossEntropyLoss()
        self.C = out_channels
        self.lr = lr
        self.weight_decay = weight_decay
        self.ood_in_val = ood_in_val
        self.readout_regularization = readout_regularization
        self._readouts_fitted = False

    @staticmethod
    def _build_reservoir(
        in_channels,
        hidden_channels,
        num_layers,
        spectral_radius,
        input_scaling,
        leakage,
        bias,
        pooling,
        fully,
        max_iterations,
        epsilon,
    ):
        from graphesn import StaticGraphReservoir, initializer

        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if pooling not in (None, "none"):
            raise ValueError("GraphESN pooling must be None for node-level classification.")
        reservoir = StaticGraphReservoir(
            num_layers=num_layers,
            in_features=in_channels,
            hidden_features=hidden_channels,
            bias=bias,
            pooling=None,
            fully=fully,
            max_iterations=max_iterations,
            epsilon=epsilon,
        )
        reservoir.initialize_parameters(
            recurrent=initializer("uniform", rho=spectral_radius),
            input=initializer("uniform", scale=input_scaling),
            bias=initializer("uniform", scale=input_scaling) if bias else None,
            leakage=leakage,
        )
        reservoir.requires_grad_(False)
        return reservoir

    def encode_member(self, data, member_idx: int):
        with torch.no_grad():
            return self.reservoirs[member_idx](data.edge_index, data.x)

    def member_logits(self, data) -> torch.Tensor:
        logits = []
        for reservoir, readout in zip(self.reservoirs, self.readouts, strict=True):
            with torch.no_grad():
                logits.append(readout(reservoir(data.edge_index, data.x)))
        return torch.stack(logits, dim=0)  # [M, N, C]

    def forward(self, data) -> torch.Tensor:
        return self.member_logits(data).mean(dim=0)

    def _fit_readouts(self):
        if self._readouts_fitted:
            return
        train_loader = self.trainer.train_dataloader
        for member_idx, readout in enumerate(self.readouts):

            def data_iterator(member_idx=member_idx):
                for batch in train_loader:
                    batch = batch.to(self.device)
                    mask = self.split_mask(batch, "train")
                    if not mask.any():
                        continue
                    yield self.encode_member(batch, member_idx)[mask].detach(), batch.y[mask].detach()

            readout.fit(data=data_iterator(), regularization=self.readout_regularization)
        self._readouts_fitted = True

    def on_train_epoch_start(self):
        self._fit_readouts()

    def _member_probabilities(self, data) -> torch.Tensor:
        return F.softmax(self.member_logits(data), dim=-1)

    def training_step(self, batch, batch_idx):
        mean_probs = self._member_probabilities(batch).mean(dim=0)
        mask = self.split_mask(batch, "train")
        probs, labels = mean_probs[mask], torch.argmax(batch.y[mask], dim=1)
        loss = F.nll_loss(torch.log(probs.clamp_min(1e-12)), labels)
        preds = torch.argmax(probs, dim=1)
        n = mask.sum()
        self.log("train_loss", loss, batch_size=n)
        self.log("train_acc", (preds == labels).float().mean(), batch_size=n)
        self.log("train_f1", self.f1(preds, labels), batch_size=n)
        return loss

    def on_validation_epoch_start(self):
        self.buffer("val").clear()

    def validation_step(self, batch, batch_idx):
        stacked = self._member_probabilities(batch)
        mask = self.split_mask(batch, "val")
        mean_probs, stacked, y = (
            stacked.mean(dim=0)[mask].detach(),
            stacked[:, mask, :].detach(),
            batch.y[mask],
        )

        is_id = id_mask(y)
        loss = None
        labels = preds = torch.empty(0, dtype=torch.long, device=self.device)
        if is_id.any():
            labels = torch.argmax(y[is_id], dim=1)
            probs_id = mean_probs[is_id]
            loss = F.nll_loss(torch.log(probs_id.clamp_min(1e-12)), labels)
            preds = torch.argmax(probs_id, dim=1)
            n = is_id.sum()
            self.log("val_loss", loss, prog_bar=True, batch_size=n, on_step=False, on_epoch=True)
            self.log("val_acc", (preds == labels).float().mean(), batch_size=n, on_step=False, on_epoch=True)

        buf = self.buffer("val")
        buf.add(labels=labels, preds=preds)
        if self.ood_in_val:
            buf.add(ood_targets=ood_targets(y), **ensemble_uncertainties(stacked))
        return loss

    def _log_uncertainty_aurocs(self, buf, split: str) -> None:
        targets = buf.cat("ood_targets")
        for family in _FAMILIES:
            for kind in _KINDS:
                key = f"{kind}_{family}"
                self.log(
                    f"{split}_auroc_{key}", self.auroc(buf.cat(key), targets), prog_bar=key == "EU_credal"
                )
        self.log(f"{split}_auroc", self.auroc(buf.cat("EU_credal"), targets), prog_bar=split == "val")

    def on_validation_epoch_end(self):
        buf = self.buffer("val")
        if not buf:
            return
        labels = buf.cat("labels")
        if labels.numel() > 0:
            self.log("val_f1", self.f1(buf.cat("preds"), labels), prog_bar=True)
        if self.ood_in_val and "EU_credal" in buf:
            self._log_uncertainty_aurocs(buf, "val")
        buf.clear()

    def on_test_epoch_start(self):
        self.buffer("test").clear()

    def test_step(self, batch, batch_idx):
        stacked = self._member_probabilities(batch)
        mask = self.split_mask(batch, "test")
        mean_probs, stacked, y = (
            stacked.mean(dim=0)[mask].detach(),
            stacked[:, mask, :].detach(),
            batch.y[mask],
        )
        unc = ensemble_uncertainties(stacked)
        is_id = id_mask(y)
        self.buffer("test").add(
            ood_targets=ood_targets(y),
            labels=torch.argmax(y[is_id], dim=1),
            preds=torch.argmax(mean_probs[is_id], dim=1),
            **unc,
        )
        return unc["EU_credal"]

    def on_test_epoch_end(self):
        buf = self.buffer("test")
        if not buf:
            return
        self._log_uncertainty_aurocs(buf, "test")
        labels, preds = buf.cat("labels"), buf.cat("preds")
        if labels.numel() > 0:
            self.log("test_acc", (preds == labels).float().mean())
            self.log("test_f1", self.f1(preds, labels))
        buf.clear()

    def configure_optimizers(self):
        return None
