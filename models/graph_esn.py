import os
import sys
from typing import Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from graphesn import Readout, StaticGraphReservoir, initializer
from torchmetrics import AUROC
from torchmetrics.classification import F1Score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.math import compute_uncertainties


class GraphEchoStateNetwork(L.LightningModule):
    """
    Graph Echo State Network for node classification/OOD scoring.

    The graph reservoir is fixed after random initialization. Each ensemble
    member has a graphesn.Readout fitted by ridge regression, preserving the
    reservoir-computing setup while fitting into the Lightning sweep flow.
    """

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
        pooling: Optional[str] = None,
        fully: bool = False,
        max_iterations: Optional[int] = None,
        epsilon: Optional[float] = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        if num_reservoirs <= 0:
            raise ValueError("num_reservoirs must be positive.")

        self.reservoirs = nn.ModuleList([
            self._build_reservoir(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                spectral_radius=spectral_radius,
                input_scaling=input_scaling,
                leakage=leakage,
                bias=bias,
                pooling=pooling,
                fully=fully,
                max_iterations=max_iterations,
                epsilon=epsilon,
            )
            for _ in range(num_reservoirs)
        ])
        self.readouts = nn.ModuleList([
            Readout(num_features=reservoir.out_features, num_targets=out_channels)
            for reservoir in self.reservoirs
        ])
        self.criterion = nn.CrossEntropyLoss()

        self.C = out_channels
        self.lr = lr
        self.weight_decay = weight_decay
        self.ood_in_val = ood_in_val
        self.readout_regularization = readout_regularization
        self._readouts_fitted = False

    def _build_reservoir(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        spectral_radius: float,
        input_scaling: float,
        leakage: float,
        bias: bool,
        pooling: Optional[str],
        fully: bool,
        max_iterations: Optional[int],
        epsilon: Optional[float],
    ):
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if pooling not in (None, "none"):
            raise ValueError(
                "GraphESN pooling must be None for node-level classification. "
                "Non-None pooling returns graph-level embeddings and is not compatible "
                "with train/val/test node masks."
            )

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
        bias_initializer = initializer("uniform", scale=input_scaling) if bias else None
        reservoir.initialize_parameters(
            recurrent=initializer("uniform", rho=spectral_radius),
            input=initializer("uniform", scale=input_scaling),
            bias=bias_initializer,
            leakage=leakage,
        )
        reservoir.requires_grad_(False)
        return reservoir

    def _split_mask(self, batch, split):
        if hasattr(batch, "batch_size") and hasattr(batch, "n_id"):
            mask = torch.zeros(batch.num_nodes, dtype=torch.bool, device=batch.x.device)
            mask[:batch.batch_size] = True
            return mask
        return getattr(batch, f"{split}_mask")

    def _f1_score(self, preds, labels):
        return F1Score(task="multiclass", num_classes=self.C).to(preds.device)(preds, labels)

    def _auroc_score(self, scores, targets):
        return AUROC(task="binary").to(scores.device)(scores, targets)

    @staticmethod
    def _calculate_shannon_entropy(probs: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
        probs_clipped = torch.clamp(probs, min=epsilon)
        return -torch.sum(probs * torch.log2(probs_clipped), dim=-1)

    def encode_member(self, data, member_idx: int):
        with torch.no_grad():
            return self.reservoirs[member_idx](data.edge_index, data.x)

    def member_logits(self, data):
        logits = []
        for reservoir, readout in zip(self.reservoirs, self.readouts):
            with torch.no_grad():
                embeddings = reservoir(data.edge_index, data.x)
                logits.append(readout(embeddings))
        return torch.stack(logits, dim=0)

    def forward(self, data):
        return self.member_logits(data).mean(dim=0)

    def _fit_readouts(self):
        if self._readouts_fitted:
            return
        train_loader = self.trainer.train_dataloader
        for member_idx, readout in enumerate(self.readouts):
            def data_iterator():
                for batch in train_loader:
                    batch = batch.to(self.device)
                    train_mask = self._split_mask(batch, "train")
                    if not train_mask.any():
                        continue
                    embeddings = self.encode_member(batch, member_idx)[train_mask]
                    targets = batch.y[train_mask]
                    yield embeddings.detach(), targets.detach()

            readout.fit(
                data=data_iterator(),
                regularization=self.readout_regularization,
            )
        self._readouts_fitted = True

    def on_train_epoch_start(self):
        self._fit_readouts()

    def _member_probabilities(self, data):
        return F.softmax(self.member_logits(data), dim=-1)

    def _ensemble_outputs(self, data):
        stacked_probs = self._member_probabilities(data)
        mean_probs = torch.mean(stacked_probs, dim=0)
        q_L = torch.min(stacked_probs, dim=0).values
        q_U = torch.max(stacked_probs, dim=0).values
        return q_L, q_U, stacked_probs, mean_probs

    def _uncertainty_outputs(self, stacked_probs):
        mean_probs = torch.mean(stacked_probs, dim=0)
        q_L = torch.min(stacked_probs, dim=0).values
        q_U = torch.max(stacked_probs, dim=0).values

        TU_credal, AU_credal, EU_credal = compute_uncertainties(q_L, q_U)
        TU_classic = self._calculate_shannon_entropy(mean_probs)
        individual_entropies = self._calculate_shannon_entropy(stacked_probs)
        AU_classic = torch.mean(individual_entropies, dim=0)
        EU_classic = TU_classic - AU_classic

        return {
            "TU_credal": TU_credal,
            "AU_credal": AU_credal,
            "EU_credal": EU_credal,
            "TU_classic": TU_classic,
            "AU_classic": AU_classic,
            "EU_classic": EU_classic,
        }

    def training_step(self, batch, batch_idx):
        stacked_probs = self._member_probabilities(batch)
        mean_probs = torch.mean(stacked_probs, dim=0)
        train_mask = self._split_mask(batch, "train")
        mean_probs_train = mean_probs[train_mask]
        y_train = batch.y[train_mask]
        labels = torch.argmax(y_train, dim=1)

        loss = F.nll_loss(torch.log(mean_probs_train.clamp_min(1e-12)), labels)
        preds = torch.argmax(mean_probs_train, dim=1)
        acc = (preds == labels).float().mean()
        f1 = self._f1_score(preds, labels)

        num_train_nodes = train_mask.sum()
        self.log("train_loss", loss, batch_size=num_train_nodes)
        self.log("train_acc", acc, batch_size=num_train_nodes)
        self.log("train_f1", f1, batch_size=num_train_nodes)
        return loss

    def on_validation_epoch_start(self):
        self._validation_outputs = []

    def validation_step(self, batch, batch_idx):
        stacked_probs = self._member_probabilities(batch)
        mean_probs = torch.mean(stacked_probs, dim=0)
        val_mask = self._split_mask(batch, "val")
        mean_probs_val = mean_probs[val_mask].detach()
        stacked_probs_val = stacked_probs[:, val_mask, :].detach()
        y_val = batch.y[val_mask].detach()

        id_mask = y_val.sum(axis=1) == 1
        if id_mask.any():
            labels = torch.argmax(y_val[id_mask], dim=1)
            probs_id = mean_probs_val[id_mask]
            loss = F.nll_loss(torch.log(probs_id.clamp_min(1e-12)), labels)
            preds = torch.argmax(probs_id, dim=1)
            acc = (preds == labels).float().mean()
            self.log("val_loss", loss, prog_bar=True, batch_size=id_mask.sum(), on_step=False, on_epoch=True)
            self.log("val_acc", acc, batch_size=id_mask.sum(), on_step=False, on_epoch=True)
        else:
            loss = None
            labels = torch.empty(0, dtype=torch.long, device=self.device)
            preds = torch.empty(0, dtype=torch.long, device=self.device)

        output = {
            "labels": labels.detach(),
            "preds": preds.detach(),
        }

        if self.ood_in_val:
            uncertainty = self._uncertainty_outputs(stacked_probs_val)
            output.update({
                "ood_targets": (1 - y_val.sum(axis=1)).long().detach(),
                **{name: value.detach() for name, value in uncertainty.items()},
            })

        self._validation_outputs.append(output)
        return loss

    def on_validation_epoch_end(self):
        if not self._validation_outputs:
            return

        labels = torch.cat([o["labels"] for o in self._validation_outputs], dim=0)
        preds = torch.cat([o["preds"] for o in self._validation_outputs], dim=0)
        if labels.numel() > 0:
            self.log("val_f1", self._f1_score(preds, labels), prog_bar=True)

        if self.ood_in_val and "EU_credal" in self._validation_outputs[0]:
            ood_targets = torch.cat([o["ood_targets"] for o in self._validation_outputs], dim=0)
            for family in ("credal", "classic"):
                for uncertainty_name in ("EU", "AU", "TU"):
                    key = f"{uncertainty_name}_{family}"
                    scores = torch.cat([o[key] for o in self._validation_outputs], dim=0)
                    self.log(
                        f"val_auroc_{uncertainty_name}_{family}",
                        self._auroc_score(scores, ood_targets),
                        prog_bar=(key == "EU_credal"),
                    )
            self.log(
                "val_auroc",
                self._auroc_score(
                    torch.cat([o["EU_credal"] for o in self._validation_outputs], dim=0),
                    ood_targets,
                ),
                prog_bar=True,
            )

        self._validation_outputs.clear()

    def on_test_epoch_start(self):
        self._test_outputs = []

    def test_step(self, batch, batch_idx):
        stacked_probs = self._member_probabilities(batch)
        mean_probs = torch.mean(stacked_probs, dim=0)
        test_mask = self._split_mask(batch, "test")
        mean_probs_test = mean_probs[test_mask].detach()
        stacked_probs_test = stacked_probs[:, test_mask, :].detach()
        y_test = batch.y[test_mask].detach()

        uncertainty = self._uncertainty_outputs(stacked_probs_test)
        ood_targets = (1 - y_test.sum(axis=1)).long()

        id_mask = y_test.sum(axis=1) == 1
        labels = torch.argmax(y_test[id_mask], dim=1)
        preds = torch.argmax(mean_probs_test[id_mask], dim=1)

        self._test_outputs.append({
            "ood_targets": ood_targets.detach(),
            "labels": labels.detach(),
            "preds": preds.detach(),
            **{name: value.detach() for name, value in uncertainty.items()},
        })
        return uncertainty["EU_credal"]

    def on_test_epoch_end(self):
        if not self._test_outputs:
            return

        ood_targets = torch.cat([o["ood_targets"] for o in self._test_outputs], dim=0)
        for family in ("credal", "classic"):
            for uncertainty_name in ("EU", "AU", "TU"):
                key = f"{uncertainty_name}_{family}"
                scores = torch.cat([o[key] for o in self._test_outputs], dim=0)
                self.log(
                    f"test_auroc_{uncertainty_name}_{family}",
                    self._auroc_score(scores, ood_targets),
                )
        self.log(
            "test_auroc",
            self._auroc_score(
                torch.cat([o["EU_credal"] for o in self._test_outputs], dim=0),
                ood_targets,
            ),
        )

        labels = torch.cat([o["labels"] for o in self._test_outputs], dim=0)
        preds = torch.cat([o["preds"] for o in self._test_outputs], dim=0)
        if labels.numel() > 0:
            self.log("test_acc", (preds == labels).float().mean())
            self.log("test_f1", self._f1_score(preds, labels))

        self._test_outputs.clear()

    def configure_optimizers(self):
        return None
