# GPU etiquette (shared 8×A100 server)

- **Never use more than 2 GPUs at the same time, in total** — count every process you have running
  (sweep agents, `cgnn run`, notebooks), not per job. This is a hard limit set by the user.
- Before launching anything on GPU run `nvidia-smi` (or `cgnn doctor`) and pick idle GPUs; other users'
  jobs appear there. Never kill, renice or otherwise touch processes you did not start.
- Always pin: `CUDA_VISIBLE_DEVICES=<id> ...`. `cgnn run/sweep` pin one idle GPU automatically when the
  variable is unset (`--gpus N`, N <= 2) and refuse `devices > 2` or `devices: auto` (Lightning's `auto`
  would grab every visible GPU and start DDP).
- Tests, smoke runs and anything on the `synthetic` dataset: CPU (`--cpu`, `CUDA_VISIBLE_DEVICES=""`).
- Full-graph inference on Reddit2 (23M edges x 602 features) needs ~52 GB: post-hoc methods on Reddit2
  must run on CPU (`--cpu`), as in the original experiments. Patents (2.9M nodes) training is mini-batched.
- Squirrel/Chameleon with SAGE need >32 GB GPU memory (dense 2089-dim messages): use `--cpu`.
- Limit CPU threads for background jobs you start on the shared machine (e.g. `OMP_NUM_THREADS=16`).
- Long jobs: run in the background with a log file and poll it; release GPUs when done (the runner frees
  memory between sweep runs).
