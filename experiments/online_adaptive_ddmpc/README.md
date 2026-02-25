# Online Adaptive Hankel DDMPC Experiment

This experiment extends the repository baseline with an **online adaptive data dictionary**:

- Maintains a FIFO window of the latest input-output samples.
- Rebuilds the Hankel matrix every fixed number of MPC cycles.
- Applies exponential column weighting with a forgetting factor.
- Runs robust DDMPC with input-rate penalty and saves a summary figure.

## Run

```bash
python experiments/online_adaptive_ddmpc/run_online_adaptive.py
```

## Outputs

- `experiments/online_adaptive_ddmpc/results/online_adaptive_summary.png`
- `experiments/online_adaptive_ddmpc/results/metrics_online_adaptive.npz`
