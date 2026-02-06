# Multimodal-feature-screening-via-autoencoder

Simulation and real data analysis.

## Project framework

This repository now includes a structured framework to support:

- Multimodal regression feature screening simulations.
- Multimodal classification feature screening simulations.
- Multimodal clustering feature screening simulations.
- Side-by-side comparisons between the proposed autoencoder-based method and baseline methods.

### Directory layout

```
./src
  /experiments      # orchestration for simulation suites
  /methods          # proposed method + baseline placeholders
  /simulations      # regression/classification/clustering tasks
  /utils            # metrics and helpers
./scripts           # runnable entrypoints
./configs           # configuration files (to be added)
./data              # datasets (to be added)
./results           # simulation outputs (to be added)
```

### Quick start

```bash
python scripts/run_simulations.py
```

The default script runs simulations in parallel across methods. You can toggle this
behavior via the `parallel` argument in `run_simulation_suite`.

For repeated Monte Carlo-style simulations, set `num_repeats` in each task config and
enable `parallel_repeats=True` to run the repeats concurrently.

## Next steps

- Fill in the proposed autoencoder model and feature screening logic.
- Add concrete simulation data generation for each task.
- Expand baseline methods for fair comparison per task.
- Add real data classification pipeline once the dataset is available.
