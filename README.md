# sequential-pinn-soil

Sequential (time-stepping) physics-informed neural networks for 2D water
infiltration in unsaturated soil, governed by the Richards equation with
van Genuchten–Mualem soil properties.

## Layout

| Path | Contents |
|---|---|
| `Infitration_training.py` | YAML-driven training run: builds the grid and boundary conditions, trains one model per time step, saves `solution.pt`, `model_0.pt`, `meta.json` and the config, then runs the metrics script |
| `Metrics_script.py` | Compares a run's final pressure head and water content with a reference solution; writes `metrics.json` |
| `configs/` | Run configurations (`test_config.yaml`) |
| `Training/` | Trainers, including `SPINNTrainer_2D` |
| `Models/` | Network architectures (`DSSE_Net`, `RC_PINN`, `PINN`, …) |
| `utils/` | Soil hydraulic functions (`theta_vg`, `K_vg`, `C_vg`) and helpers |
| `Timestep_test.py`, `Data_assimilation _test.py` | Experiments built on the same trainer |

## Usage

```bash
python -u Infitration_training.py --config configs/test_config.yaml
```

Set `output_dir` in the config to where run folders should be written.
The metrics step needs a reference solution file (not included in this repository).
