"""Central path definitions for soil_project.

Every script and notebook should import paths from here instead of hardcoding
absolute paths. This makes the project relocatable and — critically — gives each
run its own output directory so that two concurrent runs cannot overwrite each
other's checkpoints.

Usage
-----
    import paths

    ckpt = torch.load(paths.SIM_DATA / "sparse_smooth_h.pt")
    torch.save(h_old, paths.run_dir() / f"h_pred_itteration_{n+1}.pt")

Run identity
------------
Output goes to  Model_solution/<RUN_ID>/  where RUN_ID defaults to a timestamp.
Set it explicitly to label an experiment or to resume into the same folder:

    RUN_ID=loam_smooth_dataassim python SPINN_uniform.py
"""

import os
from datetime import datetime
from pathlib import Path

# Repo root = directory containing this file. Never hardcode above this.
ROOT = Path(__file__).resolve().parent

# ---- Input data ----
SIM_DATA = ROOT / "simulation_data"
DATASET = ROOT / "dataset"
SOILDATA = ROOT / "soildata"

# ---- Outputs ----
CHECKPOINTS = ROOT / "Model_solution"
SOLUTIONS = ROOT / "Solutions"
TIMESTEP_TEST = ROOT / "Timestep_test"
MODELS_2DR = ROOT / "2DR_models"
RESULTS = ROOT / "results"
OUTPUT = ROOT / "output"

# ---- Run identity ----
RUN_ID = os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")


_run_dir_ready = False


def run_dir(create: bool = True) -> Path:
    """Per-run checkpoint directory: Model_solution/<RUN_ID>/.

    Isolating each run under RUN_ID is what prevents two simultaneous jobs from
    opening the same checkpoint file and deadlocking on the Lustre write lock.

    Safe to call in a hot loop: the mkdir happens once, so repeated calls cost
    no Lustre metadata operations.
    """
    global _run_dir_ready
    d = CHECKPOINTS / RUN_ID
    if create and not _run_dir_ready:
        d.mkdir(parents=True, exist_ok=True)
        _run_dir_ready = True
    return d


def solution_path(name: str) -> Path:
    """Path for a named final solution tensor, namespaced by run."""
    SOLUTIONS.mkdir(parents=True, exist_ok=True)
    return SOLUTIONS / f"{name}__{RUN_ID}.pt"
