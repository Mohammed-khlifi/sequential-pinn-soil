import numpy as np
import time
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import json, os, subprocess, datetime, shutil, argparse, yaml
import subprocess, sys
from typing import Union, Sequence
import pandas as pd
from Training import SPINNTrainer_2D, Trainer
from Models import PINNsformer,PINN, Attention, RC_PINN, DSSE_Net
from utils.utilities import *
"""from FDM import FDM
from Training import SPINNTrainer, Trainer
from Models import PINNsformer,PINN, Attention, RC_PINN
from utils.utilities import *"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# ---------------- config (yaml) ----------------
ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml")
args = ap.parse_known_args()[0]
with open(args.config) as f:
    cfg = yaml.safe_load(f)


Nx = cfg["grid"]["Nx"]
Nz = cfg["grid"]["Nz"]
Nt = cfg["grid"]["Nt"]

precision = torch.float64


x = torch.linspace(0, 1, Nx , requires_grad=False , device = device, dtype=precision)
z = torch.linspace(0, 1, Nz, requires_grad=False , device = device, dtype=precision)
t = torch.linspace(0, cfg["grid"]["t_max"], Nt, requires_grad=False , device = device, dtype=precision)
dt = t[1] - t[0]
T,Z,X = torch.meshgrid(t,z,x, indexing='ij')

subdomains = [(Nt, dt)]



soil_parameters = {
    'Sand': {'theta_s': 0.43, 'theta_r': 0.045, 'alpha': 14.5, 'n': 2.68 , 'Ks': 3600 * 8.25e-5},
    'Loam': {'theta_s': 0.43, 'theta_r': 0.078, 'alpha': 3.6, 'n': 1.56 , 'Ks': 3600 * 2.889e-6},
    'reda': {'theta_s': 0.45, 'theta_r': 0.15, 'alpha': 1/200, 'n': 1.56 , 'Ks': 3600 * 1e-5},
}


soil = cfg["soil"]
params = soil_parameters[soil]

hb = torch.tensor(cfg["bc"]["hb"], device=device , dtype=precision)
hg = torch.tensor(cfg["bc"]["hg"], device=device, dtype=precision)
xt_one = torch.ones((Nt ,Nx), device=device, dtype=precision)
zt_one = torch.ones((Nt, Nz), device=device, dtype=precision)
zx_one = torch.ones((Nz, Nx), device=device, dtype=precision)

# Detach boundary constants to avoid reusing the same graph each epoch
Xb = X.detach()
h_top = hb * xt_one
x_infiltration_range = cfg["bc"]["x_infiltration_range"] # top boundary lenght is 1m, water infiltrates only in the first quarter
h_top[:,:x_infiltration_range] = 0.0
h_bottom = hb * xt_one
h_left   = hb * zt_one
h_right  = hb * zt_one

h_top = h_top.detach()
h_bottom = h_bottom.detach()
h_left = h_left.detach()
h_right = h_right.detach()

flux_top = 0 * torch.ones_like(h_top)
flux_top[:, :x_infiltration_range] = params['Ks']


flux_bottom = torch.zeros_like(h_bottom)
flux_left = torch.zeros_like(h_left)
flux_right = torch.zeros_like(h_right)

boundary_types = {
    'top': 'Neumann',
    'bottom': 'Neumann',
    'left': 'Neumann',
    'right': 'Neumann'
}

boundary_values = {
    'top': flux_top,
    'bottom': flux_bottom,
    'left': flux_left,
    'right': flux_right
}

h_top_rcpinn = (boundary_types['top'], boundary_values['top'])
h_bottom_rcpinn = (boundary_types['bottom'], boundary_values['bottom'])
h_left_rcpinn = (boundary_types['left'], boundary_values['left'])
h_right_rcpinn = (boundary_types['right'], boundary_values['right'])

h_init_true = hb * zx_one
h_init_true[0 , :] = h_top[0,:]
h_init_true[-1 , :] = h_bottom[0,:]
h_init_true[:, 0] = h_left[0,:]
h_init_true[:, -1] = h_right[0,:]




if __name__ == "__main__":
    k = Nt
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)


    #MLP(layers= layers , activation = nn.Tanh())
    """Models = [RC_PINN(
        conv_layers=[(3, 16, 5), (16, 32, 5), (32, 64, 5)],
        linear_layers=[(64, 128),(128,512) ,(512 ,1028) , (1028, 1028), (1028,512) ,(512,128) ,(128, 1)],
        activation=nn.ReLU()).to(device).to(precision) for _ in range(k)]"""
    #Models = [DSConvNet(in_channels=3 , width=128, depth=6, use_se=True, groups=min(8, 4), dropout=0.0).to(device).to(precision) for _ in range(k)]
    in_channels = cfg["model"]["in_channels"]
    width = cfg["model"]["width"]
    depth = cfg["model"]["depth"]
    dropout = cfg["model"]["dropout"]
    groups = cfg["model"]["groups"]
    Models = [DSSE_Net(in_channels=in_channels , width=width, depth=depth, dropout=dropout, groups=groups).to(device).to(precision) for _ in range(k)]

    lr_first = cfg["optim"]["lr_first"]
    lr_others = cfg["optim"]["lr_others"]
    weight_decay = cfg["optim"]["weight_decay"]
    optimizers = [
        torch.optim.AdamW(
            Models[i].parameters(),
            lr= lr_first , weight_decay=weight_decay
        ) if i == 0
        else torch.optim.AdamW(
            Models[i].parameters(),
            lr=lr_others , weight_decay=weight_decay
        )
        for i in range(k)
    ]




    step_size = cfg["optim"]["step_size"]
    gamma = cfg["optim"]["gamma"]
    schedulers = [
        torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
        for opt in optimizers
    ]



    K_fn     = lambda h, z: K_vg(h, z, params=params)
    theta_fn = lambda h, z: theta_vg(h, z, params=params)
    C_fn     = lambda h, z: C_vg(h, z, params=params)
    start = time.time()
    trainer = SPINNTrainer_2D(models = Models, optimizers = optimizers, schedulers = schedulers , K = K_fn, theta = theta_fn, C = C_fn , device = device)
    epochs_max = cfg["train"]["epochs_max"]
    epochs = cfg["train"]["epochs"]
    E_max = cfg["train"]["E_max"]
    tol = cfg["train"]["tol"]
    loss_min = cfg["train"]["loss_min"]
    loss_max = cfg["train"]["loss_max"]
    regulariztion = cfg["train"]["regularization"]
    adaptive_weights = cfg["train"]["adaptive_weights"]
    data_loss = cfg["train"]["data_loss"]
    H_PINN_hyb= trainer.train(T = T[:],
                          z = Z ,
                          x = X,
                          epochs_max = epochs_max,
                          epochs = epochs,
                          E_max = E_max,
                          tol = tol,
                          loss_min = loss_min,
                          loss_max = loss_max,
                          subdomains = subdomains,
                          adaptive_weights = adaptive_weights,
                          h_init = h_init_true,
                          h_bottom = h_bottom_rcpinn,
                          h_top = h_top_rcpinn,
                          h_right= h_right_rcpinn,
                          h_left = h_left_rcpinn,
                          regularization = regulariztion,
                          precision = precision,
                          data_loss = data_loss,
                         )

    end = time.time()
    print(f"execution time {end - start}")

    # ---------------- save results ----------------
    run_name = f"{soil.lower()}_nt{Nt-1}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    out_dir = os.path.join(cfg["output_dir"], run_name)
    os.makedirs(out_dir, exist_ok=True)

    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                           stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_hash = None

    meta = {
        "run_name": run_name,
        "git_commit": git_hash,
        "device": str(device),
        "precision": str(precision),
        "training_time_s": end - start,
        "soil_params": params,
        "config": cfg,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    shutil.copy(args.config, os.path.join(out_dir, "config.yaml"))

    # solution
    res = H_PINN_hyb.detach().cpu() if torch.is_tensor(H_PINN_hyb) else H_PINN_hyb
    torch.save(res, os.path.join(out_dir, "solution.pt"))

    # weights of the first model only
    torch.save(trainer.models[0].state_dict(), os.path.join(out_dir, "model_0.pt"))

    print(f"results saved in {out_dir}")

    subprocess.run([sys.executable, "Metrics_script.py", "--out_dir", out_dir], check=True)