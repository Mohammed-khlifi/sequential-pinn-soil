import numpy as np
import time
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import json, os, subprocess, datetime, shutil, argparse, yaml
from typing import Union, Sequence
import pandas as pd
from pathlib import Path
from Training import SPINNTrainer_2D, Trainer
from Models import PINNsformer,PINN, Attention, RC_PINN, DSSE_Net
from utils.utilities import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# ---------------- config (yaml) ----------------
ap = argparse.ArgumentParser()
ap.add_argument("--out_dir", default=None)
args = ap.parse_known_args()[0]

run_dir = args.out_dir if args.out_dir else "runs/test_run/loam_nt50_20261115_181131"
solution_path = Path(run_dir) / "solution.pt"


def load_pressure_file(file_path):
    """
    Load a single pressure file and return structured data.
    
    Parameters:
    -----------
    file_path : str
        Path to the pressure_*.txt file
        
    Returns:
    --------
    dict with keys:
        'x': 1D array of x coordinates
        'y': 1D array of y coordinates (depth/elevation)
        'z': 2D array of pressure head values [y, x]
        'file_name': name of the file
    """
    # Load raw data
    data = np.loadtxt(file_path)
    
    # Extract columns
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    
    # Get unique values
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    
    # Reshape into 2D grid
    z_grid = z.reshape(len(y_unique), len(x_unique))
    
    return {
        'x': x_unique,
        'y': y_unique,
        'z': z_grid,
        'file_name': Path(file_path).stem
    }

# Usage
pressure_data = load_pressure_file('final_pressure.txt')

# Access the data
x_coords = pressure_data['x']        # Shape: (Nx,)
y_coords = pressure_data['y']        # Shape: (Ny,)
pressure_grid = pressure_data['z']   # Shape: (Ny, Nx)

print(f"Grid dimensions: {pressure_grid.shape}")
print(f"Pressure range: [{pressure_grid.min():.4f}, {pressure_grid.max():.4f}] m")

soil_parameters = {
    'Sand': {'theta_s': 0.43, 'theta_r': 0.045, 'alpha': 14.5, 'n': 2.68 , 'Ks': 3600 * 8.25e-5},
    'Loam': {'theta_s': 0.43, 'theta_r': 0.078, 'alpha': 3.6, 'n': 1.56 , 'Ks': 3600 * 2.889e-6},
    'reda': {'theta_s': 0.45, 'theta_r': 0.15, 'alpha': 1/200, 'n': 1.56 , 'Ks': 3600 * 1e-5},
}


#H_sand = torch.load(f"Timestep_test/solution4(loam local right nt=401).pt",map_location=device)
H_sand = torch.load(solution_path,map_location=device)

h_pred = H_sand[:, -1,:].squeeze()

absolute_error = np.abs(h_pred.to(torch.float32).detach().cpu().numpy() - pressure_grid)
theta_error = np.abs(theta_vg(h_pred, params=soil_parameters['Loam']).detach().cpu().numpy() - theta_vg(torch.tensor(pressure_grid), params=soil_parameters['Loam']).detach().cpu().numpy())




mse_error = F.mse_loss(h_pred.to(torch.float32).detach().cpu(), torch.tensor(pressure_grid, dtype=torch.float32).to(torch.float32))
theta_mse_error = F.mse_loss(theta_vg(h_pred, params=soil_parameters['Loam']).to(torch.float32).detach().cpu(), theta_vg(torch.tensor(pressure_grid), params=soil_parameters['Loam']).to(torch.float32))

print(f"Mean absolute error in pressure head: {absolute_error.mean():.4f} ")
print(f"Mean absolute error in theta: {theta_error.mean():.4f}")

print(f"Mean squared error in pressure head: {mse_error.item()} m²")
print(f"Mean squared error in theta: {theta_mse_error.item()}")

print(f"Absolute error range: [{absolute_error.min():.4f}, {absolute_error.max():.4f}] m")
print(f"Theta error range: [{theta_error.min():.4f}, {theta_error.max():.4f}] m")


metrics = {
    "Mean Absolute Error (Pressure Head)": absolute_error.mean(),
    "Mean Absolute Error (Theta)": theta_error.mean(),
    "Mean Squared Error (Pressure Head)": mse_error.item(),
    "Mean Squared Error (Theta)": theta_mse_error.item(),
    "Absolute Error Range": [absolute_error.min(), absolute_error.max()],
    "Theta Error Range": [theta_error.min(), theta_error.max()]
}
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)