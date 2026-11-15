import gc
import random
from khlifox.soil_project.Infitration_training_original import *
import numpy as np
import torch
from soil_project.Models import DSSE_Net
from soil_project.Training.SPINN_Trainer_2D import SPINNTrainer_2D
from utils.utilities import *


NTS = [401 ]
Time = []
def seed_everything(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(seed = 42)
soil_parameters = {
    'Sand': {'theta_s': 0.43, 'theta_r': 0.045, 'alpha': 14.5, 'n': 2.68 , 'Ks': 3600 * 8.25e-5},
    'Loam': {'theta_s': 0.43, 'theta_r': 0.078, 'alpha': 3.6, 'n': 1.56 , 'Ks': 3600 * 2.889e-6},
    'reda': {'theta_s': 0.45, 'theta_r': 0.15, 'alpha': 1/200, 'n': 1.56 , 'Ks': 3600 * 1e-5},
}


soil = 'Loam'  # default soil type
params = soil_parameters[soil]

for Nt in NTS:
    Nx = 101
    Nz = 101

    precision = torch.float64


    x = torch.linspace(0, 1, Nx , requires_grad=False , device = device, dtype=precision)
    z = torch.linspace(0, 1, Nz, requires_grad=False , device = device, dtype=precision)
    t = torch.linspace(0,30, Nt, requires_grad=False , device = device, dtype=precision)
    dt = t[1] - t[0]
    T,Z,X = torch.meshgrid(t,z,x, indexing='ij')

    subdomains = [(Nt, dt)]


    hb = torch.tensor(-10.0, device=device , dtype=precision)
    hg = torch.tensor(2.0, device=device, dtype=precision)
    xt_one = torch.ones((Nt ,Nx), device=device, dtype=precision)
    zt_one = torch.ones((Nt, Nz), device=device, dtype=precision)
    zx_one = torch.ones((Nz, Nx), device=device, dtype=precision)

    # Detach boundary constants to avoid reusing the same graph each epoch
    Xb = X.detach()

    h_top = torch.log(
        torch.exp(hb / hg)
        + (1.0 - torch.exp(hb / hg) * torch.sin(torch.pi * Xb[:, 0, :] / 1.0))
    )
    h_top = hb * xt_one
    h_top[:,:26] = 0.0
    h_bottom = hb * xt_one
    h_left   = hb * zt_one
    h_right  = hb * zt_one

    h_top = h_top.detach()
    h_bottom = h_bottom.detach()
    h_left = h_left.detach()
    h_right = h_right.detach()

    flux_top = 0 * torch.ones_like(h_top)
    flux_top[:, :26] = (3600 * 2.889e-6)/2


    flux_bottom = torch.zeros_like(h_bottom)
    flux_left = torch.zeros_like(h_left)
    flux_right = torch.zeros_like(h_right)

    h_top_rcpinn = ('Neumann', flux_top)
    h_bottom_rcpinn = ('Neumann', flux_bottom)
    h_left_rcpinn = ('Neumann', flux_left)
    h_right_rcpinn = ('Neumann', flux_right)

    h_init_true = hb * zx_one
    h_init_true[0 , :] = h_top[0,:]
    h_init_true[-1 , :] = h_bottom[0,:]
    h_init_true[:, 0] = h_left[0,:]
    h_init_true[:, -1] = h_right[0,:]



    k = Nt
    seed_everything(seed = 42)
    Models = [DSSE_Net(in_channels=3 , width=64, depth=6, dropout=0.0, groups=16).to(device).to(precision) for _ in range(k)]
    optimizers = [
        torch.optim.AdamW(
            Models[i].parameters(),
            lr= 1e-3 , weight_decay=1e-4
            
        ) if i == 0 
        else torch.optim.AdamW(
            Models[i].parameters(),
            lr=1e-4 , weight_decay=1e-4
        )
        for i in range(k)
    ]





    schedulers = [
        torch.optim.lr_scheduler.StepLR(opt, step_size=5000, gamma=0.95)
        for opt in optimizers
    ]


    torch.cuda.synchronize()
    start = time.perf_counter()
    seed_everything(seed = 42)

    K_fn     = lambda h, z: K_vg(h, z, params=params)
    theta_fn = lambda h, z: theta_vg(h, z, params=params)
    C_fn     = lambda h, z: C_vg(h, z, params=params)
    trainer = SPINNTrainer_2D(models = Models, optimizers = optimizers, schedulers = schedulers , K = K_fn, theta = theta_fn, C = C_fn , device = device)

    H_PINN_hyb= trainer.train(T = T[:],
                        z = Z ,
                        x = X,
                        epochs_max = 6000,
                        epochs = 1000,
                        E_max = 20000,
                        tol = 1e-40,
                        loss_min = 1e-2,
                        loss_max = 1e-3,
                        subdomains = subdomains,
                        adaptive_weights = False,
                        h_init = h_init_true,
                        h_bottom = h_bottom_rcpinn, 
                        h_top = h_top_rcpinn, 
                        h_right= h_right_rcpinn,
                        h_left = h_left_rcpinn,
                        regularization = 0.5, 
                        data_loss = True,
                        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    Time.append(elapsed)
    Models_training_time = trainer.Total_models_training_times
    print(f"execution time {elapsed}")
    print(f"Total models training time: {Models_training_time}")

    torch.save(
        H_PINN_hyb,  # ← What to save
        f"Data_assimilation_test/solutionSM(loam local right nt={Nt}).pt"
    )

    """torch.save(
        H_PINN_hyb,  # ← What to save
        str(paths.TIMESTEP_TEST / f"solution2(loam local right nt={Nt}).pt")
    )"""

    del Models, optimizers, schedulers, trainer, H_PINN_hyb
    gc.collect()
    torch.cuda.empty_cache()

"""torch.save(Time, str(paths.TIMESTEP_TEST / "timing_list4.2.pt"))
torch.save(Models_training_time, str(paths.TIMESTEP_TEST / "models_training_time4.2.pt"))"""
