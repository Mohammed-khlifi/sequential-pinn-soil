import torch, torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time





class HeatTrainer:
    def __init__(self, Model=None, D=1, grad="auto", device="cpu"):
        super().__init__()

        self.model = Model
        self.D = D
        self.device = device

        self.weights_tracker = []
        self.losses_tracker = []

        if grad == "auto":
            self.grad = self.grad_auto
        elif grad == "FDM":
            self.grad = self.grad_fd

    def train(self, epochs, Z, T, h_init_true, h_bound1, h_bound2, lr=0.01, adapt=False, scaler = None):
        mse = nn.MSELoss()

        # Adaptive weights (log parametrization)
        w1 = torch.tensor(0.0, requires_grad=True, device=self.device)
        w2 = torch.tensor(0.0, requires_grad=True, device=self.device)
        w3 = torch.tensor(0.0, requires_grad=True, device=self.device)
        w4 = torch.tensor(0.0, requires_grad=True, device=self.device)

        if adapt:
            opt = torch.optim.Adam(
                [{'params': self.model.parameters()},
                 {'params': [w1, w2, w3, w4]}], lr=lr)
        else:
            opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=3000, gamma=0.25)

        torch.autograd.set_detect_anomaly(True)

        progress_bar = tqdm(range(epochs), desc="Training", unit="epoch")
        start = time.time()

        try:
            for epoch in progress_bar:
                #if scaler is not None:
                #with torch.cuda.amp.autocast():
                        # Zero gradients
                opt.zero_grad()
                    
                #else:
                #pass

                # Model prediction
                h_int = self.model(Z, T)

                # 1st and 2nd derivatives
                h_t = self.grad(h_int, t=T)
                h_z = self.grad(h_int, z=Z)
                h_zz = self.grad(h_z, z=Z)

                # PDE residual
                R =( h_t - self.D * h_zz)[1:-1, 1:]
                L_phys = self.L2_loss(R, torch.zeros_like(R))

                # Initial condition
                h_init = h_int[1:-1, 0, :]
                L_init = self.L2_loss(h_init.squeeze(), h_init_true[1:-1].squeeze())

                # Boundary z=0
                h_bnd1 = h_int[0, :, :]
                L_bnd1 = self.L2_loss(h_bnd1.squeeze(), h_bound1)

                # Boundary z=L
                h_bnd2 = h_int[-1, :, :]
                L_bnd2 = self.L2_loss(h_bnd2.squeeze(), h_bound2)

                # Total loss with adaptive weights
                """loss = (
                    torch.exp(-w1) * L_phys
                    + torch.exp(-w2) * L_bnd1
                    + torch.exp(-w3) * L_bnd2
                    + torch.exp(-w4) * L_init
                    + 0.5 * (torch.abs(w1) + torch.abs(w2) +
                             torch.abs(w3) + torch.abs(w4))
                )"""
                
                loss =  L_bnd2 + L_bnd1 + L_phys + L_init

                loss.backward(retain_graph=True)
                opt.step()
                scheduler.step()

                self.weights_tracker.append(
                    {torch.exp(-w1), torch.exp(-w2), torch.exp(-w3), torch.exp(-w4)})
                self.losses_tracker.append({L_phys.item(), L_bnd1.item(), L_bnd2.item(), L_init.item()})

                progress_bar.set_postfix({"Loss": f"{loss.item():.4e}"})

        except Exception as e:
            print("Training stopped due to an error:", e)

        self.training_time = time.time() - start
        self.solution = h_int

    # ----------- GRADIENT METHODS --------------

    def grad_auto(self, outputs, z=None, t=None):
        """Autograd derivative helper."""
        if z is None:
            inputs = t
        else:
            inputs = z
        return torch.autograd.grad(
            outputs,
            inputs,
            grad_outputs=torch.ones_like(outputs),
            retain_graph=True,
            create_graph=True,
            allow_unused=True
        )[0]

    def grad_fd(self, function, z=None, t=None, order=1):
        """Finite difference gradient."""
        if order == 1:
            if z is not None:
                dz = z[1, 0] - z[0, 0]
                return (function[2:, 1:-1] - function[:-2, 1:-1]) / dz
            if t is not None:
                dt = t[0, 1] - t[0, 0]
                return (function[1:-1, 2:] - function[1:-1, :-2]) / (2 * dt)

        elif order == 2:
            if z is not None:
                dz = z[1, 0] - z[0, 0]
                return (function[2:, 1:-1] - 2 * function[1:-1, 1:-1] + function[:-2, 1:-1]) / dz**2
            if t is not None:
                dt = t[0, 1] - t[0, 0]
                return (function[1:-1, 2:] - 2 * function[1:-1, 1:-1] + function[1:-1, :-2]) / dt**2

        raise ValueError("Invalid FD computation")
    
    def L2_loss(self, pred, true):
        """Compute L2 loss between prediction and true values."""
        return (torch.sum((pred - true) ** 2))**0.5

    # ----------- GETTERS ----------------
    def get_solution(self):
        return self.solution

    def get_training_time(self):
        return self.training_time

    def get_losses(self):
        return self.losses_tracker

    def get_weights(self):
        return self.weights_tracker
