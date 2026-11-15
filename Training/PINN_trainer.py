import torch, torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time

class Trainer:
    def __init__(self , Model = None, theta = None , C = None , K = None ,grad = "auto" , device = "cpu"):
        super().__init__()
        
        self.model = Model 
        self.theta = theta
        self.C = C
        self.K = K
        self.device = device
        self.weights_tracker = []
        self.losses_tracker = []
        if grad == "auto" :
            self.grad = self.grad_auto
        elif grad == "FDM" :
            self.grad = self.grad_fd
        
    def train(self , epochs , Z , T , h_init_true , h_bound1, h_bound2 , lr = 0.01, adapt = False):
        
        mse = nn.MSELoss()
        w1 = torch.tensor(0.0, requires_grad =True, device = self.device)
        w2 = torch.tensor(0.0, requires_grad =True, device = self.device)
        w3 = torch.tensor(0.0, requires_grad =True, device = self.device)
        w4 = torch.tensor(0.0, requires_grad =True, device = self.device)
        if adapt :
            opt = torch.optim.Adam([{'params':self.model.parameters()},{'params':[w1, w2, w3, w4]}  ], lr = lr)   
        else:
            opt = torch.optim.Adam(self.model.parameters(), lr = lr)    
            
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=3000, gamma=0.25)
        torch.autograd.set_detect_anomaly(True) # for debugging
        # Training loop
        progress_bar = tqdm(range(epochs), desc="Training", unit="epoch")
        start = time.time()
        for epoch in progress_bar:
            opt.zero_grad()

            # predict solution
            h_int = self.model(Z, T)
            
            #------ derivatives calculations --------
            h_t   = self.grad(h_int, t=T)          # ∂h/∂t
            if self.C != None:
                #print(h_t.shape ,self.C(h_int).shape )
                theta_t = self.C(h_int , z = Z).squeeze() * h_t
            else:
                theta_int = self.theta(h_int , z =Z)        # θ(h)
                theta_t   = self.grad(theta_int, t=T)  # ∂θ/∂t
            h_z   = self.grad(h_int, z=Z)          # ∂h/∂z
            #h_zz = grad(h_z, Z)             # ∂²h/∂z²
            K_int = self.K(h_int , Z).squeeze()      # K(h)
            #K_z = grad(K_int, Z)            # ∂K/∂z
            flux = K_int*(h_z -1)
            flux_z = self.grad(flux, z = Z)              # ∂(K(h)(∂h/∂z + 1))/∂z
            
            # physics residual
            R = theta_t-flux_z
            L_phys = mse(R, torch.zeros_like(R))  # physics loss

            # initial condition loss
            h_init = h_int[:,0,:]
            L_init = mse(h_init.squeeze(), h_init_true[0:].squeeze()) # initial condition loss

            # Top conditions loss
            h_bnd1 = h_int[0,1:,:]
            L_bnd1  = mse(h_bnd1.squeeze(),  h_bound1[1:]) # boundary condition loss at z=0
            
            # Bottom conditions loss
            h_bnd2 = h_int[-1,:,:]
            L_bnd2  = mse(h_bnd2.squeeze(), h_bound2[:]) # boundary condition loss at z=L
            L_bnd2 = mse(h_z[-1 , :].squeeze(), torch.zeros_like(h_z[-1 , :]))
            
            # sum of every loss
            loss = torch.exp(-w1)*L_phys + torch.exp(-w2)*L_bnd1 + torch.exp(-w3)*L_bnd2 + torch.exp(-w4)*L_init + 0.5  * (torch.abs(w1) +torch.abs(w2) +torch.abs(w3) +torch.abs(w4) ) #+ 10 * L_data
            
            #backpropagation and optimmization 
            loss.backward(retain_graph = False ) 
            opt.step()
            scheduler.step()

            self.weights_tracker.append({torch.exp(-w1) ,torch.exp(-w2), torch.exp(-w3) , torch.exp(-w4)})
            self.losses_tracker.append({L_phys , L_bnd1 , L_bnd2 , L_init})
            
            postfix = {"Loss": f"{loss.item():.4e}"}
            progress_bar.set_postfix(postfix)

        self.training_time = time.time() - start
        self.solution = h_int
        
    def grad_auto(self, outputs,  z = None , t = None ):
        """
        Helper for auto-diff
        """
        if z == None:
            inputs = t
        else:
            inputs = z
        derivation = torch.autograd.grad(outputs, inputs,
            grad_outputs=torch.ones_like(outputs),
            retain_graph= True, create_graph =True, allow_unused =True
        )[0]
        return derivation
    def grad_fd(delf, function , z = None , t = None , order = 1):
        """
        Helper for finite difference
        """
        if order == 1 :
          if z!= None:
              dz = z[1 , 0] - z[0 , 0]
              function_dz = (function[2: , 1:-1] - function[:-2 ,1:-1]) / dz
              return function_dz
          elif t != None:
              dt = t[0 ,1] - t[0 ,0]
              function_dt = (function[1:-1, 2:] - function[1:-1, :-2]) / 2*dt
              return function_dt
        elif order == 2:
          if z!= None:
              dz = z[1 , 0] - z[0 , 0]
              function_dzz = (function[2: , 1:-1] - 2*function[1:-1 ,1:-1] + function[:-2 ,1:-1]) / dz**2
              return function_dzz
          elif t != None:
              dt = t[0 ,1] - t[0 ,0]
              function_dtt = (function[1:-1, 2:] - 2*function[1:-1, 1:-1] + function[1:-1, :-2]) / dt**2
              return function_dtt

    
    def solution(self):
        return self.solution 
    
    def training_time(self):
        return self.training_time
        
    def losses_tracker(self):
        return self.losses_tracker
    
    def weights_tracker(self):
        return self.weights_tracker