import numpy as np
import time
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, Sequence
import pandas as pd



class SPINNTrainer_2D:

    def __init__(self, K : callable, C : callable, theta : callable, models : list, optimizers : list, schedulers : list , device='cpu'):

        ###### Hydrodynamic features ######
        self.K = K
        self.C = C
        self.theta = theta
        ###### neural networks models #####
        self.models = models
        self.device = device
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.loss_tracker = {i: [] for i in range(len(models))}  # to track losses for each model
        self.epochs_count = []


    def train(self,
              T,
              z ,
              x,
              epochs : int = 1000,
              epochs_max : int= 1e4,
              E_max : int = 2e4,
              tol : float =1e-15,
              loss_max : float = 1e-6,
              loss_min : float = 1e-4,
              subdomains : list= [] ,
              h_init = None ,
              h_top = ("Dirichlet", None) ,
              h_bottom =  ("Dirichlet", None) ,
              h_left = ("Dirichlet", None) ,
              h_right = ("Dirichlet", None) ,
              adaptive_weights = False,
              regularization = 0.05 ,
              data_loss = False,
              data_path = 'simulation_data/sparse_smooth_h.pt',
              form = "mixed",
              precision = torch.float32
    ):

        """
               train function : training PINN models sequentialy
               T : tenstor of size Nz,Nt containing vectors of time values at each time step,
               z : the space vzctor of size Nz
               epochs : number of epochs per time step
               epochs_max : maximum number of epochs per time step
               E_max : maximum number of epochs in the first time step
               tol : maximum difference between two solutions at two consicutive epochs
               loss_max : maximum loss value
               loss_min : lowest loss value
               subdomains : list of differente dt for different subdomains (nt : number of time steps in the subdomain , dt : time step in the sub domain
               h_init : initial solution value and condition
               h_top : a boundary condtion
               dz = space step
               adaptive_weights : bool value to activate the weights asinged to each loss value
        """

        Nt, Nz, Nx = T.shape
        t_local = T[:, 0, 0]
        h_old = h_init.unsqueeze(-1)
        self.solution = []
        self.solution.append(h_old)
        self.Total_models_training_times = 0

        Losses = []
        mse = nn.MSELoss()
        j=0
        nt_sub , dt_sub  = subdomains[j]

        if data_loss == True:
            ckpt       = torch.load(data_path)                                                                                                                                                                                                           
            h          = ckpt['noisy_sparse_h']                                                                                                                                                                                                                                             
            time_hours = ckpt['time_hours']
    

        #time itteration lopp
        for n in range(0,Nt -1):

            if data_loss == True:
                target = t_local[n+1].cpu().item() # hours — or any value you want
                idx        = torch.argmin(torch.abs(time_hours - target)).item()                                                                                                                                                                                                           
                h_at_t     = h[idx]   
                h_at_t = h_at_t.to(self.device).to(precision)
                print(f"Requested : {target:.6f} h, closest   : {time_hours[idx].item():.6f} h  (step {idx})")

            itteration_start_time = time.time()
            previous_h = h_old.clone()
            theta_old = self.theta(h= h_old.squeeze() , z=z[n])
            if n == 0 :
                w1,w2,w3,w4, w5 = self.adaptive_weights_init()
                E = E_max
                if adaptive_weights :
                    self.optimizers[n].add_param_group({
                    'params':       [w1,w2,w3,w4, w5],
                    'lr':           0.001     # can be different
                    })

            else :
                E = epochs
            print(f"itteration {n}")
            epoch = 0
            # Training loop
            while epoch < E:
                self.models[n].train()
                self.optimizers[n].zero_grad()

                # physics residual
                h_pred=  self.models[n](z[n], x[n] , h_old.squeeze() ).squeeze()

                theta_vec = self.theta(h_pred, z)
                # backward difference on t
                if n < nt_sub + 1 :
                    if form.lower() == "mixed":
                        theta_vec = self.theta(h_pred, z)
                        theta_t = (theta_vec - theta_old) / (dt_sub)
                    elif form.lower() == "head":
                      theta_t = self.C(h_pred , z) * ( (h_pred - h_old.squeeze() ) / (dt_sub))
                else :
                    nt_sub_c , dt_sub  = subdomains[j + 1]
                    nt_sub = nt_sub_c + nt_sub
                    print(nt_sub)
                    if form.lower() == "mixed":
                        theta_vec = self.theta(h_pred, z)
                        theta_t = (theta_vec - theta_old) / (dt_sub)
                    elif form.lower() == "head":
                      theta_t = self.C(h_pred , z) * ( (h_pred - h_old.squeeze() ) / (dt_sub))
                    j = j + 1


                #theta_t = self.C(h_pred, z) * ((h_pred - h_old.squeeze())/dt_sub)
                L_phys = self.physics_loss(h_pred, theta_t, z , x)

                if h_top[0] == "Dirichlet":
                    L_bnd1 = self.boundary_loss(h_pred[0 , :] ,h_top[1][n, :]) # top Dirichlet boundary conditions
                    #L_bnd1 =  self.boundarytop_loss(h_pred ,h_top[1][n,:],z , x , epoch)
                elif h_top[0] == "Neumann":
                    L_bnd1, q_in  = self.Neuman_boundary_loss(h_pred[: , :] , h_top[1][n, :],z , x, epoch, location = 'top') # top Neumann boundary conditions
                else :
                    raise TypeError("Type do not exist or not implemented")

                if h_bottom[0] == "Dirichlet":
                    L_bnd2 = self.boundary_loss(h_pred[-1 , :] ,h_bottom[1][n, :]) # Bottom Dirichlet boundary conditions
                elif h_bottom[0] == "Neumann":
                    L_bnd2, q_bot = self.Neuman_boundary_loss(h_pred[: , :] , h_bottom[1][n, :],z , x, epoch,  location ='bottom') # Bottom Neumann boundary conditions
                else :
                    raise TypeError("Type do not exist or not implemented")

                if h_left[0] == "Dirichlet":
                    L_bnd3 = self.boundary_loss(h_pred[: , 0] ,h_left[1][n,:]) # left Dirichlet boundary conditions
                elif h_left[0] == "Neumann":
                    L_bnd3, q_left = self.Neuman_boundary_loss(h_pred[: , :] , h_left[1][n,:],z , x, epoch,  location ='left') # left Neumann boundary conditions
                else :
                    raise TypeError("Type do not exist or not implemented")

                if h_right[0] == "Dirichlet":
                    L_bnd4 = self.boundary_loss(h_pred[: , -1] ,h_right[1][n,:]) # right Dirichlet boundary conditions
                elif h_right[0] == "Neumann":
                    L_bnd4, q_right = self.Neuman_boundary_loss(h_pred[:, :], h_right[1][n,:],z , x, epoch,  location ='right') # right Neumann boundary conditions
                else :
                    raise TypeError("Type do not exist or not implemented")

                                                                                                                                                                                                                                                       
                if data_loss == True:
                    loss_data = F.mse_loss(h_pred[::10, ::10].squeeze(), h_at_t.squeeze())

                if adaptive_weights :
                    if data_loss == True:
                        loss = torch.exp(-w1)*L_phys + torch.exp(-w2)*L_bnd1 + torch.exp(-w3)*L_bnd2*10 + torch.exp(-w4)*L_bnd3 + torch.exp(-w5)*L_bnd4 + loss_data + regularization*(torch.abs(w1)+torch.abs(w2)+torch.abs(w3)+torch.abs(w4)+torch.abs(w5))
                    else :
                        loss = torch.exp(-w1)*L_phys + torch.exp(-w2)*L_bnd1 + torch.exp(-w3)*L_bnd2*10 + torch.exp(-w4)*L_bnd3 + torch.exp(-w5)*L_bnd4 +regularization*(torch.abs(w1)+torch.abs(w2)+torch.abs(w3)+torch.abs(w4)+torch.abs(w5))
                else :

                    if data_loss == True:
                    
                        loss = L_phys + L_bnd1 + L_bnd2 + L_bnd3 + L_bnd4 + loss_data
                    else :
                        
                        loss = L_phys + L_bnd1 + L_bnd2 + L_bnd3 + L_bnd4

                #self.Loss_tracker(itteration= n, loss=loss)  # track the loss
                loss.backward(retain_graph = False)
                self.optimizers[n].step() ; self.schedulers[n].step() # update the model

                #mse(h_pred , previous_h.squeeze()) < tol 
                """if loss < loss_max :
                    print(f"epoch {epoch:4d}, itteration {n+1}  loss={loss:.2e} , variance={mse(h_pred.squeeze() , previous_h.squeeze()):.2e}")
                    break
                elif epoch == E - 1  and loss > loss_min and E < epochs_max:
                    E += 100"""
                L_bounds_tresh = 1e-4
                if n==0 :
                    if loss < 5*loss_max and L_phys < loss_max and L_bnd2 < L_bounds_tresh and L_bnd3 < L_bounds_tresh and L_bnd4 < L_bounds_tresh:
                        print(f"epoch {epoch:4d}, itteration {n+1}  loss={loss:.2e}, L_phys={L_phys:.2e}, variance={mse(h_pred.squeeze() , previous_h.squeeze()):.2e}")
                        break
                else :
                    if loss < loss_max :
                        print(f"epoch {epoch:4d}, itteration {n+1}  loss={loss:.2e} , variance={mse(h_pred.squeeze() , previous_h.squeeze()):.2e}")
                        break

                    elif epoch == E - 1  and loss > loss_min and E < epochs_max:
                        E += 100
                    #pass


                previous_h = h_pred.clone()
                epoch += 1


                if (epoch) %500 ==0:
                    print(f"epoch {(epoch):4d}  loss={loss:.2e} , L_phys={L_phys:.2e} , L_bnd1={L_bnd1:.2e}, L_bnd2={L_bnd2:.2e}, L_bnd1={L_bnd3:.2e}, L_bnd2={L_bnd4:.2e}")
                    if (epoch) %1000 ==0:
                      #print(f"weights: {w1 , w2, w3, w4 , w5}")
                      pass
            itteration_end_time = time.time()
            itteration_time = itteration_end_time - itteration_start_time
            self.Total_models_training_times += itteration_time
            print(f"Time for itteration {n+1} : {itteration_time:.2f} seconds")
            # update the head
            with torch.no_grad():
                self.models[n].eval()
                h_old = h_pred.detach().clone().unsqueeze(-1) # update the head
                self.solution.append(h_pred.unsqueeze(-1)) # store the head
                self.models[n+1].load_state_dict(self.models[n].state_dict()) # copy the model
               
                """if n == 0 :
                    torch.save(self.models[n].state_dict(), f"Model_solution/model_itteration_{n+1}.pt")

                torch.save(
                    h_old,  # ← What to save
                    f"Model_solution/h_pred_itteration_{n+1}.pt"
                )"""


            #self.epochs_counter(n , epoch)
        self.solution = torch.stack(self.solution , dim = 1)
        #self.solution = Solution
        return self.solution

    def remember_who_you_are(self, itteration,z,T,x):


        loss = 0
        h_old = self.solution[0]
        for i in range(itteration):
            #h_old = self.solution[i] #[i , : , -1].unsqueeze(-1)

            prediction = self.models[itteration](T[i], z[i], x[i] , h_old.squeeze() )

            loss += 0.5 * F.mse_loss(prediction.squeeze(), self.solution[i+1].squeeze())
            h_old = prediction.detach()

        return loss


    def physics_loss(self, h_pred , theta_t , z , x , k = 1):
        """
        physical loss calculation
        h_pred : the prediction given by the model
        theta_t : derivative of theta respect to time at h_pred
        """
        dz = (z[0, 1 ,0] - z[0 ,0, 0])
        dx = (x[0,0,1] - x[0,0,0])
        Kvec = self.K(h_pred , z).squeeze()


        Kmidz = 0.5 * (Kvec[:-1, 1:-1] + Kvec[1:, 1:-1])  # Shape: [Nz-1, Nx-2]
        h_z_forward  = (h_pred[1:-1, 1:-1] - h_pred[:-2, 1:-1]) / dz  # Shape: [Nz-2, Nx-2]
        h_z_backward = (h_pred[2:, 1:-1] - h_pred[1:-1, 1:-1]) / dz   # Shape: [Nz-2, Nx-2]


        Kmidx= 0.5 * (Kvec[1:-1, :-1] + Kvec[1:-1, 1:])
        h_x_forward = (h_pred[1:-1, 1:-1] - h_pred[1:-1, :-2]) / (dx )
        h_x_backward = (h_pred[1:-1, 2:] - h_pred[1:-1, 1:-1]) / (dx )



        dflux = (1/dz) * (Kmidz[1:, :] * (h_z_backward - 1) - Kmidz[:-1, :] * (h_z_forward - 1)) + (1/dx) * (Kmidx[:,1:] * (h_x_backward) - Kmidx[:,:-1] * (h_x_forward))

        # residual
        R = theta_t[1:-1 , 1:-1]-dflux[:, :]
        return self.Training_loss(R.squeeze(), torch.zeros_like(R).squeeze(), type='mae')
        (R1, R2, R3), (R4, R5, R6) = [                                                                                                                                                                                                                                              
            torch.chunk(row, 3, dim=1)
            for row in torch.chunk(R, 2, dim=0)                                                                                                                                                                                                                                     
        ]  
        L_phys = sum(torch.mean(torch.abs(r - torch.zeros_like(r))) for r in [R1,R2,R3,R4,R5,R6])
        #loss1 = F.mse_loss(R1.squeeze(), torch.zeros_like(R1).squeeze())
        #loss2 = torch.mean(torch.abs(R2.squeeze() - torch.zeros_like(R2).squeeze()))
        return L_phys #loss1 + loss2
        

        """ABS_R = torch.abs(R)
        p = ABS_R**k / torch.sum(ABS_R**k)

        weighted_R = p * R"""

        return F.mse_loss(R.squeeze(), torch.zeros_like(R).squeeze()) #+ F.mse_loss(R_top.squeeze(), torch.zeros_like(R_top).squeeze())     # physics loss
    
    def compute_weights(self, h_pred , theta_t , z , x):
        dz = (z[0, 1 ,0] - z[0 ,0, 0])
        dx = (x[0,0,1] - x[0,0,0])
        Kvec = self.K(h_pred , z).squeeze()


        Kmidz = 0.5 * (Kvec[:-1, 1:-1] + Kvec[1:, 1:-1])
        h_z_forward = (h_pred[1:-1, 1:-1] - h_pred[:-2, 1:-1]) / (dz )
        h_z_backward = (h_pred[2:, 1:-1] - h_pred[1:-1, 1:-1]) / (dz )


        Kmidx= 0.5 * (Kvec[1:-1, :-1] + Kvec[1:-1, 1:])
        h_x_forward = (h_pred[1:-1, 1:-1] - h_pred[1:-1, :-2]) / (dx )
        h_x_backward = (h_pred[1:-1, 2:] - h_pred[1:-1, 1:-1]) / (dx )



        dflux = (1/dz) * (Kmidz[1:, :] * (h_z_backward - 1) - Kmidz[:-1, :] * (h_z_forward - 1)) + (1/dx) * (Kmidx[:,1:] * (h_x_backward) - Kmidx[:,:-1] * (h_x_forward))

        # residual
        R = theta_t[1:-1 , 1:-1]-dflux[:, :]
        p = R**k / torch.sum(R**k)
        return p


    def adaptive_weights_init(self):
        """
        adaptive weight initialization
        """
        w1 = torch.tensor(0.0, requires_grad =True, device = self.device)
        w2 = torch.tensor(0.0, requires_grad =True, device = self.device)
        w3 = torch.tensor(0.0, requires_grad =True, device = self.device)
        w4 = torch.tensor(0.0, requires_grad =True, device = self.device)
        w5 = torch.tensor(0.0, requires_grad =True, device = self.device)
        return w1, w2, w3, w4, w5
    def boundary_losses(self, h_pred , h_top , h_init):
        """
        boundary losses calculations
        """
        L_bnd1 = F.mse_loss(h_pred[0].squeeze(), h_top.squeeze() ) # first boundary condition
        L_bnd2 = F.mse_loss(h_pred[-1].squeeze(), h_init.squeeze() ) # second boundary condition

        return L_bnd1 , L_bnd2

    def boundary_loss(self, h_pred , value):
        """
        boundary losse calculations
        """
        Loss = F.mse_loss(h_pred.squeeze(), value.squeeze() ) # second boundary condition

        return Loss

    def boundarytop_loss(self, h_pred , value, z , x, ep ):
        """
        boundary losse calculations
        """

        dz = (z[0, 1 ,0] - z[0 ,0, 0])
        dx = (x[0,0,1] - x[0,0,0])
        Kvec = self.K(self.solution[-1] , z).squeeze()

        dirichlet_loss = F.mse_loss(h_pred[0,:].squeeze(), value[:].squeeze() ) # second boundary condition
        q_pred1 = -Kvec[0,:22] * (((h_pred[1, :22] - h_pred[0, :22])/dz )  -   1)
        q_pred2 = -Kvec[0,25:] * (((h_pred[1, 25:] - h_pred[0, 25:])/dz )  -   1)

        loss_neumann = F.mse_loss(q_pred1[1:] , torch.zeros_like(q_pred1[1:]) ) + F.mse_loss(q_pred2[:-1] , torch.zeros_like(q_pred2[:-1]) )

        Loss = dirichlet_loss #+ loss_neumann


        q_pred = -Kvec[0, :] * (((h_pred[1, :] - h_pred[0, :])/dz )  -   1)
        if (ep+1) % 1000 == 0:
          plt.plot(q_pred.detach().cpu().numpy() , label = 'predicted flux')
          plt.legend()
          plt.show()

          plt.plot(h_pred[0, :].detach().cpu().numpy() , label = 'predicted h_top')
          plt.legend()
          plt.show()
        return Loss

    def Neuman_boundary_loss(self, h_pred , value , z , x, it, k = 1 , location = 'top'):
        """
        boundary bottom losses calculations
        """
        dz = (z[0, 1 ,0] - z[0 ,0, 0])
        dx = (x[0,0,1] - x[0,0,0])
        Kvec = self.K(h_pred, z).squeeze()


        if location == 'top' :
            q_pred = -Kvec[0, :] * (((h_pred[1, :] - h_pred[0, :])/(dz) )  -   1)  #-Kvec[0, :] * (((h_pred[1, :] - h_pred[0, :])/(1*dz) )  -   1) #-Kvec[0, :] *(-3.0*h_pred[0,:] + 4.0*h_pred[1,:] - h_pred[2,:]) / (2.0*dz) #
  
            K_mid_0 = (Kvec[0, :] + Kvec[1, :])/2

            #return F.mse_loss(q_pred[22:25] , value[22:25]  ) + 50 * F.mse_loss(q_pred[:22] , value[:22]) + 50 * F.mse_loss(q_pred[25:] , value[25:]  ) , q_pred
            #K_mid_0 = K_mid_0.clamp(min=1e-7)
            R_top =h_pred[1, :] - ((-value/K_mid_0 + 1) * (dz) + h_pred[0, :] )
            ABS_R_top = torch.abs(R_top)
            P_top = ABS_R_top**k / torch.sum(ABS_R_top**k)
            weighted_R_top = P_top * R_top


            return self.Training_loss(R_top.squeeze(), torch.zeros_like(R_top).squeeze(), type='mae'), q_pred
        elif location == 'bottom' :
            q_pred = -Kvec[-1, :] * (((h_pred[-1, :] - h_pred[-2, :])/(dz) )  -   1)
            return self.Training_loss(h_pred[-1, :].squeeze(), ((dz) + h_pred[-2, :]).squeeze(), type='mae'), q_pred
        elif location == 'left' :
            q_pred = -Kvec[:, 0] * (((h_pred[: , 1] - h_pred[: , 0])/(dx) ) )
            return self.Training_loss(h_pred[:-1 , 0] , h_pred[:-1 , 1] , type='mae'), q_pred
        elif location == 'right' :
            q_pred = -Kvec[:, -1] * (((h_pred[: , -1] - h_pred[: , -2])/(dx) ) )
            return self.Training_loss(h_pred[:-1 , -1] , h_pred[:-1 , -2] , type='mae'), q_pred
        #print(q_pred)

        """Kmid = (Kvec[:-1]+Kvec[1:])/2
        h_z = (h_pred[2:] - h_pred[:-2]) / (2*dz )

        #h_z1 = (h_pred[1:] - h_pred[:-1]) / (dz )
        #h_z2 = (h_pred[1:-1] - h_pred[:-2]) / (dz )
        K_z = (Kvec[2:] - Kvec[:-2]) / (2*dz )
        # second order central difference on z
        h_zz = (h_pred[2:] - 2*h_pred[1:-1] + h_pred[:-2]) / (dz**2)

        # flux
        flux = K_z*(h_z -1) + Kvec[1:-1]*h_zz
        #flux = (1/dz) * (Kmid[1:]*(h_z - 1) - Kmid[:-1]*(h_z - 1))
        flux_surface = 2 * flux[0] - flux[1]"""



        """h_z = self.grad_auto(h_pred , z = z)
        Iflux = Kvec*(h_z - 1 )

        flux = self.grad_auto(Iflux , z)"""

        #print(Kvec[0, :])

        #print(q_pred, value)
        #Loss = F.mse_loss(q_pred , value )

        return F.mse_loss(q_pred , value ), q_pred
    

    def Training_loss(self, h_pred , h_true, type = "mse"):
        """
        training loss calculation
        """
        if type.lower() == "mse":
            return F.mse_loss(h_pred.squeeze(), h_true.squeeze())
        elif type.lower() == "mae":
            return F.l1_loss(h_pred.squeeze(), h_true.squeeze())
        else :
            raise TypeError("Type do not exist or not implemented")

    def Loss_tracker(self , itteration , loss):
        """
        Tracker of loss evolution in each time step
        """
        self.loss_tracker[itteration].append(loss.item())

    def epochs_counter(self, itteration , ep_final):
        """
        Save the number of epochs needed at each time step
        """
        self.epochs_count.append((itteration , ep_final))
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



# ──────────────────────────────────────────────────────────────────────────────
# Profiled version of SPINNTrainer_not
# Each section of the training loop is timed and reported after every time step.
# GPU sections use torch.cuda.synchronize() before/after for accurate measurement.
# Sections labelled [SYNC] force a CPU←GPU data transfer (pipeline bubble).
# ──────────────────────────────────────────────────────────────────────────────
class SPINNTrainer_profiled(SPINNTrainer_2D):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_cuda = (self.device != 'cpu' and torch.cuda.is_available())

    # ── timing helpers ────────────────────────────────────────────────────────

    def _sync(self):
        if self._use_cuda:
            torch.cuda.synchronize()

    def _gpu_timed(self, fn):
        """Run fn() with GPU-accurate wall-clock timing via explicit synchronize."""
        self._sync()
        t0 = time.perf_counter()
        result = fn()
        self._sync()
        return result, time.perf_counter() - t0

    def _cpu_timed(self, fn):
        """Run fn() and time it on CPU (no sync — use only for pure Python/CPU work)."""
        t0 = time.perf_counter()
        result = fn()
        return result, time.perf_counter() - t0

    @staticmethod
    def _print_timing_table(timings, n_epochs):
        GPU_KEYS  = {'forward_pass', 'transition_loss', 'theta_compute',
                     'physics_loss', 'boundary_top', 'boundary_bottom',
                     'boundary_left', 'boundary_right', 'loss_combine',
                     'backward', 'optimizer_step', 'clone_op'}
        SYNC_KEYS = {'convergence_check'}   # forces CPU←GPU transfer every epoch
        # everything else is CPU

        total = sum(timings.values()) or 1e-12
        W = 24
        print(f"\n{'─'*60}")
        print(f"  Timing breakdown  (epochs={n_epochs})")
        print(f"{'─'*60}")
        print(f"  {'Tag':<7} {'Section':<{W}}  {'Total(s)':>9}  {'ms/epoch':>9}  {'%':>6}")
        print(f"{'─'*60}")
        for key, val in sorted(timings.items(), key=lambda kv: -kv[1]):
            if key in GPU_KEYS:
                tag = '[GPU] '
            elif key in SYNC_KEYS:
                tag = '[SYNC]'
            else:
                tag = '[CPU] '
            pct    = 100.0 * val / total
            per_ep = 1000.0 * val / max(n_epochs, 1)
            print(f"  {tag} {key:<{W}}  {val:>9.4f}  {per_ep:>9.3f}  {pct:>5.1f}%")
        print(f"{'─'*60}")
        print(f"  {'':7} {'TOTAL':<{W}}  {total:>9.4f}")
        print(f"{'─'*60}\n")

    # ── instrumented train ────────────────────────────────────────────────────

    def train(self, T, z, x,
              epochs=1000, epochs_max=1e4, E_max=2e4,
              tol=1e-15, loss_max=1e-6, loss_min=1e-4,
              subdomains=[], h_init=None,
              h_top=("Dirichlet", None), h_bottom=("Dirichlet", None),
              h_left=("Dirichlet", None), h_right=("Dirichlet", None),
              adaptive_weights=False, regularization=0.05, form="mixed"):

        Nt = T.shape[0]
        h_old = h_init.unsqueeze(-1)
        self.solution = []
        self.solution.append(h_old)

        mse_fn = nn.MSELoss()
        j = 0
        nt_sub, dt_sub = subdomains[j]

        for n in range(0, Nt - 1):
            iter_t0 = time.perf_counter()

            timings = {
                'forward_pass':      0.0,   # [GPU]
                'transition_loss':   0.0,   # [GPU]  ← computed but not used in loss!
                'theta_compute':     0.0,   # [GPU]
                'physics_loss':      0.0,   # [GPU]  includes K_2D call
                'boundary_top':      0.0,   # [GPU]  includes K_2D call
                'boundary_bottom':   0.0,   # [GPU]  includes K_2D call
                'boundary_left':     0.0,   # [GPU]  includes K_2D call
                'boundary_right':    0.0,   # [GPU]  includes K_2D call
                'loss_combine':      0.0,   # [GPU]
                'backward':          0.0,   # [GPU]
                'optimizer_step':    0.0,   # [GPU]
                'convergence_check': 0.0,   # [SYNC] forces CPU←GPU sync every epoch
                'clone_op':          0.0,   # [GPU]
            }

            previous_h = h_old.clone()
            theta_old  = self.theta(h_old.squeeze(), z[n])

            if n == 0:
                w1, w2, w3, w4, w5 = self.adaptive_weights_init()
                E = E_max
                if adaptive_weights:
                    self.optimizers[n].add_param_group({'params': [w1,w2,w3,w4,w5], 'lr': 0.001})
            else:
                E = epochs

            print(f"iteration {n}")
            epoch = 0

            while epoch < E:
                self.models[n].train()
                self.optimizers[n].zero_grad()

                # ── [GPU] Forward pass ────────────────────────────────────────
                h_pred, dt = self._gpu_timed(
                    lambda: self.models[n](z[n], x[n], h_old.squeeze()).squeeze()
                )
                timings['forward_pass'] += dt

                # ── [GPU] Transition loss (NOTE: not added to final loss) ─────
                def _transition_fn():
                    dh = h_pred - h_old.squeeze()
                    tx = dh[:, 1:] - dh[:, :-1]
                    return self.Training_loss(tx, torch.zeros_like(tx), type='mae')
                _, dt = self._gpu_timed(_transition_fn)
                timings['transition_loss'] += dt

                # ── [GPU] Theta + time derivative ─────────────────────────────
                def _theta_fn():
                    nonlocal nt_sub, dt_sub, j
                    tv = self.theta(h_pred, z)
                    if n < nt_sub + 1:
                        if form.lower() == "mixed":
                            return (tv - theta_old) / dt_sub
                        return self.C(h_pred, z) * ((h_pred - h_old.squeeze()) / dt_sub)
                    else:
                        nt_sub_c, dt_sub = subdomains[j + 1]
                        nt_sub = nt_sub_c + nt_sub
                        j += 1
                        if form.lower() == "mixed":
                            return (self.theta(h_pred, z) - theta_old) / dt_sub
                        return self.C(h_pred, z) * ((h_pred - h_old.squeeze()) / dt_sub)
                theta_t, dt = self._gpu_timed(_theta_fn)
                timings['theta_compute'] += dt

                # ── [GPU] Physics loss (calls K_2D once internally) ───────────
                L_phys, dt = self._gpu_timed(lambda: self.physics_loss(h_pred, theta_t, z, x))
                timings['physics_loss'] += dt

                # ── [GPU] Boundary losses (each calls K_2D independently) ─────
                if h_top[0] == "Dirichlet":
                    L_bnd1, dt = self._gpu_timed(lambda: self.boundary_loss(h_pred[0, :], h_top[1][n, :]))
                else:
                    (L_bnd1, _), dt = self._gpu_timed(lambda: self.Neuman_boundary_loss(h_pred, h_top[1][n, :], z, x, epoch, location='top'))
                timings['boundary_top'] += dt

                if h_bottom[0] == "Dirichlet":
                    L_bnd2, dt = self._gpu_timed(lambda: self.boundary_loss(h_pred[-1, :], h_bottom[1][n, :]))
                else:
                    (L_bnd2, _), dt = self._gpu_timed(lambda: self.Neuman_boundary_loss(h_pred, h_bottom[1][n, :], z, x, epoch, location='bottom'))
                timings['boundary_bottom'] += dt

                if h_left[0] == "Dirichlet":
                    L_bnd3, dt = self._gpu_timed(lambda: self.boundary_loss(h_pred[:, 0], h_left[1][n, :]))
                else:
                    (L_bnd3, _), dt = self._gpu_timed(lambda: self.Neuman_boundary_loss(h_pred, h_left[1][n, :], z, x, epoch, location='left'))
                timings['boundary_left'] += dt

                if h_right[0] == "Dirichlet":
                    L_bnd4, dt = self._gpu_timed(lambda: self.boundary_loss(h_pred[:, -1], h_right[1][n, :]))
                else:
                    (L_bnd4, _), dt = self._gpu_timed(lambda: self.Neuman_boundary_loss(h_pred, h_right[1][n, :], z, x, epoch, location='right'))
                timings['boundary_right'] += dt

                # ── [GPU] Combine losses ──────────────────────────────────────
                def _combine_fn():
                    if adaptive_weights:
                        return (torch.exp(-w1)*L_phys + torch.exp(-w2)*L_bnd1 +
                                torch.exp(-w3)*L_bnd2*10 + torch.exp(-w4)*L_bnd3 +
                                torch.exp(-w5)*L_bnd4 +
                                regularization*(torch.abs(w1)+torch.abs(w2)+torch.abs(w3)+torch.abs(w4)+torch.abs(w5)))
                    return L_phys + L_bnd1 + L_bnd2 + L_bnd3 + L_bnd4
                loss, dt = self._gpu_timed(_combine_fn)
                timings['loss_combine'] += dt

                # ── [GPU] Backward ────────────────────────────────────────────
                _, dt = self._gpu_timed(lambda: loss.backward(retain_graph=False))
                timings['backward'] += dt

                # ── [GPU] Optimizer + scheduler ───────────────────────────────
                def _opt_fn():
                    self.optimizers[n].step()
                    self.schedulers[n].step()
                _, dt = self._gpu_timed(_opt_fn)
                timings['optimizer_step'] += dt

                # ── [SYNC] Convergence check — triggers CPU←GPU transfer ──────
                # mse(...) < tol  and  loss.item()  both call .item() which flushes
                # the entire GPU pipeline and blocks Python until the value arrives.
                def _conv_fn():
                    return (mse_fn(h_pred, previous_h.squeeze()) < tol
                            or loss.item() < loss_max)
                converged, dt = self._cpu_timed(_conv_fn)
                timings['convergence_check'] += dt

                if converged:
                    print(f"epoch {epoch:4d}, iter {n+1}  loss={loss.item():.2e}")
                    break

                if epoch == E - 1 and loss.item() > loss_min and E < epochs_max:
                    E += 1000

                # ── [GPU] Clone previous_h ────────────────────────────────────
                previous_h, dt = self._gpu_timed(lambda: h_pred.clone())
                timings['clone_op'] += dt

                epoch += 1

                if epoch % 500 == 0:
                    print(f"epoch {epoch:4d}  loss={loss.item():.2e} "
                          f"L_phys={L_phys.item():.2e} "
                          f"L_bnd1={L_bnd1.item():.2e} L_bnd2={L_bnd2.item():.2e} "
                          f"L_bnd3={L_bnd3.item():.2e} L_bnd4={L_bnd4.item():.2e}")

            # ── Summary ───────────────────────────────────────────────────────
            self._print_timing_table(timings, epoch + 1)
            print(f"Total iteration {n+1} wall time: {time.perf_counter() - iter_t0:.2f}s")

            with torch.no_grad():
                self.models[n].eval()
                h_old = h_pred.detach().clone().unsqueeze(-1)
                self.solution.append(h_pred.unsqueeze(-1))
                self.models[n+1].load_state_dict(self.models[n].state_dict())

        self.solution = torch.stack(self.solution, dim=1)
        return self.solution