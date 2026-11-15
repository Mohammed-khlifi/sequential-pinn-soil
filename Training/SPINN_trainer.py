from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
from typing import Callable, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass




class SPINNTrainer:
   
    def __init__(self, K : callable, C : callable, theta : callable, models : list, optimizers : list, schedulers : list,discritization = 'normal',form = 'theta' , device='cpu'):
        
        ###### Hydrodynamic features ######
        self.K = K
        self.C = C
        self.theta = theta
        ##### Form and discritization #####
        self.discritization = discritization 
        self.form = form
        ###### neural networks models #####
        self.models = models
        self.device = device
        self.optimizers = optimizers
        self.schedulers = schedulers
        ##### Trackers #########
        self.loss_tracker = {i: [] for i in range(len(models))}  # to track losses for each model
        self.epochs_count = []
        self.solution = []
       
       
    def train(self,
              T,
              z , 
              epochs : int = 1000,
              epochs_max : int= 1e4,
              E_max : int = 2e4,
              tol : float =1e-15,
              loss_max : float = 1e-6,
              loss_min : float = 1e-4,
              max_patience = 10,
              subdomains : list= [] ,
              h_init = None ,
              h_top = ("Dirichlet", None) ,
              h_bottom =  ("Dirichlet", None) ,
              adaptive_weights = False
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
    
        Nt, Nz = T.shape 
        self.h_old = h_init.unsqueeze(-1) 
        self.solution.append(self.h_old)
        self.subdomains = subdomains
        
        Losses = []
        mse = nn.MSELoss()
        self.j=0
        self.nt_sub , self.dt_sub  = subdomains[self.j]
        #nt_sub , dt_sub  = subdomains[j]
        
        #time itteration lopp
        for n in range(0,Nt -1):

            try :            
                previous_h = self.h_old.clone()
                theta_old = self.theta(self.h_old.squeeze() , z)
                if n == 0 :
                    E = E_max
                    self.w1,self.w2,self.w3 = self.adaptive_weights_init()      
                else : 
                    E = epochs
                if adaptive_weights :
                    self.optimizers[n].add_param_group({
                    'params':       [self.w1,self.w2,self.w3],
                    'lr':           0.001     # can be different
                    })
                print(f"itteration {n}")
                epoch = 0
                # Training loop
                self.args = theta_old, h_top, h_bottom, T , z , n
    
                old_loss = 1e+20
                patience = 0
                while epoch < E:
                    self.optimizers[n].step(self.train_one_epoch)
                
                    stop = self.check_stop_condition(n , epoch, previous_h, E ,epochs_max ,loss_min ,loss_max, tol)
                    if stop == True :
                        break
                    elif stop == False :
                        E += epochs
                    else :
                        pass
    
                    if self.check_divergence == True :
                        break
                    else:
                        old_loss = self.check_divergence
                   
                    previous_h = self.h_pred.clone()
                    epoch += 1
                    
                    self.print_evolution(epoch)
                       
                self.epochs_counter(n , epoch)
                # update the head
                self.update_head(n)
            except:
                print(f"an Error has occured , the evolution level is {n+1}")
                break
                
        self.solution =  torch.stack(self.solution , dim = 1)
        return self.solution
       
    def train_one_epoch(self):
        theta_old, h_top, h_bottom, T , z , n = self.args 
        self.models[n].train()
        self.optimizers[n].zero_grad()
       
        # physics residual
        h_pred=  self.models[n](T[n], z , self.h_old).squeeze() 
        theta_vec = self.theta(h_pred, z)
        
        # backward difference on t      
        theta_t = self.calculate_theta_t(n ,h_pred , theta_old)

        # the physics loss
        L_phys = self.physics_loss(h_pred, theta_t, z , discritization = self.discritization)

        # the upper boundary condition 
        if h_top[0].lower() == "dirichlet":
            L_bnd1 = self.boundary_loss(h_pred[0] ,h_top[1]) # top Dirichlet boundary conditions
        elif h_top[0].lower() == "neumann":
            L_bnd1 = self.Neuman_boundary_loss(h_pred , h_top[1], z) # top Neumann boundary conditions
        else :
            raise TypeError("Boundary type is not valid")
            
        # the loer boundary condition    
        if h_bottom[0].lower() == "dirichlet":
            L_bnd2 = self.boundary_loss(h_pred[-1] ,h_bottom[1]) # Bottom Dirichlet boundary conditions
        elif h_bottom[0].lower() == "neumann":
            L_bnd2 = self.Neuman_boundary_loss(h_pred , h_bottom[1], z) # Bottom Neumann boundary conditions
        else :
            raise TypeError("Boundary type is not valid")

        # the total loss function 
        loss = self.adaptive_loss(L_phys, L_bnd1, L_bnd2)

        self.Loss_tracker(itteration= n, loss=loss)  # track the loss
        loss.backward(retain_graph = True)
        #self.optimizers[n].step() ; self.schedulers[n].step() # update the model
        
       

        self.loss, self.L_phys , self.L_bnd1 , self.L_bnd2 , self.h_pred = loss, L_phys.item() , L_bnd1.item() , L_bnd2.item() , h_pred.detach()
        
        return loss
    def adaptive_loss(self ,L_phys, L_bnd1, L_bnd2 ):
        loss = torch.exp(-self.w1) * L_phys + torch.exp(-self.w2) * L_bnd1 + torch.exp(-self.w3) * L_bnd2 
        +  0.5  * (torch.abs(self.w1) +torch.abs(self.w2) +torch.abs(self.w3))
        return loss
    
    def physics_loss(self, h_pred , theta_t , z , discritization = 'normal'):
        """
        physical loss calculation
        h_pred : the prediction given by the model
        theta_t : derivative of theta respect to time at h_pred
        """
        dz = (z[1] - z[0])
        Kvec = self.K(h_pred , z).squeeze()    
        
        if discritization.lower() == 'normal':
            Kmid = 0.5 * (Kvec[:-1] + Kvec[1:]) 
            h_z_forward = (h_pred[1:-1] - h_pred[:-2]) / (dz )
            h_z_backward = (h_pred[2:] - h_pred[1:-1]) / (dz )
            
            dflux = (1/dz) * (Kmid[1:] * (h_z_backward - 1) - Kmid[:-1] * (h_z_forward - 1))
            R = theta_t[1:-1] - dflux
        elif discritization.lower() == 'hybrid':
            h_z = (h_pred[2:] - h_pred[:-2]) / (2*dz )
            K_z = (Kvec[2:] - Kvec[:-2]) / (2*dz )
            
            # second order central difference on z
            h_zz = (h_pred[2:] - 2*h_pred[1:-1] + h_pred[:-2]) / (dz**2)
            
            # flux
            dflux = K_z*(h_z -1) + Kvec[1:-1]*h_zz
            R = theta_t[1:-1] - dflux
        elif discritization.lower() == 'autograd':
            h_z = self.grad_auto(h_pred , z = z)
            Iflux = Kvec*(h_z - 1 )
            
            dflux = self.grad_auto(Iflux , z)
            R= theta_t[1:-1] -dflux[1:-1]
        
        # residual
        #R = theta_t[1:-1] - dflux
        # = theta_t[1:-1] -flux[1:-1]
        return F.mse_loss(R.squeeze(), torch.zeros_like(R).squeeze())    # physics loss
    def adaptive_weights_init(self):
        """
        adaptive weight initialization
        """
        w1 = torch.tensor(0.0, requires_grad =True, device = self.device)
        w2 = torch.tensor(0.0, requires_grad =True, device = self.device)
        w3 = torch.tensor(0.0, requires_grad =True, device = self.device)
        
        return w1, w2, w3
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
    
    def Neuman_boundary_loss(self, h_pred , value , z):
        """
        boundary bottom losses calculations
        """
        dz = (z[1] - z[0])
        Kvec = self.K(h_pred , z).squeeze()
        Kmid0 = 0.5 * (Kvec[0] + Kvec[1])
        q_top = value
        condition = h_pred[1] - dz * (1.0 - q_top / Kmid0)
        
        q_pred = - Kvec[0]*(((h_pred[1] - h_pred[0])/dz )  -   1)
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
        return F.mse_loss(q_pred , value)

    def calculate_theta_t(self,n ,h_pred , theta_old):
        theta_vec = self.theta(h_pred, z)
        if n < self.nt_sub + 1 :
            if self.form.lower() == 'theta':
                theta_t = (theta_vec - theta_old) / (self.dt_sub)
            elif self.form.lower()  == 'mixte':
                theta_t = self.C(h_pred, z) * ((h_pred - self.h_old.squeeze())/(self.dt_sub))
            else :
                Exception(f"{self.form.lower()} is not valid") 
        else :       
            nt_sub_c , self.dt_sub = self.subdomains[self.j+1] 
            self.nt_sub += nt_sub_c 
            if self.form.lower() == 'theta':
                theta_t = (theta_vec - theta_old) / (self.dt_sub)
            elif self.form.lower()  == 'mixte':
                theta_t = self.C(h_pred, z) * ((h_pred - self.h_old.squeeze())/(self.dt_sub))
            else :
                Exception(f"{self.form.lower()} is not valid")
            self.j += 1
        return theta_t

    def check_divergence(self, old_loss):
        if self.loss > old_loss :
            if patience < max_patience :
                patience += 1
            else : 
                print('the loss is diverging')
                return True #break
        else :
            patience = 0
        return self.loss

    def print_evolution(self, epoch , rate = 100):
        if (epoch) %rate ==0:
            print(f"epoch {(epoch):4d}  loss={self.loss:.2e} , L_phys={self.L_phys:.2e} , L_bnd1={self.L_bnd1:.2e}, L_bnd2={self.L_bnd2:.2e} ")
            pass
    def check_stop_condition(self ,n ,epoch ,previous_h ,E ,epochs_max ,loss_min ,loss_max, tol):
        if F.mse_loss(self.h_pred , previous_h.squeeze()) < tol or self.loss < loss_max :
            print(f"epoch {epoch:4d}, itteration {n+1}  loss={self.loss.item():.2e} , variance={F.mse_loss(self.h_pred.squeeze() , previous_h.squeeze()):.2e}")
            return True
        elif epoch == E - 1  and self.loss > loss_min and E < epochs_max:
            return False #E += 200
            
    def update_head(self , n):
        with torch.no_grad():
            self.models[n].eval()
            self.h_old = self.h_pred.detach().clone().unsqueeze(-1) # update the head
            self.solution.append(self.h_pred.unsqueeze(-1)) # store the head
            self.models[n+1].load_state_dict(self.models[n].state_dict())
        
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

        



