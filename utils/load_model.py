import torch ; import torch.nn as nn
from utils.utilities import init_weights
from Models import PINNsformer,PINN, Attention, RC_PINN



def load_model(param, args):
    if args.use_gpu == 1 :
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else :
        device = torch.device('cpu')
    if param['model'] == "RC_PINN":
        Models = [RC_PINN(
        conv_layers=[(3, 16, 5), (16, 32, 5), (32, 64, 5)],
        linear_layers=[(64, 128),(128,512) ,(512 ,512) ,(512,128) ,(128, 1)],
        activation=nn.ReLU()).to(device) for _ in range(args.Nt)]
    else :
        raise Exception("Sorry, RC_PINN is the only available model currently") 
    Models[0].apply(init_weights)
    
    
    
    optimizers = [
    torch.optim.Adam(
        Models[i].parameters(),
        lr= float(param['lr']) if i < 5 else float(param['lr'])
    )
    for i in range(args.Nt)
    ]

    schedulers = [
        torch.optim.lr_scheduler.StepLR(opt, step_size=10000, gamma=0.25)
        for opt in optimizers
    ]
    
    
    return Models , optimizers , schedulers