import torch



def load_data(args):
    
    if args.use_gpu == 1 :
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else :
        device = torch.device('cpu')
    
    z = torch.linspace(0 , args.z_final , args.Nz , device =device )
    t = torch.linspace(0 , args.t_final , args.Nt , device =device )
    
    Z,T = torch.meshgrid(t, z, indexing='ij')
    T = T + (args.t_final /(args.t_final +1))
    
    if args.subdomains == 'classic':
        subdomains = [(args.Nt , args.t_final/args.Nt)]
    
    
    return z, t, Z, T, subdomains
    