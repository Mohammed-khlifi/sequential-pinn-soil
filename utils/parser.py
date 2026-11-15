import yaml
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='Train SPINNs')
    parser.add_argument('--env', type=str, choices=['SG', 'YLC'],
                      default='SG', help='Sable de Grenoble / Argile YOLO ..')
    parser.add_argument('--config', type=str, default='Config/params.yml',
                      help='Path to config file')
    parser.add_argument('--use_gpu', type=int ,
                      default=1 , help='usage of gpu')
    parser.add_argument('--Nz', type=int ,
                      default=1 , help='Number of spatial points')
    parser.add_argument('--Nt', type=int ,
                      default=1 , help='Number of time steps')
    parser.add_argument('--t_final', type = float , default = 1 , help ="final T")
    parser.add_argument('--z_final', type = int , default = 1 , help ="final Z")
    parser.add_argument('--subdomains', type = str , default = 1 , help ="subdomains")
    parser.add_argument('--h_top', type = float , default = 1 , help ="h_top")
    parser.add_argument('--h_bot', type = float , default = 1 , help ="h_bot")
    
    parser.add_argument('--epochs', type=int ,
                      default=1 , help='number of epochs')
     
    parser.add_argument('--save_dir' , type = str , default = None , help = 'saving path')
    
    
    return parser