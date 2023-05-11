import torch
import os
from .network_parser import parse


# change checkpoint file if needed
checkpoint_file = "best_hole_run_0_dilated_1e-4_4_128.pth"
config_file_name = "hole0.yaml"



dtype = None
n_steps = None
network_config = None
layer_config = None
devices = None
params = {}

def init(devs):
    #load best trained fsvae model 
    cwd = os.getcwd()
    fsvae_path = os.path.join(cwd, "gym-env", "gym_env", "envs", "fsvae")

    checkpoint_path = os.path.join(fsvae_path, checkpoint_file)
    config_path = os.path.join(fsvae_path, "NetworkConfigs", config_file_name)

    p = parse(config_path)
    n_config = p['Network']


    global dtype, devices, n_steps, tau_s, network_config, layer_config, params
    dtype = torch.float32
    devices = devs
    network_config = n_config
    network_config['batch_size'] = network_config['batch_size'] * len(devices)
    network_config['lr'] = network_config['lr'] * len(devices) * network_config['batch_size'] / 250
    layer_config = {'threshold': 0.2}
    n_steps = network_config['n_steps']
    
    network_config['checkpoint_path'] = checkpoint_path
