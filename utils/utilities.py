import numpy as np
import torch
import torch.nn as nn

# ---------- Defining the parameters ----------
# SG parameters
params_SG = {
    'h0': -61.5,          # initial head (cm)
    'h1': -20.73,         # Top head (cm)
    'Ks': 34.0,        # saturated hydraulic conductivity (cm/s)
    'tr': 0.075,          # residual water content
    'ts': 0.287,           # saturated water content
    'a1': 1.175e6,       # van Genuchten parameter
    'a2': 4.74,          # van Genuchten parameter
    'a3': 1.611e6,       # van Genuchten parameter
    'a4': 3.96,          # van Genuchten parameter
}
# YLC parameters
params_YLC = {
    'h0': -599.974,          # initial head (cm)
    'h1': 0.0,         # Top head (cm)
    'Ks': 4.428e-2,        # saturated hydraulic conductivity (cm/s)
    'tr': 0.124,          # residual water content
    'ts': 0.495,           # saturated water content
    'a1': 124.6,       # van Genuchten parameter
    'a2': 1.77,          # van Genuchten parameter
    'a3': 739,       # van Genuchten parameter
    'a4': 4,          # van Genuchten parameter
}


param_multylayer = {
    'layer0-20' : {
        'tr' : 0.05,
        'ts' : 0.28,
        'n' : 2.4,
        'hg': -83.334,
        'Ks' : 0.52,
        'B' : 0.1,
        'N' : 2.6,
        'R2h':87.59
    },
    'layer20-60' : {
        'tr' : 0.05,
        'ts' : 0.32,
        'n' : 2.18,
        'hg': -62.5,
        'Ks' : 0.23,
        'B' : 0.1,
        'N' : 2.6,
        'R2h': 91.0
    },
    'layer60-80' : {
        'tr' : 0.05,
        'ts' : 0.36,
        'n' : 2.2,
        'hg': -17.857,
        'Ks' : 0.187,
        'B' : 0.1,
        'N' : 2.6,
        'R2h':97.84
    },
    'layer80-90' : {
        'tr' : 0.05,
        'ts' : 0.321,
        'n' : 2.26,
        'hg': -15.625,
        'Ks' : 0.187,
        'B' : 0.1,
        'N' : 2.6,
        'R2h':98.99
    },
    'layer90-150' : {
        'tr' : 0.12,
        'ts' : 0.263,
        'n' : 3.81,
        'hg': -3.226,
        'Ks' : 110.47,
        'B' : 0.244,
        'N' : 4.1,
        'R2h':97.19
    },
}


#--------- Hydrodynamics caracteristics functions ---------
# SG
def K_SG(h, z = None):
    return params_SG['Ks'] * (params_SG['a1'] / (params_SG['a1'] + torch.abs(h)**params_SG['a2']))

def theta_SG(h, z = None):
    return params_SG['tr'] + (params_SG['ts']-params_SG['tr'])*(params_SG['a3']/(params_SG['a3']+torch.abs(h)**params_SG['a4']))

def C_SG(h, z = None):
    p = params_SG
    return (p['ts']-p['tr'])*p['a3']*p['a4']* \
           torch.abs(h)**(p['a4']-1) / (p['a3']+torch.abs(h)**p['a4'])**2
# YLC
# YLC
def K_YLC(h, z = None):
    p = params_YLC
    # use dict shorthand
    return (p['Ks']*p['a1']) / (p['a1'] + torch.abs(h)**p['a2'])

def theta_YLC(h, z = None):
    p = params_YLC
    Theta = torch.ones_like(h)*p['ts'] 
    Theta[h<-1] = p['tr'] + (p['ts'] - p['tr']) * (p['a3'] / (p['a3'] + torch.log(torch.abs(h[h < -1]))**p['a4']))
    return Theta
def C_YLC(h, z = None):
    p = params_YLC
    C = torch.zeros_like(h)  
    C[h >= -1] = 0.0
    C[h < -1] = (p['ts'] - p['tr'])* (1/torch.abs(h[h < -1])) * p['a3'] * p['a4'] * \
        torch.log(torch.abs(h[h < -1]))**(p['a4'] - 1) / \
        (p['a3'] + torch.log(torch.abs(h[h < -1]))**p['a4'])**2

    return C


# numpy version of the functions
# SG
def K_np(h , soil='SG',z = None):
    if soil == 'YLC':
        p = params_YLC
    else:
        p = params_SG           # use dict shorthand
    return (p['Ks']*p['a1']) / (p['a1'] + np.abs(h)**p['a2'])

def theta_np(h,z = None):
    p = params_SG 
    return p['tr'] + ((p['ts']-p['tr'])*p['a3']/(p['a3']+np.abs(h)**p['a4']))



def C_np(h ,z = None):
    p = params_SG
    return (p['ts']-p['tr'])*p['a3']*p['a4']* \
           np.abs(h)**(p['a4']-1) / (p['a3']+np.abs(h)**p['a4'])**2
        
# YLC
def K_YLC_np(h,z = None ):
    p = params_YLC
    # use dict shorthand
    return (p['Ks']*p['a1']) / (p['a1'] + np.abs(h)**p['a2'])

def theta_YLC_np(h,z = None):
    p = params_YLC
    Theta = np.ones_like(h)*p['ts'] 
    Theta[h<-1] = p['tr'] + (p['ts'] - p['tr']) * (p['a3'] / (p['a3'] + np.log(np.abs(h[h < -1]))**p['a4']))
    return Theta
def C_YLC_np(h,z = None):
    p = params_YLC
    C = np.zeros_like(h)  
    C[h >= -1] = 0.0
    C[h < -1] = (p['ts'] - p['tr'])* (1/np.abs(h[h < -1])) * p['a3'] * p['a4'] * \
        np.log(np.abs(h[h < -1]))**(p['a4'] - 1) / \
        (p['a3'] + np.log(np.abs(h[h < -1]))**p['a4'])**2

    return C

#multylayer

import torch

def theta_multy(h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Vectorized evaluation of theta(h, z) across depth vector z.
    
    h: Tensor of shape [..., N] or [N], same shape as z
    z: Tensor of shape [..., N] or [N], depth values
    Returns: Tensor of same shape, theta at each depth
    """
    # Ensure h and z have the same shape
    if h.shape != z.shape:
        raise ValueError(f"h and z must have same shape, got {h.shape} vs {z.shape}")
    
    # Pre‐allocate output
    Theta = torch.empty_like(h, device=h.device, dtype=h.dtype)
    
    # Define your layer ranges and corresponding param keys
    layer_specs = [
        (0,   20, 'layer0-20'),
        (20,  60, 'layer20-60'),
        (60,  80, 'layer60-80'),
        (80,  90, 'layer80-90'),
        (90, 151, 'layer90-150'),
    ]
    
    # Fill Theta by mask
    covered = torch.zeros_like(z, dtype=torch.bool)
    for z_min, z_max, key in layer_specs:
        mask_z = (z >= z_min) & (z < z_max +1)
        p = param_multylayer[key]
        Theta[mask_z] = p['ts']
        mask = mask_z & (h<0)
        if mask.any():
            p = param_multylayer[key]
            n = p['n']
            m = 1 - 2/n
            # compute theta for this layer
            # p['tr'], p['ts'], p['hg'] are scalars
            term = (1 + (h[mask] / p['hg']) ** n) ** (-m)
            Theta[mask] = p['tr'] + (p['ts'] - p['tr']) * term
            covered |= mask
    
    """# Check for any depths outside defined ranges
    if not covered.all():
        bad = z[~covered]
        raise ValueError(f"theta is undefined for depths: {bad}")"""
    
    return Theta


def K_multy(h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Vectorized evaluation of hydraulic conductivity K(h, z) across depth vector z.
    
    h: Tensor of shape [..., N] or [N], matric potential values
    z: Tensor of same shape, depth values
    Returns: Tensor of same shape, K at each depth
    """
    # Ensure h and z have the same shape
    if h.shape != z.shape:
        raise ValueError(f"h and z must have same shape, got {h.shape} vs {z.shape}")
    
    # Pre-allocate output
    Kout = torch.empty_like(h, device=h.device, dtype=h.dtype)
    
    # Layer definitions (min_depth, max_depth, dict_key)
    layer_specs = [
        (0,   20,  'layer0-20'),
        (20,  60,  'layer20-60'),
        (60,  80,  'layer60-80'),
        (80,  90,  'layer80-90'),
        (90, 151,  'layer90-150'),
    ]
    
    covered = torch.zeros_like(z, dtype=torch.bool)
    
    for z_min, z_max, key in layer_specs:
        mask = (z >= z_min) & (z < z_max+1)
        if mask.any():
            p = param_multylayer[key]
            # Ks, B, N are scalars in p
            numer = p['Ks']
            denom = 1 + (p['B'] * h[mask].abs())**p['N']
            Kout[mask] = numer / denom
            covered |= mask
    
    if not covered.all():
        bad_z = z[~covered]
        raise ValueError(f"K is undefined for depths: {bad_z}")
    
    return Kout

def C_multy(h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Derivative of theta_multy_np with respect to h, vectorized over depth z.

    h: ndarray [..., N] or [N]  (same shape as z)
    z: ndarray [..., N] or [N]
    returns: ndarray of same shape as h, dTheta/dh evaluated per layer.
    """
    if h.shape != z.shape:
        raise ValueError(f"h and z must have same shape, got {h.shape} vs {z.shape}")

    C = torch.zeros_like(h)

    # (min_depth, max_depth, parameter-key) — same partitions used in theta_multy_np
    layer_specs = [
        (0,   20, 'layer0-20'),
        (20,  60, 'layer20-60'),
        (60,  80, 'layer60-80'),
        (80,  90, 'layer80-90'),
        (90, 151, 'layer90-150'),
    ]

    #covered = torch.zeros_like(z, dtype=bool)

    for z_min, z_max, key in layer_specs:
        mask = (z >= z_min) & (z < z_max) & (h < 0)
        if torch.any(mask):
            p = param_multylayer[key]
            tr, ts, hg, n = p['tr'], p['ts'], p['hg'], p['n']
            m = 1.0 - 2.0 / n
            A = ts - tr

            C[mask] = A * (-m) * (n) * (1/hg) * ((h[mask]/hg)**(n-1)) * (1 +(h[mask]/hg)**(n) )**(-m-1)

            ##covered |= mask


    return C


####### numpy version ########


def theta_multy_np(h: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Vectorized evaluation of theta(h, z) across depth vector z (NumPy version).
    
    h: ndarray of shape [..., N] or [N], same shape as z
    z: ndarray of shape [..., N] or [N], depth values
    Returns: ndarray of same shape, theta at each depth
    """
    if h.shape != z.shape:
        raise ValueError(f"h and z must have same shape, got {h.shape} vs {z.shape}")
    
    Theta = np.empty_like(h, dtype=float)

    layer_specs = [
        (0,   20, 'layer0-20'),
        (20,  60, 'layer20-60'),
        (60,  80, 'layer60-80'),
        (80,  90, 'layer80-90'),
        (90, 151, 'layer90-150'),
    ]

    covered = np.zeros_like(z, dtype=bool)
    for z_min, z_max, key in layer_specs:
        mask_z = (z >= z_min) & (z < z_max +1)
        p = param_multylayer[key]
        Theta[mask_z] = p['ts']
        mask = mask_z & (h<0)
        if np.any(mask):
            #p = param_multylayer[key]
            n = p['n']
            m = 1 - 2 / n
            term = (1 + (h[mask] / p['hg']) ** n) ** (-m)
            Theta[mask] = p['tr'] + (p['ts'] - p['tr']) * term
            covered |= mask

    """if not np.all(covered):
        bad = z[~covered]
        raise ValueError(f"theta is undefined for depths: {bad}")"""

    return Theta


def K_multy_np(h: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Vectorized evaluation of hydraulic conductivity K(h, z) across depth vector z (NumPy version).
    
    h: ndarray of shape [..., N] or [N], matric potential values
    z: ndarray of same shape, depth values
    Returns: ndarray of same shape, K at each depth
    """
    if h.shape != z.shape:
        raise ValueError(f"h and z must have same shape, got {h.shape} vs {z.shape}")
    
    Kout = np.empty_like(h, dtype=float)

    layer_specs = [
        (0,   20,  'layer0-20'),
        (20,  60,  'layer20-60'),
        (60,  80,  'layer60-80'),
        (80,  90,  'layer80-90'),
        (90, 151,  'layer90-150'),
    ]

    covered = np.zeros_like(z, dtype=bool)
    for z_min, z_max, key in layer_specs:
        mask = (z >= z_min) & (z < z_max+1)
        if np.any(mask):
            p = param_multylayer[key]
            numer = p['Ks']
            denom = 1 + (p['B'] * np.abs(h[mask])) ** p['N']
            Kout[mask] = numer / denom
            covered |= mask

    if not np.all(covered):
        bad_z = z[~covered]
        raise ValueError(f"K is undefined for depths: {bad_z}")

    return Kout


def C_multy_np(h: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Derivative of theta_multy_np with respect to h, vectorized over depth z.

    h: ndarray [..., N] or [N]  (same shape as z)
    z: ndarray [..., N] or [N]
    returns: ndarray of same shape as h, dTheta/dh evaluated per layer.
    """
    if h.shape != z.shape:
        raise ValueError(f"h and z must have same shape, got {h.shape} vs {z.shape}")

    C = np.zeros_like(h, dtype=float)

    # (min_depth, max_depth, parameter-key) — same partitions used in theta_multy_np
    layer_specs = [
        (0,   20, 'layer0-20'),
        (20,  60, 'layer20-60'),
        (60,  80, 'layer60-80'),
        (80,  90, 'layer80-90'),
        (90, 151, 'layer90-150'),
    ]

    covered = np.zeros_like(z, dtype=bool)

    for z_min, z_max, key in layer_specs:
        mask = (z >= z_min) & (z < z_max) & (h < 0)
        if np.any(mask):
            p = param_multylayer[key]
            tr, ts, hg, n = p['tr'], p['ts'], p['hg'], p['n']
            m = 1.0 - (2.0 / n)
            A = ts - tr

            C[mask] = A * (-m) * (n) * (1/hg) * ((h[mask]/hg)**(n-1)) * ((1 +(h[mask]/hg)**(n) )**(-m-1))

            covered |= mask


    return C

def theta_vg(h, z=None, params=None):

    theta_s  = params['theta_s']
    theta_r  = params['theta_r']
    alpha    = params['alpha']
    n        = params['n']
    m        = 1.0 - 1.0/n

    # For h > 0 (ponding): return theta_s exactly, no gradient artifact
    h_neg    = torch.clamp(h, max=0.0)           # only use h ≤ 0
    term     = (1.0 + (alpha * torch.abs(h_neg))**n)**(-m)
    theta    = theta_r + (theta_s - theta_r) * term

    # Saturated when h > 0
    theta    = torch.where(h > 0, torch.full_like(h, theta_s), theta)
    return theta


def K_vg(h, z=None, params=None):

    theta_s = params['theta_s']
    theta_r = params['theta_r']
    Ks      = params['Ks']
    n       = params['n']
    m       = 1.0 - 1.0/n                        # defined once, used consistently

    Se      = (theta_vg(h, z, params) - theta_r) / (theta_s - theta_r)
    Se      = torch.clamp(Se, min=1e-8, max=1.0 - 1e-8)

    term1   = Ks * Se**0.5

    inner   = 1.0 - Se**(1.0/m)                  # 1 - Se^(1/m), cleaner than n/(n-1)
    inner   = torch.clamp(inner, min=1e-8, max=1.0 - 1e-8)  # prevents grad blow-up

    term2   = (1.0 - inner**m)**2

    K       = term1 * term2

    # Saturated when h > 0
    K       = torch.where(h > 0, torch.full_like(h, Ks), K)
    return K

def C_vg(h, z=None, params=None):
    """
    Capillary capacity C(h) = dtheta/dh  [1/m]
    Van Genuchten-Mualem analytical derivative.

    For h >= 0 (saturated): C = 0  (theta = theta_s = constant)
    For h <  0 (unsaturated): C > 0 always
    """
    
    theta_s = params['theta_s']
    theta_r = params['theta_r']
    alpha   = params['alpha']
    n       = params['n']
    m       = 1.0 - 1.0 / n

    # Work only on the unsaturated part
    h_neg  = torch.clamp(h, max=0.0)
    abs_h  = torch.abs(h_neg).clamp(min=1e-10)   # avoid 0^(n-1) when n < 1

    alpha_h_n   = (alpha * abs_h) ** n            # (α|h|)^n
    denom       = (1.0 + alpha_h_n) ** (m + 1.0) # [1 + (α|h|)^n]^(m+1)
    numerator   = m * n * (alpha ** n) * (abs_h ** (n - 1.0))

    C = (theta_s - theta_r) * numerator / denom

    # Saturated: C = 0
    C = torch.where(h >= 0.0, torch.zeros_like(h), C)

    return C






def grad(outputs, inputs):
    """
    Helper for auto-diff
    """
    derivation = torch.autograd.grad(outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        retain_graph= True, create_graph =True, allow_unused =True
    )[0]
    return derivation

def grad_fd(function , z = None , t = None , order = 1):
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


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
