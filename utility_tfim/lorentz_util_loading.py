import types
import os
import sys
import importlib.util
import torch
import torch.nn as nn

def manual_load(module_name, path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = module_name.rpartition('.')[0]
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# 1. Create the shell structure
for pkg in ["hypercore", "hypercore.utils", "hypercore.manifolds"]:
    if pkg not in sys.modules:
        m = types.ModuleType(pkg)
        m.__path__ = []
        sys.modules[pkg] = m

# 2. Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.abspath(os.path.join(current_dir, "hypercore_main/hypercore"))

math_utils_path = os.path.join(base_path, "utils", "math_utils.py")
manifold_utils_path = os.path.join(base_path, "manifolds", "utils.py")
lmath_path = os.path.join(base_path, "manifolds", "lmath.py")
lorentz_path = os.path.join(base_path, "manifolds", "lorentzian.py")

# 3. Satisfy Import Demands
manual_load("hypercore.utils", math_utils_path)
manual_load("hypercore.manifolds.utils", manifold_utils_path)

# --- Define the Wrapper ---
def wrap_geoopt_style_old(method):
    def wrapper(*args, **kwargs):
        # 1. Strip curvature/normalization flags
        #kwargs.pop('k', None)
        #kwargs.pop('c', None)
        kwargs.pop('is_tan_normalize', None)

        # 2. Shape Safety: Ensure args are at least 2D (batch_size, dim)
        # We look for tensors in the arguments and unsqueeze them if 1D
        new_args = []
        for arg in args:
            if torch.is_tensor(arg) and arg.dim() == 1:
                new_args.append(arg.unsqueeze(0))
            else:
                new_args.append(arg)

        # 3. Handle bias and other tensors in kwargs similarly
        for key, value in kwargs.items():
            if torch.is_tensor(value) and value.dim() == 1:
                kwargs[key] = value.unsqueeze(0)

        result = method(*new_args, **kwargs)

        # If we unsqueezed it, we might want to squeeze it back,
        # but usually, keeping the batch dim is safer for NQS loops.
        return result
    return wrapper


import inspect

def wrap_geoopt_style(method):
    sig = inspect.signature(method)
    
    def wrapper(*args, **kwargs):
        # 1. Shape safety
        new_args = [arg.unsqueeze(0) if torch.is_tensor(arg) and arg.dim() == 1 else arg for arg in args]
        
        # 2. Logic: Only inject defaults for arguments the method actually declares
        # This prevents 'expmap0' from seeing 'is_tan_normalize' (which it hates)
        # while ensuring 'logmap0' gets it (which it requires)
        for param_name, param in sig.parameters.items():
            if param.kind == inspect.Parameter.KEYWORD_ONLY and param_name not in kwargs:
                # If it has a default, use it. If not, we might be in trouble, 
                # but for your functions, they usually have defaults.
                if param.default is not param.empty:
                    kwargs[param_name] = param.default
        
        # 3. Filter: Only pass what the method actually accepts
        # This keeps the call clean for expmap0
        final_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        
        return method(*new_args, **final_kwargs)
    return wrapper
# 4. Load Logic

# 5. Instantiate and Bind
lmath_mod = manual_load("hypercore.manifolds.lmath", lmath_path)
hc_math = manual_load("hypercore.manifolds.lorentzian", lorentz_path)
if hasattr(hc_math, 'Lorentz'):
    # Define the manifold instance
    lorentz_manifold = hc_math.Lorentz(c=1.0, learnable=False)

    # 6. Bind and Wrap simultaneously to fix the NameError
    hc_math.expmap = wrap_geoopt_style(lmath_mod.expmap)
    hc_math.logmap = wrap_geoopt_style(lmath_mod.logmap)
    hc_math.expmap0 = wrap_geoopt_style(lmath_mod.expmap0)
    hc_math.logmap0 = wrap_geoopt_style(lmath_mod.logmap0)
    hc_math.mobius_add = wrap_geoopt_style(lorentz_manifold.mobius_add)
    hc_math.mobius_matvec = wrap_geoopt_style(lorentz_manifold.mobius_matvec)
    hc_math.mobius_scalar_mult = wrap_geoopt_style(lorentz_manifold.mobius_scalar_mult)
    hc_math.mobius_add_clamped = wrap_geoopt_style(lorentz_manifold.mobius_add_clamped)
    hc_math.mobius_matvec_clamped = wrap_geoopt_style(lorentz_manifold.mobius_matvec_clamped)
    
    # Bridge Parallel Transport (Critical for GRUs)
    if hasattr(lorentz_manifold, 'ptransp0'):
        hc_math.ptransp0 = wrap_geoopt_style(lorentz_manifold.ptransp0)
    elif hasattr(lmath_mod, 'ptransp0'):
        # Fallback to lmath functional version if manifold lacks it
        hc_math.ptransp0 = lmath_mod.ptransp0

else:
    raise AttributeError("Lorentz class not found in lorentzian.py")

print("Hypercore Lorentzian module loaded successfully with Geoopt wrappers.")
