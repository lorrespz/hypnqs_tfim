import torch
from torch.optim import Optimizer
import poincare_util as util

class RSGD(Optimizer):
    def __init__(self, params, lr, c_val=1.0, hyp_opt='rsgd'):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        defaults = dict(lr=lr, c_val=c_val, hyp_opt=hyp_opt)
        super(RSGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            c = group['c_val']
            hyp_opt = group['hyp_opt']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                if hyp_opt == 'rsgd':
                    # True RSGD: var = Exp_map(var, -lr * Riemannian_grad)
                    # Note: For Poincare ball, the Riemannian gradient is 
                    # usually the Euclidean gradient scaled by (1/lambda_x^2)
                    # However, to match your TF logic:
                    p.copy_(util.th_exp_map_x(p, -lr * grad, c))
                else:
                    # Retraction-based RSGD: Euclidean step + Projection
                    new_p = p - lr * grad
                    # Project to ensure it stays within the Poincare Ball (radius 1/sqrt(c))
                    p.copy_(util.th_project_hyp_vecs(new_p, c))

        return loss