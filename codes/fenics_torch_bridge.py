# fenics_torch_bridge.py
import torch
import numpy as np
import dolfin_adjoint as da

torch.set_default_dtype(torch.float64)


class FenicsTopFluxMisfit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pde, C, sigma2, cP, cN):
        C_np = C.detach().cpu().numpy()
        cP_val = float(cP.detach().cpu().item())
        cN_val = float(cN.detach().cpu().item())

        J_val, tape, J_adj = pde.forward_record(
            C_np,
            sigma2=float(sigma2),
            cP=cP_val,
            cN=cN_val,
        )

        ctx.pde = pde
        ctx.tape = tape
        ctx.J_adj = J_adj
        ctx.sigma2 = float(sigma2)

        ctx.C_np = C_np
        ctx.cP_val = cP_val
        ctx.cN_val = cN_val

        ctx.C_device = C.device
        ctx.C_dtype = C.dtype
        ctx.scalar_device = cP.device
        ctx.scalar_dtype = cP.dtype

        return C.new_tensor(J_val)

    @staticmethod
    def backward(ctx, grad_out):
        pde = ctx.pde
    
        # -----------------------------------
        # 1) exact adjoint gradient wrt C
        # -----------------------------------
        ctrl_C = da.Control(pde.C)
        adj_value = da.AdjFloat(float(grad_out.item()))
    
        gC_fun = da.compute_gradient(
            ctx.J_adj,
            ctrl_C,
            tape=ctx.tape,
            adj_value=adj_value,
        )
    
        gC = gC_fun.vector().get_local()
        gC_torch = torch.from_numpy(np.asarray(gC, dtype=np.float64)).to(
            ctx.C_device,
            ctx.C_dtype,
        )
    
        # -----------------------------------
        # 2) scalar sensitivities wrt cp, cn
        #    only compute them if PyTorch needs them
        # -----------------------------------
        need_cP_grad = ctx.needs_input_grad[3]
        need_cN_grad = ctx.needs_input_grad[4]
    
        scale = float(grad_out.item())
    
        if need_cP_grad or need_cN_grad:
            zeros_C = np.zeros_like(ctx.C_np)
    
            if need_cP_grad:
                gcp_direct, _ = pde.tangent_misfit_from_params(
                    dC_vec=zeros_C,
                    cP=ctx.cP_val,
                    cN=ctx.cN_val,
                    dcP=1.0,
                    dcN=0.0,
                )
    
                gcP_torch = torch.tensor(
                    scale * gcp_direct,
                    dtype=ctx.scalar_dtype,
                    device=ctx.scalar_device,
                )
            else:
                gcP_torch = None
    
            if need_cN_grad:
                gcn_direct, _ = pde.tangent_misfit_from_params(
                    dC_vec=zeros_C,
                    cP=ctx.cP_val,
                    cN=ctx.cN_val,
                    dcP=0.0,
                    dcN=1.0,
                )
    
                gcN_torch = torch.tensor(
                    scale * gcn_direct,
                    dtype=ctx.scalar_dtype,
                    device=ctx.scalar_device,
                )
            else:
                gcN_torch = None
    
        else:
            gcP_torch = None
            gcN_torch = None
    
        return (None, gC_torch, None, gcP_torch, gcN_torch)
        

