"""Independent full jzfmm step validation against a softened binary ODE."""
import os
os.environ.setdefault('JAX_PLATFORMS','cuda,cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
from pathlib import Path
import json
import numpy as np
from scipy.integrate import solve_ivp
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from dataclasses import replace
from jzfmm.config import SimConfig,DirectSummationConfig,PlummerKernel,KDKConfig,UnitConfig
from jzfmm.data import Particles
from jzfmm.time_integration import force_and_potential,timestep
from integrators import fmm_s4g


def main():
    eps=.03; cfg=SimConfig(force=DirectSummationConfig(kernel=PlummerKernel(eps)),integrator=KDKConfig(),units=UnitConfig(mass_in_msol=1/4.30071057317063e-6))
    x=np.array([[-.5,0.,0.],[.5,0.,0.]])
    # Eccentric initial conditions exercise the changing force gradient.
    v=np.array([[0.,-.3,0.],[0.,.3,0.]])
    def rhs(t,y):
        xx=y[:6].reshape(2,3); vv=y[6:].reshape(2,3)
        d=xx[1]-xx[0]; a=.5*d/(np.dot(d,d)+eps**2)**1.5
        return np.r_[vv.ravel(),a,-a]
    ref=solve_ivp(rhs,[0,2.],np.r_[x.ravel(),v.ravel()],method='DOP853',rtol=2.3e-14,atol=1e-15).y[:,-1]
    p=Particles(pos=jnp.array(x),vel=jnp.array(v),mass=jnp.full(2,.5))
    p=replace(p,loc=force_and_potential(p,cfg));results={}
    for name,step in [('KDK',timestep),('S4G',fmm_s4g)]:
        integrate=jax.jit(lambda p,h,n:jax.lax.fori_loop(0,n,lambda i,p:step(p,h,cfg),p))
        errors=[]
        for n in [128,256,512,1024]:
            out=jax.block_until_ready(integrate(p,2/n,n))
            errors.append(float(np.linalg.norm(np.r_[np.asarray(out.pos).ravel(),np.asarray(out.vel).ravel()]-ref)))
        back=integrate(out,-2/1024,1024)
        rev=float(np.linalg.norm(np.r_[np.asarray(back.pos-x).ravel(),np.asarray(back.vel-v).ravel()]))
        order=float(np.polyfit(np.log(2/np.array([128,256,512,1024])),np.log(errors),1)[0])
        results[name]=dict(errors=errors,order=order,reversibility=rev)
        assert abs(order-(2 if name=='KDK' else 4))<.1 and rev<1e-10
    Path(__file__).with_name('results').joinpath('binary_validation.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
