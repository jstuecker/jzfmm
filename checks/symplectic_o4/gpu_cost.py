"""Phases B: synchronized timings, equal-mass snapshots and derivative validation."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('JAX_PLATFORMS','cuda,cpu')
import argparse, json, time, platform, subprocess
from pathlib import Path
from dataclasses import replace
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jzfmm, jztree, jzfmm_cuda
from jzfmm.config import FMMConfig, OpeningByAngle, PlummerKernel, SimConfig, KDKConfig, UnitConfig, DirectSummationConfig
from jzfmm.data import Particles, PosMass
from jzfmm.fmm import fast_multipole_method, direct_summation
from jzfmm.time_integration import timestep, force_and_potential
from integrators import fmm_s4g, force_gradient


def snapshot(n,dtype):
    rng=np.random.default_rng(20260913)
    # Conditional Hernquist spatial distribution r<100a; tail truncation avoids
    # pathological root extents, retains 98.03% of the untruncated mass.
    u=rng.uniform(0,(100/101)**2,n); r=np.sqrt(u)/(1-np.sqrt(u))
    direction=rng.normal(size=(n,3));direction/=np.linalg.norm(direction,axis=1)[:,None]
    x=r[:,None]*direction
    # Representative velocities for cost only; this is NOT equilibrium ICs.
    v=rng.normal(size=(n,3))*np.sqrt(r/(1+r)**2)[:,None]/np.sqrt(3)
    return jnp.asarray(x,dtype),jnp.asarray(v,dtype)


def measure(fn,args,repeats):
    jax.block_until_ready(args)
    start=time.perf_counter();out=jax.block_until_ready(fn(*args)); compile_s=time.perf_counter()-start
    for _ in range(3): out=jax.block_until_ready(fn(*args))
    samples=[]
    for _ in range(repeats):
        jax.block_until_ready(args);start=time.perf_counter();out=jax.block_until_ready(fn(*args));samples.append(time.perf_counter()-start)
    assert all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(out))
    return dict(samples_s=samples,median_s=float(np.median(samples)),p16_s=float(np.percentile(samples,16)),p84_s=float(np.percentile(samples,84)),first_call_s=compile_s)


def validate(cfg):
    x,_=snapshot(512,jnp.float64);mass=jnp.full(512,1/512,dtype=x.dtype)
    direct=lambda x:direct_summation(PosMass(pos=x,mass=mass),DirectSummationConfig(kernel=cfg.kernel)).force()
    fmm=lambda x:fast_multipole_method(PosMass(pos=x,mass=mass),cfg).force()
    ac, gc=jax.jit(lambda x:force_gradient(direct,x))(x)
    af,gf=jax.jit(lambda x:force_gradient(fmm,x))(x)
    # Independent Plummer J_a a, including motion of both source and target.
    z=np.array(x);a=np.array(ac);d=z[None,:,:]-z[:,None,:]; da=a[None,:,:]-a[:,None,:]
    r2=np.sum(d*d,axis=-1)+cfg.kernel.softening**2
    exact=np.sum(da/r2[:,:,None]**1.5-3*d*np.sum(d*da,axis=-1)[:,:,None]/r2[:,:,None]**2.5,axis=1)/512
    norm=lambda z:float(np.linalg.norm(z))
    checks={'direct_vjp_vs_pair_formula':norm(gc-exact)/norm(exact),'fmm_force_vs_direct':norm(af-ac)/norm(ac),'fmm_gradient_vs_direct':norm(gf-gc)/norm(gc)}
    for delta in [1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]:
        fd=(fmm(x+delta*af)-fmm(x-delta*af))/(2*delta)
        checks[f'fmm_vjp_vs_fd_jvp_{delta:g}']=norm(fd-gf)/norm(gf)
    assert checks['direct_vjp_vs_pair_formula']<1e-10
    return checks


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=100000);ap.add_argument('--dtype',choices=['float32','float64'],default='float32');ap.add_argument('--theta',type=float,default=.8);ap.add_argument('--repeats',type=int,default=21);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args(); args.output.parent.mkdir(parents=True,exist_ok=True)
    cfg=FMMConfig(p=5,opening=OpeningByAngle(args.theta),kernel=PlummerKernel(.01),alloc_fac_ilist=256)
    cfg=replace(cfg,tree=replace(cfg.tree,alloc_fac_nodes=8.0))
    result=dict(n=args.n,dtype=args.dtype,config=repr(cfg),gpu=str(jax.devices()),jax=jax.__version__,jzfmm_path=jzfmm.__file__,cuda_path=jzfmm_cuda.__file__,jztree_path=jztree.__file__,platform=platform.platform(),seed=20260913,truncation_radius=100.,timings={})
    result['validation']=validate(cfg); print(result['validation'],flush=True)
    x,v=snapshot(args.n,getattr(jnp,args.dtype));mass=jnp.full(args.n,1/args.n,dtype=x.dtype)
    acceleration=lambda x:fast_multipole_method(PosMass(pos=x,mass=mass),cfg).force()
    force=jax.jit(acceleration)
    combined=jax.jit(lambda x:force_gradient(acceleration,x))
    # Contraction given a cached cotangent, includes rebuilding primal tape.
    contraction=jax.jit(lambda x,a:jax.vjp(acceleration,x)[1](a)[0])
    a=force(x);jax.block_until_ready(a)
    sim=SimConfig(force=cfg,integrator=KDKConfig(),units=UnitConfig(mass_in_msol=1/4.30071057317063e-6))
    p=Particles(pos=x,vel=v,mass=mass);p=replace(p,loc=force_and_potential(p,sim));jax.block_until_ready(p)
    h=jnp.asarray(1e-3,dtype=x.dtype)
    step4=lambda p,h:fmm_s4g(p,h,sim)
    ops={'acceleration':(force,(x,)),'contraction_cached_a':(contraction,(x,a)),'combined_central':(combined,(x,)),
         'KDK_step':(jax.jit(lambda p,h:timestep(p,h,sim)),(p,h)),'S4G_step':(jax.jit(step4),(p,h))}
    for name,(fn,inputs) in ops.items():
        result['timings'][name]=measure(fn,inputs,args.repeats)
        print(name,result['timings'][name]['median_s'],flush=True)
        args.output.write_text(json.dumps(result,indent=2))
    # Amortized full sequence with actual changing positions and cached forces.
    for name,fn in [('KDK',lambda p,h:timestep(p,h,sim)),('S4G',step4)]:
        batch=jax.jit(lambda p,h:jax.lax.fori_loop(0,8,lambda i,p:fn(p,h),p))
        result['timings'][name+'_8steps']=measure(batch,(p,h),args.repeats)
        print(name+'_8steps',result['timings'][name+'_8steps']['median_s'],flush=True)
    result['R_cost']=result['timings']['S4G_8steps']['median_s']/result['timings']['KDK_8steps']['median_s']
    result['R_cost_single']=result['timings']['S4G_step']['median_s']/result['timings']['KDK_step']['median_s']
    args.output.write_text(json.dumps(result,indent=2)); print('R_cost',result['R_cost'],flush=True)

if __name__=='__main__':main()
