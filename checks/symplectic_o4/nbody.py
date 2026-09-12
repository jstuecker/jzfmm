"""Conditional Phase D: equal-mass isotropic Hernquist comparison.

Unsoftened isotropic DF initial conditions; softening and r<100 truncation
mean near-equilibrium, not exact equilibrium for the softened Hamiltonian.
All methods share the same ICs. No claim of individual long-time orbit fidelity.
"""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('JAX_PLATFORMS','cuda,cpu')
import argparse,json,time
from pathlib import Path
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
from jzfmm.config import FMMConfig,PlummerKernel,OpeningByAngle,SimConfig,KDKConfig,UnitConfig
from jzfmm.data import Particles
from jzfmm.time_integration import timestep,force_and_potential
from integrators import fmm_s4g


def df(E):
    q=np.sqrt(np.clip(E,1e-14,1-1e-14))
    # Hernquist (1990), eq. 17; omitted positive normalization cancels.
    return (3*np.arcsin(q)+q*np.sqrt(1-q*q)*(1-2*q*q)*(8*q**4-8*q*q-3))/(1-q*q)**2.5


def initial(n):
    rng=np.random.default_rng(20260913)
    u=rng.uniform(0,(100/101)**2,n);r=np.sqrt(u)/(1-np.sqrt(u))
    direction=rng.normal(size=(n,3));direction/=np.linalg.norm(direction,axis=1)[:,None]
    x=r[:,None]*direction;psi=1/(1+r)
    vmax=np.sqrt(2*psi);v=np.zeros_like(x);pending=np.arange(n);attempts=0
    # Uniform volume proposal in the escape-velocity ball. f(E)<=f(psi)
    # is an exact bound for this monotone isotropic DF.
    while len(pending):
        s=rng.random(len(pending))**(1/3)
        accept=rng.random(len(pending))<df(psi[pending]*(1-s*s))/df(psi[pending])
        ids=pending[accept]
        d=rng.normal(size=(len(ids),3));d/=np.linalg.norm(d,axis=1)[:,None]
        v[ids]=d*(s[accept]*vmax[ids])[:,None]
        pending=pending[~accept];attempts+=1
        if attempts>1000000:raise RuntimeError('DF rejection sampler did not finish')
    x-=x.mean(0);v-=v.mean(0)
    return x.astype('float32'),v.astype('float32')


def configuration(theta):
    c=FMMConfig(p=5,opening=OpeningByAngle(theta),kernel=PlummerKernel(.01),alloc_fac_ilist=1024)
    c=replace(c,tree=replace(c.tree,alloc_fac_nodes=8.))
    return SimConfig(force=c,integrator=KDKConfig(),units=UnitConfig(mass_in_msol=1/4.30071057317063e-6))


def fields(x,v,phi):
    x=np.asarray(x,dtype=float);v=np.asarray(v,dtype=float);phi=np.asarray(phi,dtype=float)
    assert np.isfinite(x).all() and np.isfinite(v).all() and np.isfinite(phi).all()
    r=np.linalg.norm(x,axis=1); edges=np.geomspace(.01,100,33)
    radial=(x*v).sum(1)/r;vt2=(v*v).sum(1)-radial**2
    sigmar=[];sigmat=[];counts=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(r>=lo)&(r<hi);counts.append(int(mask.sum()))
        sigmar.append(float(np.var(radial[mask])) if mask.any() else None)
        sigmat.append(float(np.mean(vt2[mask])/2) if mask.any() else None)
    return dict(r_edges=edges.tolist(),cumulative_mass=[float(np.mean(r<e)) for e in edges],shell_count=counts,
                sigma_r2=sigmar,sigma_t2=sigmat,energy=float(np.mean(np.sum(v*v,1)/2+phi/2)),
                angular_momentum=np.mean(np.cross(x,v),axis=0).tolist(),com=x.mean(0).tolist(),com_velocity=v.mean(0).tolist())


def run_case(root,name,x,v,method,theta,h,repeats=1):
    dest=root/name;dest.mkdir(exist_ok=True)
    if (dest/'metrics.json').exists():return
    cfg=configuration(theta)
    p=Particles(pos=jnp.asarray(x),vel=jnp.asarray(v),mass=jnp.full(len(x),1/len(x)))
    init=jax.jit(lambda p:replace(p,loc=force_and_potential(p,cfg)))
    p=jax.block_until_ready(init(p));p0=p
    fn=(lambda p,h:timestep(p,h,cfg)) if method=='KDK' else (lambda p,h:fmm_s4g(p,h,cfg))
    # Static nsteps per output; compilation measured separately.
    batch=jax.jit(lambda p,h,n:jax.lax.fori_loop(0,n,lambda i,p:fn(p,h),p))
    start=time.perf_counter();jax.block_until_ready(batch(p,jnp.float32(h),1));compile_s=time.perf_counter()-start
    jax.block_until_ready(batch(p,jnp.float32(h),2))
    metrics=dict(name=name,method=method,theta=theta,h=h,n=len(x),config=repr(cfg),compile_s=compile_s,initial=fields(p.pos,p.vel,p.loc.potential()),outputs=[])
    last=0.
    for t in [.25,1.,2.,4.,6.]:
        nsteps=round((t-last)/h);assert abs(nsteps*h-(t-last))<1e-12
        jax.block_until_ready(p);start=time.perf_counter();p=jax.block_until_ready(batch(p,jnp.float32(h),nsteps));elapsed=time.perf_counter()-start
        vals=fields(p.pos,p.vel,p.loc.potential());vals.update(t=t,elapsed_s=elapsed,nsteps=nsteps)
        metrics['outputs'].append(vals)
        np.savez_compressed(dest/f't{t:g}.npz',x=np.asarray(p.pos),v=np.asarray(p.vel))
        print(name,t,elapsed,flush=True);last=t
    metrics['total_s']=sum(o['elapsed_s'] for o in metrics['outputs']);(dest/'metrics.json').write_text(json.dumps(metrics,indent=2))


def compare(root):
    ref=root/'reference_fine';meta=json.loads((ref/'metrics.json').read_text());rows=[]
    with np.load(root/'initial.npz') as ic:
        rscale=np.maximum(np.linalg.norm(ic['x'].astype(float),axis=1),.05)
    vscale=np.sqrt(rscale)/(1+rscale)
    for directory in sorted(root.iterdir()):
        if not (directory/'metrics.json').exists():continue
        m=json.loads((directory/'metrics.json').read_text())
        runtime_to_output=0.
        for o,ro in zip(m['outputs'],meta['outputs']):
            runtime_to_output+=o['elapsed_s']
            with np.load(directory/f't{o["t"]:g}.npz') as d,np.load(ref/f't{o["t"]:g}.npz') as rr:
                # Global scales a=1 and circular speed at a=1, vscale=0.5.
                err=np.sqrt(np.sum((d['x'].astype(float)-rr['x'])**2,axis=1)+np.sum((d['v'].astype(float)-rr['v'])**2,axis=1)/.25)
                local_err=np.sqrt(np.sum((d['x'].astype(float)-rr['x'])**2,axis=1)/rscale**2+np.sum((d['v'].astype(float)-rr['v'])**2,axis=1)/vscale**2)
            e0=m['initial']['energy'];prof=np.array(o['cumulative_mass'])-ro['cumulative_mass']
            rows.append(dict(case=m['name'],method=m['method'],theta=m['theta'],h=m['h'],t=o['t'],phase_rms=float(np.sqrt(np.mean(err**2))),phase_median=float(np.median(err)),local_phase_rms=float(np.sqrt(np.mean(local_err**2))),phase_p90=float(np.percentile(err,90)),phase_p99=float(np.percentile(err,99)),mass_profile_max_abs=float(np.max(abs(prof))),energy_relative=(o['energy']-e0)/abs(e0),com_drift=float(np.linalg.norm(np.array(o['com'])-m['initial']['com'])),L_drift=float(np.linalg.norm(np.array(o['angular_momentum'])-m['initial']['angular_momentum'])),runtime_to_output_s=runtime_to_output,total_runtime_s=m['total_s']))
    import csv
    with (root/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--n',type=int,default=100000);ap.add_argument('--case',default='all');args=ap.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=True)
    if not (root/'initial.npz').exists():
        x,v=initial(args.n);np.savez_compressed(root/'initial.npz',x=x,v=v)
    else:
        with np.load(root/'initial.npz') as f:x=f['x'];v=f['v']
        assert len(x)==args.n
    cases=[('reference_coarse','S4G',.4,1/128),('reference_fine','S4G',.4,1/256)]
    for theta in [.8,.5]:
        cases += [(f'{method}_theta{theta:g}_d{d}',method,theta,1/d) for method,ds in [('KDK',[16,32,64,128,256,512]),('S4G',[4,8,16,32,64,128])] for d in ds]
    for name,method,theta,h in cases:
        if args.case in ['all',name]:run_case(root,name,x,v,method,theta,h)
    if args.case=='all':compare(root)

if __name__=='__main__':main()
