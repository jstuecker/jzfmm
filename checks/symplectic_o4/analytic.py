"""Phase A: CPU float64 analytic orbit accuracy (GPU costs measured separately)."""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import argparse, csv, json
from pathlib import Path
from functools import partial
import numpy as np
from scipy.integrate import solve_ivp, quad
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from integrators import kdk, s4g, force_gradient


def acc(x):
    r = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return -x / (r * (1+r)**2)


def energy(x,v):
    return (v*v).sum(-1)/2 - 1/(1+jnp.linalg.norm(x,axis=-1))


def suite():
    out=[]
    for ra, ratio in [(r,1.) for r in [.05,.1,.3,1.,3.,10.]] + [(r,q) for r in [.1,1.,10.] for q in [.5,.2,.05]]:
        rp=ra*ratio
        L2=ra**3/(1+ra)**2 if ratio==1 else 2*(-1/(1+rp)+1/(1+ra))/(1/ra**2-1/rp**2)
        vt=np.sqrt(L2)/ra
        E=vt**2/2-1/(1+ra)
        if ratio==1:
            T=2*np.pi*ra/vt
        else:
            # r = mid - half*cos(u) removes turning-point singularities.
            def f(u):
                r=(ra+rp)/2-(ra-rp)/2*np.cos(u)
                return (ra-rp)/2*np.sin(u)/np.sqrt(max(2*(E+1/(r+1))-L2/r**2,1e-300))
            T=2*quad(f,0,np.pi,epsabs=1e-10,epsrel=1e-10)[0]
        out.append(dict(name=f'ra{ra:g}_q{ratio:g}',ra=ra,rp=rp,ratio=ratio,T=T,vt=vt,
                        family='circular' if ratio==1 else 'strong' if ratio==.05 else 'moderate'))
    return out


@partial(jax.jit, static_argnames=('n','method'))
def integrate(x,v,h,n,method):
    e0=energy(x,v); L0=jnp.cross(x,v)
    r=jnp.linalg.norm(x,axis=-1)
    state=(x,v,acc(x),jnp.zeros_like(r),jnp.zeros_like(r),r,r,jnp.zeros_like(r))
    step=kdk if method=='KDK' else s4g
    def body(i,state):
        x,v,a,em,lm,rmin,rmax,ang=state
        xn,vn,an=step(x,v,a,h,acc)
        rn=jnp.linalg.norm(xn,axis=-1)
        da=jnp.arctan2(x[:,0]*xn[:,1]-x[:,1]*xn[:,0],(x*xn).sum(-1))
        return (xn,vn,an,jnp.maximum(em,jnp.abs((energy(xn,vn)-e0)/e0)),
                jnp.maximum(lm,jnp.linalg.norm(jnp.cross(xn,vn)-L0,axis=-1)/jnp.linalg.norm(L0,axis=-1)),
                jnp.minimum(rmin,rn),jnp.maximum(rmax,rn),ang+da)
    return jax.lax.fori_loop(0,n,body,state)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=Path('results')); ap.add_argument('--max-denom',type=int,default=32768)
    args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    orbits=suite(); ra=np.array([o['ra'] for o in orbits]); vs=np.sqrt(ra)/(1+ra)
    T=np.array([o['T'] for o in orbits]); x=np.zeros((15,3));x[:,0]=ra
    v=np.zeros_like(x);v[:,1]=[o['vt'] for o in orbits]
    refs=[]; refangles=[]; refchecks=[]
    for o,xx,vv in zip(orbits,x,v):
        def rhs(t,y):
            r=np.linalg.norm(y[:3]);return np.r_[y[3:],-y[:3]/(r*(1+r)**2)]
        sols=[]
        for tol in [2e-11,2.3e-14]:
            sols.append(solve_ivp(rhs,[0,10*o['T']],np.r_[xx,vv],method='DOP853',rtol=tol,atol=tol*min(o['rp'],o['vt'],1)*.01,dense_output=True))
            assert sols[-1].success
        yy=sols[-1].y[:,-1]; refs.append(yy)
        # Dense samples chosen to resolve even strongest pericentres; check
        # unwrapped angular phase using twice as many samples as well.
        angles=[]
        for count in [65537,131073]:
            zz=sols[-1].sol(np.linspace(0,10*o['T'],count))
            angles.append(float(np.unwrap(np.arctan2(zz[1],zz[0]))[-1]))
        assert abs(angles[0]-angles[1])<1e-10
        refangles.append(angles[-1])
        d=sols[0].y[:,-1]-yy
        refchecks.append(float(np.sqrt(np.sum(d[:3]**2)/o['ra']**2+np.sum(d[3:]**2)/(o['ra']/(1+o['ra'])**2))))
    refs=np.array(refs)
    # Independent radial Hessian contraction: J_a a = -2 x/[r(r+1)^5].
    rng=np.random.default_rng(123); z=rng.normal(size=(100,3));z*=np.exp(rng.uniform(-3,3,(100,1)))
    aa,gg=force_gradient(acc,jnp.array(z)); rr=np.linalg.norm(z,axis=1,keepdims=True)
    analytic=-2*z/(rr*(1+rr)**5)
    fd=(np.array(acc(z+1e-5*np.array(aa)))-np.array(acc(z-1e-5*np.array(aa))))/(2e-5)
    gradcheck=dict(vjp_relative=float(np.linalg.norm(gg-analytic)/np.linalg.norm(analytic)),finite_difference_relative=float(np.linalg.norm(fd-analytic)/np.linalg.norm(analytic)))
    assert gradcheck['vjp_relative']<1e-12 and gradcheck['finite_difference_relative']<1e-5
    rows=[]; denoms=sorted(set([4,6,8,12,16,24,32,48,64,96,128]+[int(q*2**i) for i in range(6,15) for q in [2,3] if q*2**i<=args.max_denom]))
    for denom in denoms:
        for method in ['KDK','S4G']:
            result=integrate(jnp.array(x),jnp.array(v),jnp.array(T[:,None]/denom),n=10*denom,method=method)
            xx,vv,_,em,lm,rmin,rmax,ang=map(np.array,result)
            err=np.sqrt(((xx-refs[:,:3])**2).sum(1)/ra**2+((vv-refs[:,3:])**2).sum(1)/vs**2)
            ef=np.abs((np.array(energy(xx,vv))-np.array(energy(x,v)))/np.array(energy(x,v)))
            for i,o in enumerate(orbits):
                rows.append(dict(**o,method=method,denom=denom,h=T[i]/denom,e_phase=err[i],energy_max=em[i],energy_final=ef[i],L_max=lm[i],pericentre_error=(rmin[i]-o['rp'])/o['rp'],apocentre_error=(rmax[i]-o['ra'])/o['ra'],angular_phase_error=ang[i]-refangles[i]))
        print('denom',denom,'max S4G error',float(err.max()),flush=True)
    with (args.output/'analytic.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
    validation={'gradient':gradcheck,'reference_phase_difference':dict(zip([o['name'] for o in orbits],refchecks)),'orders':{},'reversibility':{}}
    for method in ['KDK','S4G']:
        orders={}
        for o in orbits:
            rr=[r for r in rows if r['name']==o['name'] and r['method']==method and 1e-10<r['e_phase']<.001]
            rr=rr[-4:]
            orders[o['name']]=float(np.polyfit(np.log([r['h'] for r in rr]),np.log([r['e_phase'] for r in rr]),1)[0]) if len(rr)>=3 else None
        validation['orders'][method]=orders
        h=jnp.array(T[:,None]/4096)
        forward=integrate(jnp.array(x),jnp.array(v),h,n=4096,method=method)
        backward=integrate(forward[0],forward[1],-h,n=4096,method=method)
        validation['reversibility'][method]=float(np.max(np.sqrt(np.sum((np.array(backward[0])-x)**2,1)/ra**2+np.sum((np.array(backward[1])-v)**2,1)/vs**2)))
    np.savez_compressed(args.output/'references.npz',x0=x,v0=v,reference=refs,period=T,angular_phase=refangles)
    (args.output/'validation.json').write_text(json.dumps(validation,indent=2))
    print(json.dumps(validation,indent=2),flush=True)
    assert all(s is not None and abs(s-order)<.25 for m,order in [('KDK',2),('S4G',4)] for s in validation['orders'][m].values())

if __name__=='__main__':main()
