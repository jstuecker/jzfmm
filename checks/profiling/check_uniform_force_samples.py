"""Compare saved uniform-FMM outputs with direct Plummer sums at sampled targets."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from jztree_utils import ics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories',type=Path,nargs='+')
    parser.add_argument('--samples',type=int,default=128)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    configs=[json.loads((d/'timings.json').read_text()) for d in args.directories]
    n,seed=configs[0]['n'],configs[0]['seed']
    assert all((c['n'],c['seed'])==(n,seed) for c in configs)
    part=jax.block_until_ready(ics.uniform_particles(n,seed=seed))
    ids=np.sort(np.random.default_rng(123).choice(n,size=args.samples,replace=False))
    # Float64 direct reference; sample targets only, eight at a time.
    with jax.enable_x64():
        pos=part.pos.astype(jnp.float64)
        @jax.jit
        def direct(target,eps):
            dx=target[:,None,:]-pos[None,:,:]
            r2=jnp.sum(dx*dx,axis=-1)+eps*eps
            inv=jax.lax.rsqrt(r2)
            potential=-jnp.sum(inv,axis=1)/n + 1/(n*eps)
            gradient=jnp.sum(dx*(inv**3)[...,None],axis=1)/n
            return jnp.concatenate([potential[:,None],gradient],axis=1)
        output={'n':n,'seed':seed,'target_indices':ids.tolist(),'reference':'float64 direct Plummer sum; self potential removed','cases':{}}
        refs={}
        for d,c in zip(args.directories,configs):
            eps=c['config']['kernel']['softening']
            if eps not in refs:
                refs[eps]=np.concatenate([np.asarray(direct(pos[ids[i:i+8]],eps)) for i in range(0,len(ids),8)])
            ref=refs[eps]
            values=np.load(d/'values.npy',mmap_mode='r')[ids].astype(np.float64)
            error=values[:,1:]-ref[:,1:]
            relative=np.linalg.norm(error,axis=1)/np.linalg.norm(ref[:,1:],axis=1)
            record={'force_relative_l2':float(np.linalg.norm(error)/np.linalg.norm(ref[:,1:])),
                    'force_relative_error_median':float(np.median(relative)),
                    'force_relative_error_p95':float(np.quantile(relative,.95)),
                    'force_relative_error_max':float(np.max(relative)),
                    'potential_relative_l2':float(np.linalg.norm(values[:,0]-ref[:,0])/np.linalg.norm(ref[:,0]))}
            output['cases'][d.name]=record
            print(d.name,record,flush=True)
    args.output.write_text(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
