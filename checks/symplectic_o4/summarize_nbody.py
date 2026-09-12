"""Plot actual N-body accuracy/work, profiles and matched-error runtime ratios."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/jzfmm-s4g-matplotlib')
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def matched_runtime(rr,eps,metric):
    # Start at the most accurate (smallest timestep), reject coarse pockets.
    rr=sorted(rr,key=lambda r:r['h'])
    e=np.maximum.accumulate([r[metric] for r in rr]);cost=np.array([r['runtime_to_output_s'] for r in rr])
    if eps<e[0] or eps>=e[-1]:return None
    j=np.searchsorted(e,eps,side='right')
    return float(np.exp(np.interp(np.log(eps),np.log(e[j-1:j+1]),np.log(cost[j-1:j+1]))))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,default=Path('results/nbody'));args=ap.parse_args();root=args.results
    rows=list(csv.DictReader((root/'comparison.csv').open()))
    for r in rows:
        for k in r.keys()-{'case','method'}:r[k]=float(r[k])
    speedups=[]
    for t in [.25,6.]:
        reference=next(r for r in rows if r['case']=='reference_coarse' and r['t']==t)
        for theta in [.8,.5]:
            for metric in ['phase_rms','local_phase_rms']:
                for eps in [.1,.03,.01,.003,.001,.0003,.0001,.00003,.00001]:
                    base=[r for r in rows if r['t']==t and r['theta']==theta]
                    c2=matched_runtime([r for r in base if r['method']=='KDK'],eps,metric)
                    c4=matched_runtime([r for r in base if r['method']=='S4G'],eps,metric)
                    speedups.append(dict(t=t,theta=theta,metric=metric,epsilon=eps,KDK_runtime=c2,S4G_runtime=c4,
                                         reference_difference=reference[metric],reference_resolved=eps>=10*reference[metric],
                                         speedup=c2/c4 if c2 and c4 else None))
    (root/'matched_speedups.json').write_text(json.dumps(speedups,indent=2))
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for t,ax in zip([.25,6.],axes[0]):
        for theta in [.8,.5]:
            for method,col in [('KDK','#2166ac'),('S4G','#b2182b')]:
                rr=sorted([r for r in rows if r['t']==t and r['theta']==theta and r['method']==method],key=lambda r:r['h'])
                ax.loglog([r['runtime_to_output_s'] for r in rr],[r['local_phase_rms'] for r in rr],color=col,ls='-' if theta==.5 else '--',marker='o',label=f'{method}, θ={theta}')
        coarse=next(r for r in rows if r['case']=='reference_coarse' and r['t']==t)
        ax.axhline(coarse['local_phase_rms'],color='gray',ls=':',label='reference refinement difference')
        ax.set(xlabel=f'Measured integration runtime to t={t:g} [s]',ylabel='RMS relative phase-space error',title=f't={t:g}; reference θ=0.4, h=1/256');ax.legend(fontsize=7)
    ref=json.loads((root/'reference_fine/metrics.json').read_text());edge=np.array(ref['outputs'][-1]['r_edges'])
    for name in ['reference_fine','KDK_theta0.5_d128','S4G_theta0.5_d32']:
        m=json.loads((root/name/'metrics.json').read_text());o=m['outputs'][-1]
        axes[1,0].semilogx(edge,o['cumulative_mass'],label=name)
        rad=np.sqrt(edge[:-1]*edge[1:]);valid=np.array(o['shell_count'])>=50
        axes[1,1].semilogx(rad[valid],np.array(o['sigma_r2'],dtype=float)[valid],label=name+' radial')
        axes[1,1].semilogx(rad[valid],np.array(o['sigma_t2'],dtype=float)[valid],ls='--',label=name+' tangential')
    axes[1,0].semilogx(edge,ref['initial']['cumulative_mass'],color='gray',ls=':',label='initial')
    axes[1,0].set(xlabel='Radius',ylabel='Cumulative mass',title='Ensemble profiles at t=6');axes[1,0].legend(fontsize=7)
    axes[1,1].set(xlabel='Radius',ylabel='Velocity dispersion squared',title='Shells with ≥50 particles');axes[1,1].legend(fontsize=6)
    for ax in axes.flat:ax.grid(alpha=.2)
    for ext in ['png','pdf']:fig.savefig(root/f'nbody.{ext}',dpi=180)
    print(json.dumps([r for r in speedups if r['speedup']],indent=2))

if __name__=='__main__':main()
