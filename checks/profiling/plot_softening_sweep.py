"""Plot a completed sweep_softening.py run and retain compact measured data."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/bender-matplotlib')
import argparse
import csv
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    meta=json.loads((args.run/'sweep.json').read_text())
    series=[('leaf_leaf_ms','Near field','#d55e00')]+[(f'plane_{p}_ms',f'Node2Node plane {p}',c) for p,c in enumerate(['#0072b2','#009e73','#cc79a7','#e69f00'])]
    rows=[]
    for i in range(len(meta['ratios'])):
        source=args.run/f'point_{i:02d}';dest=args.output/source.name;dest.mkdir(exist_ok=True)
        for f in ['timings.json','trace_summary.json','trace_samples.csv','plane_details.csv']:
            if (source/f).resolve()!=(dest/f).resolve():shutil.copy2(source/f,dest/f)
        t=json.loads((source/'timings.json').read_text());s=json.loads((source/'trace_summary.json').read_text())
        samples=t['timings']['walk_near']['ms']
        row=dict(epsilon_over_spacing=t['ratio'],epsilon=t['epsilon'],total_ms=float(np.median(samples)),total_min_ms=min(samples),total_max_ms=max(samples))
        for key,_,_ in series:
            for stat in ['median','min','max']:row[f'{key}_{stat}']=s[key][stat]
        row['gpu_total_ms']=s['gpu_total_ms']['median'];row['interaction_sum_ms']=s['interaction_sum_ms']['median'];row['profiled_wall_ms']=s['profiled_wall_ms']['median']
        rows.append(row)
    if (args.run/'sweep.json').resolve()!=(args.output/'sweep.json').resolve():shutil.copy2(args.run/'sweep.json',args.output/'sweep.json')
    with (args.output/'results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    x=np.array([r['epsilon_over_spacing'] for r in rows])
    fig,axs=plt.subplots(1,2,figsize=(13,4.8),layout='constrained',sharex=True)
    for ax in axs:
        for key,label,color in series:
            y=np.array([r[f'{key}_median'] for r in rows]);lo=[r[f'{key}_min'] for r in rows];hi=[r[f'{key}_max'] for r in rows]
            ax.plot(x,y,'o-',label=label,color=color,markersize=3,linewidth=1.5)
            ax.fill_between(x,lo,hi,color=color,alpha=.1)
        y=[r['total_ms'] for r in rows]
        ax.plot(x,y,'o-',color='black',label='Tree walk + near field (total)',linewidth=2.1,markersize=3)
        ax.fill_between(x,[r['total_min_ms'] for r in rows],[r['total_max_ms'] for r in rows],color='black',alpha=.1)
        ax.set_xscale('log');ax.set_xlim(.1,100);ax.set_xlabel(r'Softening / mean particle separation, $\epsilon/\ell$')
        ax.grid(which='major',alpha=.2);ax.set_xticks([.1,1,10,100],['0.1','1','10','100'])
    axs[0].set_ylim(0,175);axs[0].set_ylabel('Time per evaluation (ms)');axs[0].set_title('Absolute cost')
    axs[1].set_yscale('log');axs[1].set_ylim(1.5,190);axs[1].set_title('Logarithmic time scale');axs[1].legend(loc='upper left',bbox_to_anchor=(1.02,1),fontsize=8)
    fig.suptitle('4 million uniform particles · GTX 1070 · softened opening (θ = 0.8, p = 5)',fontsize=12)
    fig.supxlabel('Total starts from a ready tree and multipoles; includes downward translations. Shading: observed min–max.',fontsize=9)
    for ext in ['png','pdf']:fig.savefig(args.output/f'epsilon_sweep.{ext}',dpi=190)
    print(json.dumps([rows[i] for i in [0,6,9,12,18]],indent=2))


if __name__=='__main__':main()
