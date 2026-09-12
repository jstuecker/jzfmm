"""Phase C: conservative log-interpolated timestep limits and decision plot."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def limit(rows,eps):
    rr=sorted(rows,key=lambda r:r['h'])
    h=np.array([r['h'] for r in rr]);e=np.maximum.accumulate([r['e_phase'] for r in rr])
    # Require a bracket; never extrapolate or accept isolated coarse-step dips.
    if eps<e[0] or eps>=e[-1]:return float('nan')
    j=np.searchsorted(e,eps,side='right')
    return float(np.exp(np.interp(np.log(eps),np.log(e[j-1:j+1]),np.log(h[j-1:j+1]))))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,default=Path('results'));ap.add_argument('--cost',type=Path,required=True)
    args=ap.parse_args();root=args.results;cost=json.loads(args.cost.read_text());R=cost['R_cost']
    rows=list(csv.DictReader((root/'analytic.csv').open()))
    for r in rows:
        for k in r.keys()-{'name','family','method'}:r[k]=float(r[k])
    names=sorted(set(r['name'] for r in rows));epsilons=np.geomspace(.001,.1,81);targets=[.1,.03,.01,.003,.001]
    out=[]
    for eps in sorted(set(list(epsilons)+targets)):
        for name in names:
            rs=[r for r in rows if r['name']==name]
            h2=limit([r for r in rs if r['method']=='KDK'],eps);h4=limit([r for r in rs if r['method']=='S4G'],eps)
            out.append(dict(name=name,family=rs[0]['family'],epsilon=eps,h2=h2,h4=h4,timestep_gain=h4/h2,R_cost=R,speedup=h4/h2/R))
    with (root/'speedups.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=out[0],lineterminator='\n');w.writeheader();w.writerows(out)
    summary={str(eps):dict(median=float(np.nanmedian([r['speedup'] for r in out if r['epsilon']==eps])),p16=float(np.nanpercentile([r['speedup'] for r in out if r['epsilon']==eps],16)),p84=float(np.nanpercentile([r['speedup'] for r in out if r['epsilon']==eps],84)),n_above_2=sum(r['speedup']>2 for r in out if r['epsilon']==eps)) for eps in targets}
    (root/'decision.json').write_text(json.dumps(dict(cost_file=args.cost.name,R_cost=R,targets=summary,promising=any(v['median']>2 for v in summary.values())),indent=2))
    fig,(a,b)=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    colors={'KDK':'#2166ac','S4G':'#b2182b'};styles={'circular':'-','moderate':'--','strong':':'}
    for name in ['ra1_q1','ra1_q0.2','ra1_q0.05']:
        for method in ['KDK','S4G']:
            rr=sorted([r for r in rows if r['name']==name and r['method']==method],key=lambda r:r['h'])
            a.loglog([(1 if method=='KDK' else R)/r['h'] for r in rr],[r['e_phase'] for r in rr],color=colors[method],ls=styles[rr[0]['family']],alpha=.9,lw=1.6)
    from matplotlib.lines import Line2D
    a.legend(handles=[Line2D([],[],color=c,label=m) for m,c in colors.items()]+[Line2D([],[],color='gray',ls=s,label=f) for f,s in styles.items()],fontsize=8)
    a.set(xlabel='Projected work / unit time [KDK-step equivalents]',ylabel='Final dimensionless phase-space error',ylim=(1e-4,1),xlim=(1,1000),title='10 periods; representative rₐ=1 orbits')
    es=sorted(set(r['epsilon'] for r in out))
    for fam,col in [('all','black'),('circular','#7b3294'),('strong','#008837')]:
        ys=np.array([[r['speedup'] for r in out if r['epsilon']==e and (fam=='all' or r['family']==fam)] for e in es])
        b.semilogx(es,np.nanmedian(ys,axis=1),color=col,label=fam+' median')
        if fam=='all':b.fill_between(es,np.nanpercentile(ys,16,axis=1),np.nanpercentile(ys,84,axis=1),color=col,alpha=.15,label='all 16–84%')
    for y in [1,2]:b.axhline(y,color='gray',ls='--',lw=.8)
    b.set(xlabel='Target phase-space error',ylabel='Projected S4G speed-up',title=f'GPU cost ratio {R:.2f}; N={cost["n"]:,}, {cost["dtype"]}')
    b.legend(fontsize=8);a.grid(alpha=.2);b.grid(alpha=.2)
    for ext in ['pdf','png']:fig.savefig(root/f'decision.{ext}',dpi=180)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
