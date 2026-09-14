import sys, numpy as np
sys.path.insert(0,'src')
from warningshot.data import modalities as mo, events as ev
from warningshot.core import regression as rg

rng=np.random.default_rng(7)
def block_boot_ci(t,y,fitter,key,block=4,n=1500):
    f=fitter(t,y)
    pred=np.log(f["amplitude"]) - t/f["tau"]
    resid=np.log(y)-pred
    N=len(resid); vals=[]
    nb=int(np.ceil(N/block))
    for _ in range(n):
        starts=rng.integers(0,max(1,N-block+1),size=nb)
        r=np.concatenate([resid[s:s+block] for s in starts])[:N]
        g=fitter(t,np.exp(pred+r))
        if g and np.isfinite(g.get(key,np.nan)): vals.append(g[key])
    return (float(np.percentile(vals,2.5)), float(np.percentile(vals,97.5))) if len(vals)>50 else None

def prep(ch, hourly=False):
    if hourly:
        y=np.array(ch["hourly"],dtype=float); t=np.arange(1.0,y.size+1.0)
    else:
        t,y=ch["t"],ch["y"]
    k=y>0; return t[k],y[k]

cases=[("lookups", prep(mo.lookups("Hugging_Face",ev.JULY_INCIDENT.t0,horizon=25)), 1.0),
       ("media",   prep(mo.media('"Hugging Face" OpenAI',ev.JULY_INCIDENT.t0,horizon=25)), 1.0),
       ("engagement", prep(mo.engagement("48997548"),hourly=True), 24.0)]

print(f"{'channel':12s}{'point':>8}{'paper iid CI':>22}{'block CI (b=4)':>24}{'width x':>9}")
out={}
for name,(t,y),div in cases:
    f=rg.fit_exponential(t,y)
    iid=rg.bootstrap_ci(t,y,rg.fit_exponential,"half_life",n=1500,seed=1)
    blk=block_boot_ci(t,y,rg.fit_exponential,"half_life")
    pi=f["half_life"]/div
    ii=(iid[0]/div,iid[1]/div); bb=(blk[0]/div,blk[1]/div)
    out[name]=bb
    print(f"{name:12s}{pi:8.3f}  ({ii[0]:6.3f},{ii[1]:6.3f})      ({bb[0]:6.3f},{bb[1]:6.3f})   "
          f"{(bb[1]-bb[0])/(ii[1]-ii[0]):6.2f}")
print()
a,b=out["lookups"],out["media"]
ov = a[0]<=b[1] and b[0]<=a[1]
print(f"lookups vs media overlap under BLOCK bootstrap: {'YES (still not shown to differ)' if ov else 'NO -- conclusion CHANGES'}")
a,c=out["lookups"],out["engagement"]
print(f"lookups vs engagement overlap: {'YES' if (a[0]<=c[1] and c[0]<=a[1]) else 'NO (still separated)'}")
