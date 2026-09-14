import sys, numpy as np
sys.path.insert(0,'src')
from warningshot.data import modalities as mo, events as ev
from warningshot.core import regression as rg

def prep(ch,hourly=False):
    if hourly:
        y=np.array(ch["hourly"],dtype=float); t=np.arange(1.0,y.size+1.0)
    else: t,y=ch["t"],ch["y"]
    k=y>0; return t[k],y[k]

print(f"{'channel':12s}{'n':>4}{'cands':>7}{'AIC gap':>10}{'sel.penalty':>13}{'survives?':>11}")
for name,(t,y),ms in [("lookups",prep(mo.lookups('Hugging_Face',ev.JULY_INCIDENT.t0,horizon=25)),5),
                      ("media",prep(mo.media('"Hugging Face" OpenAI',ev.JULY_INCIDENT.t0,horizon=25)),5),
                      ("engagement",prep(mo.engagement('48997548'),hourly=True),6)]:
    n=t.size; cands=max(0,n-2*ms+1)
    tp=rg.fit_two_phase(t,y,min_seg=ms); ex=rg.fit_exponential(t,y)
    gap=ex["aic"]-tp["aic"]; pen=2*np.log(cands) if cands>1 else 0.0
    print(f"{name:12s}{n:>4}{cands:>7}{gap:>10.2f}{pen:>13.2f}{('YES' if gap>pen else 'NO'):>11}")
