import sys, numpy as np, statistics
sys.path.insert(0,'src')
from warningshot.data import sources as src, events as ev
from warningshot.core import metrics as m

print("="*72); print("ATTACK 3 — placebo null seasonality (is Jan-May exchangeable with Jul-Aug?)"); print("="*72)
CONC=["AI_safety","AI_alignment","Existential_risk_from_artificial_intelligence","Artificial_general_intelligence"]
for p in CONC:
    s=src.pageviews(p,"20260101","20260903")
    if not s: continue
    win=lambda a,b:[v for k,v in s.items() if a<=k<=b]
    jan_may=statistics.median(win("20260115","20260531"))
    jun_jul=statistics.median(win("20260615","20260714"))
    print(f"  {p:46s} Jan-May median={jan_may:7.0f}  pre-event median={jun_jul:7.0f}  ratio={jun_jul/jan_may:.2f}x")

print(); print("="*72); print("ATTACK 4 — Kimi-K3 contamination of the Hugging_Face decay window"); print("="*72)
hf=src.pageviews("Hugging_Face","20260701","20260903")
print("  HN 2026-07-27: 'Kimi-K3 on HuggingFace' 1,382 pts; 'Kimi-K3 Technical Report' 391 pts")
for d in ["20260724","20260725","20260726","20260727","20260728","20260729"]:
    mark = "  <-- Kimi-K3 release day" if d=="20260727" else ""
    print(f"    {d}  HF views={hf.get(d,0):6d}{mark}")
seq=[hf.get(d,0) for d in ["20260725","20260726","20260727","20260728"]]
print(f"  monotone decay broken? 26->27 change = {seq[2]-seq[1]:+d} views ({(seq[2]/seq[1]-1)*100:+.0f}%)")

print(); print("="*72); print("ATTACK 5 — breakpoint search inflates effective parameters"); print("="*72)
from warningshot.data import modalities as mo
from warningshot.core import regression as rg
ch=mo.lookups("Hugging_Face",ev.JULY_INCIDENT.t0,horizon=25)
t,y=ch["t"],ch["y"]; k=y>0; t,y=t[k],y[k]
n=t.size
cand=range(5,n-5+1)
print(f"  n={n}, candidate breakpoints searched={len(list(cand))}")
print(f"  AIC penalty applied for two-phase: k=5 -> 2k=10")
print(f"  but breakpoint was CHOSEN by minimising SSE over {len(list(cand))} options,")
print(f"  so effective d.o.f. > 5; a selection-corrected AIC would need ~+2*log({len(list(cand))})={2*np.log(len(list(cand))):.1f} extra penalty")
tp=rg.fit_two_phase(t,y,min_seg=5); ex=rg.fit_exponential(t,y)
print(f"  two-phase AIC={tp['aic']:.2f}  exponential AIC={ex['aic']:.2f}  gap={ex['aic']-tp['aic']:.2f}")
print(f"  gap survives selection penalty? {'YES' if (ex['aic']-tp['aic'])>2*np.log(len(list(cand))) else 'NO -- two-phase win may be a selection artifact'}")
