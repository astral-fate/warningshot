import sys, numpy as np
sys.path.insert(0,'src')
from warningshot.data import modalities as mo, events as ev
from warningshot.core import regression as rg
from scipy import optimize

def nls_exp(t,y):
    def f(p): return p[0]*np.exp(-t/p[1]) - y
    r = optimize.least_squares(f, x0=[max(y), 5.0], bounds=([0,0.01],[np.inf,1e4]))
    return r.x[1]*np.log(2)

rows=[]
for name, ch, unit, div in [
    ("lookups",    mo.lookups("Hugging_Face", ev.JULY_INCIDENT.t0, horizon=25), "d", 1.0),
    ("media",      mo.media('"Hugging Face" OpenAI', ev.JULY_INCIDENT.t0, horizon=25), "d", 1.0)]:
    t,y = ch["t"], ch["y"]; k=y>0; t,y=t[k],y[k]
    rows.append((name, rg.fit_exponential(t,y)["half_life"]/div, nls_exp(t,y)/div, unit))

ch = mo.engagement("48997548")
h = np.array(ch["hourly"], dtype=float)
t = np.arange(1.0, h.size+1.0)
k = h>0; th,hh = t[k], h[k]
rows.append(("engagement", rg.fit_exponential(th,hh)["half_life"]/24.0, nls_exp(th,hh)/24.0, "d"))

print(f"{'channel':12s}{'log-OLS (d)':>14}{'NLS raw (d)':>14}{'ratio':>8}")
for n,a,b,u in rows: print(f"{n:12s}{a:14.3f}{b:14.3f}{b/a:8.2f}")
ols=[r[1] for r in rows]; nls=[r[2] for r in rows]
print()
print(f"HEADLINE SPREAD  log-OLS: {max(ols)/min(ols):.1f}x   NLS(raw): {max(nls)/min(nls):.1f}x")
print(f"slowest/fastest  log-OLS: {rows[int(np.argmax(ols))][0]}/{rows[int(np.argmin(ols))][0]}"
      f"   NLS: {rows[int(np.argmax(nls))][0]}/{rows[int(np.argmin(nls))][0]}")
