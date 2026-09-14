import sys, numpy as np
sys.path.insert(0,'src')
from warningshot.data import modalities as mo, events as ev
from warningshot.core import regression as rg
from scipy import optimize, stats

print("="*72)
print("ATTACK 1 — residual autocorrelation (invalidates iid residual bootstrap)")
print("="*72)
for name, ch in [("lookups", mo.lookups("Hugging_Face", ev.JULY_INCIDENT.t0, horizon=25)),
                 ("media",   mo.media('"Hugging Face" OpenAI', ev.JULY_INCIDENT.t0, horizon=25))]:
    t,y = ch["t"], ch["y"]
    keep = y>0; t,y = t[keep], y[keep]
    f = rg.fit_exponential(t,y)
    resid = np.log(y) - (np.log(f["amplitude"]) - t/f["tau"])
    r1 = np.corrcoef(resid[:-1], resid[1:])[0,1]
    # Durbin-Watson
    dw = np.sum(np.diff(resid)**2)/np.sum(resid**2)
    # runs test for sign changes
    signs = np.sign(resid); runs = 1+np.sum(signs[1:]!=signs[:-1])
    exp_runs = 1 + 2*np.sum(signs>0)*np.sum(signs<0)/len(signs)
    print(f"{name:9s} n={len(t):3d}  lag-1 autocorr={r1:+.3f}  Durbin-Watson={dw:.3f} (2.0=indep)  runs={runs} vs expected {exp_runs:.1f}")

print()
print("="*72)
print("ATTACK 2 — log-OLS vs nonlinear least squares on the raw scale")
print("="*72)
def nls_exp(t,y):
    def f(p): return p[0]*np.exp(-t/p[1]) - y
    r = optimize.least_squares(f, x0=[y[0], 5.0], bounds=([0,0.01],[np.inf,1e4]))
    return r.x
for name, ch, unit in [("lookups", mo.lookups("Hugging_Face", ev.JULY_INCIDENT.t0, horizon=25), "d"),
                       ("media",   mo.media('"Hugging Face" OpenAI', ev.JULY_INCIDENT.t0, horizon=25), "d")]:
    t,y = ch["t"], ch["y"]; keep=y>0; t,y=t[keep],y[keep]
    f = rg.fit_exponential(t,y)
    A,tau = nls_exp(t,y)
    print(f"{name:9s} log-OLS t_half={f['half_life']:6.2f}{unit}   NLS(raw) t_half={tau*np.log(2):6.2f}{unit}"
          f"   ratio={(tau*np.log(2))/f['half_life']:.2f}x")
ch = mo.engagement("48997548"); h=np.array(ch["hourly"],float); t=np.arange(1,h.size+1,float)
keep=h>0; th,hh=t[keep],h[keep]
f=rg.fit_exponential(th,hh); A,tau=nls_exp(th,hh)
print(f"{'engagement':9s} log-OLS t_half={f['half_life']:6.2f}h   NLS(raw) t_half={tau*np.log(2):6.2f}h"
      f"   ratio={(tau*np.log(2))/f['half_life']:.2f}x")
