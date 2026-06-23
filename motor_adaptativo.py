"""
MOTOR AUTO-ADAPTATIVO — Chile siempre activo
===============================================
En cada corrida (programada cada pocas horas en la nube):
  1. ESCANEA todos los sismos de Chile (USGS) hasta el instante actual.
  2. RE-AJUSTA los parámetros ETAS por máxima verosimilitud usando una
     ventana reciente (los últimos N años) — así APRENDE del comportamiento
     ACTUAL de Chile, no solo del histórico congelado.
  3. RE-ESTIMA el riesgo por zona para los próximos días con esos
     parámetros frescos + la actividad más reciente.
  4. GUARDA el estado y registra cómo cambiaron los parámetros (historial
     de aprendizaje) para poder ver la evolución.

"Auto-aprende" significa dos cosas que aquí SÍ ocurren:
  - la TASA de riesgo cambia con cada sismo nuevo (instantáneo, vía ETAS)
  - los PARÁMETROS del modelo (valor-b, productividad K, decaimiento p)
    se reajustan con los datos recientes (aprendizaje real del régimen
    actual de la zona)
"""
import numpy as np
import pandas as pd
import requests, json, os
from io import StringIO
from datetime import datetime, timezone
from scipy.optimize import minimize

MC = 4.5
BBOX = (-56, -17, -76, -66)
VENTANA_APRENDIZAJE_DIAS = 2555  # ~7 años: aprende del régimen reciente
VENTANA_MEMORIA_DIAS = 1095      # 3 años de activación para el riesgo

ZONAS = {
    "Norte Grande (Arica–Antofagasta)": (-26, -17),
    "Norte Chico (Atacama–Coquimbo)": (-32, -26),
    "Centro (Valparaíso–Maule)": (-37, -32),
    "Sur (Biobío–Los Lagos)": (-44, -37),
    "Austral (Aysén–Magallanes)": (-56, -44),
}


def escanear_chile(dias_atras=2600):
    """Escanea TODOS los sismos de Chile hasta ahora."""
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    hoy = datetime.now(timezone.utc)
    inicio = (hoy - pd.Timedelta(days=dias_atras)).strftime("%Y-%m-%d")
    params = {"format": "csv", "starttime": inicio,
              "endtime": hoy.strftime("%Y-%m-%d"),
              "minlatitude": BBOX[0], "maxlatitude": BBOX[1],
              "minlongitude": BBOX[2], "maxlongitude": BBOX[3],
              "minmagnitude": MC, "orderby": "time-asc"}
    r = requests.get(url, params=params, timeout=120); r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    return df.sort_values("time").reset_index(drop=True)


# ---- AUTO-APRENDIZAJE: reajuste ETAS por máxima verosimilitud ----
def neg_loglik(params, t_days, mags, T):
    mu, K, alpha, c, p = params
    if mu<=0 or K<=0 or c<=0 or p<=1.001 or alpha<0: return 1e10
    n=len(t_days); s=0.0
    for i in range(n):
        lo=np.searchsorted(t_days, t_days[i]-VENTANA_MEMORIA_DIAS, side="left")
        if lo<i:
            dt=t_days[i]-t_days[lo:i]
            lam=mu+(K*np.exp(alpha*(mags[lo:i]-MC))/(dt+c)**p).sum()
        else: lam=mu
        s+=np.log(max(lam,1e-12))
    integ=K*np.exp(alpha*(mags-MC))*(c**(1-p)-(T-t_days+c)**(1-p))/(p-1)
    return -(s - (mu*T + integ.sum()))


def reaprender_parametros(cat):
    """Reajusta ETAS con la ventana reciente. Aprende el régimen actual."""
    hoy = cat["time"].max()
    reciente = cat[cat["time"] >= hoy - pd.Timedelta(days=VENTANA_APRENDIZAJE_DIAS)]
    t0 = reciente["time"].min()
    t_days = (reciente["time"]-t0).dt.total_seconds().values/86400.0
    mags = reciente["mag"].values.astype(float)
    T = t_days[-1]
    # valor-b reciente (Aki MLE)
    b = np.log10(np.e)/(mags.mean()-(MC-0.05))
    x0=[0.3,0.02,1.8,0.12,1.3]
    bounds=[(1e-4,5),(1e-5,2),(0.1,4),(1e-3,10),(1.01,3)]
    res=minimize(neg_loglik,x0,args=(t_days,mags,T),method="L-BFGS-B",
                 bounds=bounds,options={"maxiter":80})
    mu,K,alpha,c,p=res.x
    return {"mu":float(mu),"K":float(K),"alpha":float(alpha),"c":float(c),
            "p":float(p),"b":float(b),"n_eventos_aprendizaje":len(reciente),
            "convergencia":bool(res.success)}


# ---- RE-ESTIMACIÓN de riesgo por zona ----
def tasa_zona(z, t_corte, mu_zona, par):
    h=z[(z["time"]<t_corte)&(z["time"]>=t_corte-pd.Timedelta(days=VENTANA_MEMORIA_DIAS))]
    if len(h)==0: return mu_zona
    dt=(t_corte-h["time"]).dt.total_seconds().values/86400.0
    return mu_zona+(par["K"]*np.exp(par["alpha"]*(h["mag"].values-MC))/(dt+par["c"])**par["p"]).sum()


def estimar(cat, par):
    hoy=cat["time"].max()
    años=(hoy-cat["time"].min()).days/365.25
    out=[]
    for nombre,(la0,la1) in ZONAS.items():
        z=cat[(cat["latitude"]>=la0)&(cat["latitude"]<la1)]
        mu_zona=len(z)/años/365.25 if años>0 else par["mu"]
        tasa=tasa_zona(z,hoy,mu_zona,par)
        def prob(mag,dias=7):
            lam=tasa*10**(-par["b"]*(mag-MC)); return 1-np.exp(-lam*dias)
        ult7=z[z["time"]>=hoy-pd.Timedelta(days=7)]
        ult30=z[z["time"]>=hoy-pd.Timedelta(days=30)]
        p6=prob(6.0)
        nivel="ELEVADO" if p6>=0.15 else "MODERADO" if (p6>=0.07 or prob(5.0)>=0.5) else "NORMAL"
        out.append({"zona":nombre,"prob_M5_7d":round(prob(5.0),3),
                    "prob_M6_7d":round(p6,3),"nivel":nivel,
                    "n_ult7d":int(len(ult7)),"n_ult30d":int(len(ult30)),
                    "mag_max_ult30d":round(float(ult30["mag"].max()),1) if len(ult30) else 0.0})
    return out, hoy


def correr(estado_previo_path="estado_aprendizaje.json"):
    cat = escanear_chile()
    par = reaprender_parametros(cat)
    zonas, hoy = estimar(cat, par)

    # historial de aprendizaje: cómo evolucionan los parámetros
    historial=[]
    if os.path.exists(estado_previo_path):
        try: historial=json.load(open(estado_previo_path)).get("historial_parametros",[])
        except: pass
    historial.append({"fecha":datetime.now(timezone.utc).isoformat(),
                      "mu":round(par["mu"],4),"K":round(par["K"],4),
                      "alpha":round(par["alpha"],3),"p":round(par["p"],3),
                      "b":round(par["b"],3),"n_total":len(cat)})
    historial=historial[-200:]  # conservar últimas 200 corridas

    return {
        "actualizado":datetime.now(timezone.utc).isoformat(),
        "ultimo_sismo":hoy.isoformat(),
        "n_eventos_escaneados":len(cat),
        "parametros_aprendidos":par,
        "zonas":zonas,
        "historial_parametros":historial,
        "descargo":("Sistema auto-adaptativo. Probabilidades por zona para 7 días, "
                    "NO predicción de día/hora/lugar. Oficial: SENAPRED, sismologia.cl."),
    }


if __name__ == "__main__":
    print("Escaneando sismos de Chile y reaprendiendo...")
    estado = correr()
    json.dump(estado, open("estado_aprendizaje.json","w"), ensure_ascii=False, indent=2)

    p=estado["parametros_aprendidos"]
    print(f"\nEscaneados: {estado['n_eventos_escaneados']} sismos | "
          f"último: {estado['ultimo_sismo'][:16]}")
    print(f"\nParámetros REAPRENDIDOS del Chile reciente:")
    print(f"  valor-b={p['b']:.3f}  K={p['K']:.4f}  alpha={p['alpha']:.2f}  "
          f"p={p['p']:.3f}  (n={p['n_eventos_aprendizaje']}, conv={p['convergencia']})")
    print(f"\nEstimación de riesgo actual por zona (7 días):")
    for z in estado["zonas"]:
        print(f"  [{z['nivel']:8s}] {z['zona'][:30]:30s} M≥5:{z['prob_M5_7d']*100:3.0f}% "
              f"M≥6:{z['prob_M6_7d']*100:3.0f}%")
