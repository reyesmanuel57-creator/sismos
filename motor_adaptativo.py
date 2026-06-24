"""
MOTOR AUTO-ADAPTATIVO v2 — Chile
==================================
Mejoras sobre v1:
  1. ZONAS FINAS: 9 zonas de ~330km (antes 5 grandes) -> mejor "dónde"
  2. CALIBRACIÓN CONTINUA: guarda cada pronóstico y, en la corrida
     siguiente, compara con lo que realmente pasó. Acumula una "tasa de
     acierto" real para que sepas cuánto confiar.
  3. RANKING: las zonas se ordenan por riesgo (mayor primero).
  4. Mantiene: USGS histórico + CSN reciente, auto-reajuste ETAS.
"""
import numpy as np
import pandas as pd
import requests, json, os
from io import StringIO
from datetime import datetime, timezone

MC = 4.5
BBOX = (-56, -17, -76, -66)
VENTANA_APRENDIZAJE_DIAS = 2555
VENTANA_MEMORIA_DIAS = 1095

# ZONAS FINAS: franjas de 3° de latitud con nombre reconocible
ZONAS = {
    "Arica–Parinacota": (-20, -17),
    "Tarapacá (Iquique)": (-23, -20),
    "Antofagasta": (-26, -23),
    "Atacama (Copiapó)": (-29, -26),
    "Coquimbo (La Serena)": (-32, -29),
    "Valparaíso–Metropolitana": (-35, -32),
    "Maule–Ñuble": (-38, -35),
    "Biobío–Araucanía": (-41, -38),
    "Los Lagos–Aysén": (-44, -41),
}


def escanear_sismologia_cl():
    """
    Lee sismos del catálogo oficial del CSN (sismologia.cl) del día actual
    y el anterior. Trae microsismicidad chilena con todo el detalle, directo
    de la fuente oficial nacional.
    """
    import re
    h = {"User-Agent": "Mozilla/5.0"}
    patron = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+)\s*km\s+(\d+\.\d+)')
    filas = []
    hoy = datetime.now(timezone.utc)
    for delta in (0, 1):  # hoy y ayer
        f = hoy - pd.Timedelta(days=delta)
        url = (f"https://www.sismologia.cl/sismicidad/catalogo/"
               f"{f.year}/{f.month:02d}/{f.year}{f.month:02d}{f.day:02d}.html")
        try:
            r = requests.get(url, headers=h, timeout=25)
            if r.status_code != 200:
                continue
            txt = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", r.text))
            for m in patron.finditer(txt):
                try:
                    filas.append({"time": pd.to_datetime(m.group(1), utc=True),
                                  "latitude": float(m.group(2)), "longitude": float(m.group(3)),
                                  "depth": float(m.group(4)), "mag": float(m.group(5)),
                                  "place": "sismologia.cl"})
                except (ValueError, KeyError):
                    continue
        except Exception:
            continue
    return pd.DataFrame(filas)


def escanear_csn():
    try:
        r = requests.get("https://api.boostr.cl/earthquakes/recent.json", timeout=20)
        if r.status_code != 200:
            return pd.DataFrame()
        filas = []
        for e in r.json().get("data", []):
            try:
                t = pd.to_datetime(f"{e['date']} {e['hour']}").tz_localize(
                    "America/Santiago").tz_convert("UTC")
                filas.append({"time": t, "latitude": float(e["latitude"]),
                              "longitude": float(e["longitude"]),
                              "depth": float(str(e["depth"]).replace(" km","").strip()),
                              "mag": float(e["magnitude"]), "place": e.get("place","CSN")})
            except (ValueError, KeyError):
                continue
        return pd.DataFrame(filas)
    except Exception:
        return pd.DataFrame()


def escanear_chile(dias_atras=2600):
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    hoy = datetime.now(timezone.utc)
    inicio = (hoy - pd.Timedelta(days=dias_atras)).strftime("%Y-%m-%d")
    params = {"format": "csv", "starttime": inicio, "endtime": hoy.strftime("%Y-%m-%d"),
              "minlatitude": BBOX[0], "maxlatitude": BBOX[1],
              "minlongitude": BBOX[2], "maxlongitude": BBOX[3],
              "minmagnitude": MC, "orderby": "time-asc"}
    r = requests.get(url, params=params, timeout=120); r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    df = df[["time","latitude","longitude","depth","mag","place"]]
    ult = df["time"].max()
    # fuentes chilenas recientes: CSN (boostr) + sismologia.cl oficial
    for fuente in (escanear_csn(), escanear_sismologia_cl()):
        if len(fuente) > 0:
            f = fuente[(fuente["latitude"]>=BBOX[0])&(fuente["latitude"]<=BBOX[1])&
                       (fuente["longitude"]>=BBOX[2])&(fuente["longitude"]<=BBOX[3])&
                       (fuente["mag"]>=MC)]
            nuevos = f[f["time"] > ult - pd.Timedelta(minutes=2)]
            if len(nuevos) > 0:
                df = pd.concat([df, nuevos], ignore_index=True)
    # quitar duplicados (mismo sismo reportado por varias fuentes): redondear a minuto
    df["_t"] = df["time"].dt.floor("min")
    df = df.drop_duplicates(subset=["_t","mag"]).drop(columns="_t")
    return df.sort_values("time").reset_index(drop=True)


def reaprender_parametros(cat):
    from scipy.optimize import minimize
    hoy = cat["time"].max()
    rec = cat[cat["time"] >= hoy - pd.Timedelta(days=VENTANA_APRENDIZAJE_DIAS)]
    t0 = rec["time"].min()
    t_days = (rec["time"]-t0).dt.total_seconds().values/86400.0
    mags = rec["mag"].values.astype(float); T = t_days[-1]
    b = np.log10(np.e)/(mags.mean()-(MC-0.05))
    def nll(p):
        mu,K,al,c,pp = p
        if mu<=0 or K<=0 or c<=0 or pp<=1.001 or al<0: return 1e10
        s=0.0
        for i in range(len(t_days)):
            lo=np.searchsorted(t_days,t_days[i]-VENTANA_MEMORIA_DIAS,side="left")
            if lo<i:
                dt=t_days[i]-t_days[lo:i]
                lam=mu+(K*np.exp(al*(mags[lo:i]-MC))/(dt+c)**pp).sum()
            else: lam=mu
            s+=np.log(max(lam,1e-12))
        integ=K*np.exp(al*(mags-MC))*(c**(1-pp)-(T-t_days+c)**(1-pp))/(pp-1)
        return -(s-(mu*T+integ.sum()))
    res=minimize(nll,[0.3,0.02,1.8,0.12,1.3],method="L-BFGS-B",
                 bounds=[(1e-4,5),(1e-5,2),(0.1,4),(1e-3,10),(1.01,3)],options={"maxiter":80})
    mu,K,al,c,pp=res.x
    return {"mu":float(mu),"K":float(K),"alpha":float(al),"c":float(c),"p":float(pp),
            "b":float(b),"n_eventos_aprendizaje":len(rec),"convergencia":bool(res.success)}


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
            return 1-np.exp(-(tasa*10**(-par["b"]*(mag-MC)))*dias)
        ult7=z[z["time"]>=hoy-pd.Timedelta(days=7)]
        ult30=z[z["time"]>=hoy-pd.Timedelta(days=30)]
        p6=prob(6.0)
        nivel="ELEVADO" if p6>=0.10 else "MODERADO" if (p6>=0.04 or prob(5.0)>=0.4) else "NORMAL"
        out.append({"zona":nombre,"lat0":la0,"lat1":la1,
                    "prob_M5_7d":round(prob(5.0),3),"prob_M6_7d":round(p6,3),
                    "nivel":nivel,"n_ult7d":int(len(ult7)),"n_ult30d":int(len(ult30)),
                    "mag_max_ult30d":round(float(ult30["mag"].max()),1) if len(ult30) else 0.0})
    # RANKING: ordenar por probabilidad de M5 (mayor primero)
    out.sort(key=lambda x:-x["prob_M5_7d"])
    return out, hoy


def evaluar_calibracion(estado_previo, cat):
    """
    Compara los pronósticos de la corrida ANTERIOR con lo que realmente
    pasó después. Acumula aciertos para mostrar la confiabilidad real.
    Un 'acierto' = la zona que tenía mayor prob_M5 fue de las que tuvo
    actividad M>=5 en la semana siguiente.
    """
    calib = estado_previo.get("calibracion", {"evaluaciones":0,"aciertos_top3":0,
                                                "brier_sum":0.0,"n_brier":0})
    pron_prev = estado_previo.get("zonas")
    fecha_prev = estado_previo.get("ultimo_sismo")
    if not pron_prev or not fecha_prev:
        return calib
    try:
        t_prev = pd.to_datetime(fecha_prev, utc=True)
    except Exception:
        return calib
    # ventana evaluada: 7 días después del pronóstico previo
    fin = t_prev + pd.Timedelta(days=7)
    if cat["time"].max() < fin:
        return calib  # aún no pasó la semana completa, no evaluar todavía

    # ¿qué zonas tuvieron realmente M>=5 en esa semana?
    real = cat[(cat["time"]>t_prev)&(cat["time"]<=fin)&(cat["mag"]>=5.0)]
    zonas_con_evento = set()
    for _,e in real.iterrows():
        for z in pron_prev:
            if z["lat0"]<=e["latitude"]<z["lat1"]:
                zonas_con_evento.add(z["zona"])
    # top-3 zonas pronosticadas (ya vienen ordenadas por riesgo)
    top3 = [z["zona"] for z in pron_prev[:3]]
    acierto = len(zonas_con_evento & set(top3)) > 0 if zonas_con_evento else None

    # Brier score: qué tan bien calibradas las probabilidades de M5 por zona
    for z in pron_prev:
        ocurrio = 1 if z["zona"] in zonas_con_evento else 0
        calib["brier_sum"] += (z["prob_M5_7d"]-ocurrio)**2
        calib["n_brier"] += 1

    if acierto is not None:
        calib["evaluaciones"] += 1
        if acierto: calib["aciertos_top3"] += 1
    return calib


def actividad_reciente(cat, hoy):
    """
    Detecta sismos relevantes (M>=5) de las últimas 24-48h para mostrarlos
    destacados. NO es alerta temprana (el dato llega con minutos/horas de
    retraso) sino información rápida de lo que acaba de ocurrir + aviso de
    que pueden venir réplicas.
    """
    recientes = cat[(cat["time"] >= hoy - pd.Timedelta(hours=48)) & (cat["mag"] >= 5.0)]
    eventos = []
    for _, e in recientes.sort_values("time", ascending=False).iterrows():
        horas = (hoy - e["time"]).total_seconds() / 3600
        # ¿en qué zona cayó?
        zona = "Chile"
        for nombre, (la0, la1) in ZONAS.items():
            if la0 <= e["latitude"] < la1:
                zona = nombre; break
        eventos.append({
            "mag": round(float(e["mag"]), 1),
            "zona": zona,
            "lugar": str(e.get("place", "")),
            "hace_horas": round(horas, 1),
            "profundidad": round(float(e.get("depth", 0)), 0),
        })
    return eventos


def evaluar_vigilancia(zonas, eventos_recientes):
    """
    Define el estado global de vigilancia del país:
    - ALERTA: hubo M>=6 en últimas 48h (réplicas probables) o alguna zona ELEVADO
    - VIGILANCIA: hubo M>=5 reciente, o alguna zona MODERADO
    - NORMAL: todo tranquilo
    El mensaje promueve PREPARACIÓN, nunca afirma que vaya a ocurrir un sismo.
    """
    hay_m6 = any(e["mag"] >= 6.0 for e in eventos_recientes)
    hay_m5 = any(e["mag"] >= 5.0 for e in eventos_recientes)
    zona_elevada = any(z["nivel"] == "ELEVADO" for z in zonas)
    zona_moderada = any(z["nivel"] == "MODERADO" for z in zonas)

    if hay_m6 or zona_elevada:
        return {"estado": "ALERTA",
                "mensaje": ("Actividad sísmica elevada en Chile. Pueden ocurrir réplicas. "
                            "Buen momento para revisar tu kit de emergencia y plan familiar. "
                            "Esto NO predice un sismo: indica mayor actividad reciente.")}
    if hay_m5 or zona_moderada:
        return {"estado": "VIGILANCIA",
                "mensaje": ("Una o más zonas muestran actividad por sobre lo normal. "
                            "Mantente informado y ten lista tu preparación ante sismos.")}
    return {"estado": "NORMAL",
            "mensaje": ("Actividad sísmica dentro de lo normal para Chile. "
                        "Recuerda mantener siempre tu preparación ante sismos.")}


def correr(estado_previo_path="estado_aprendizaje.json"):
    cat = escanear_chile()
    par = reaprender_parametros(cat)
    zonas, hoy = estimar(cat, par)

    # cargar estado previo para historial y calibración
    previo = {}
    if os.path.exists(estado_previo_path):
        try: previo = json.load(open(estado_previo_path))
        except: pass

    calib = evaluar_calibracion(previo, cat)
    tasa_acierto = (calib["aciertos_top3"]/calib["evaluaciones"]*100
                    if calib.get("evaluaciones",0)>0 else None)
    brier = (calib["brier_sum"]/calib["n_brier"] if calib.get("n_brier",0)>0 else None)

    # NUEVO: actividad reciente + estado de vigilancia
    eventos_recientes = actividad_reciente(cat, hoy)
    vigilancia = evaluar_vigilancia(zonas, eventos_recientes)

    historial = previo.get("historial_parametros",[])
    historial.append({"fecha":datetime.now(timezone.utc).isoformat(),
                      "mu":round(par["mu"],4),"K":round(par["K"],4),
                      "alpha":round(par["alpha"],3),"p":round(par["p"],3),
                      "b":round(par["b"],3),"n_total":len(cat)})
    historial=historial[-200:]

    return {
        "actualizado":datetime.now(timezone.utc).isoformat(),
        "ultimo_sismo":hoy.isoformat(),
        "n_eventos_escaneados":len(cat),
        "fuente_datos":"USGS (histórico) + CSN y sismologia.cl (reciente)",
        "vigilancia":vigilancia,
        "actividad_reciente":eventos_recientes,
        "parametros_aprendidos":par,
        "zonas":zonas,
        "calibracion":{**calib,
                       "tasa_acierto_top3_pct":round(tasa_acierto,1) if tasa_acierto is not None else None,
                       "brier_score":round(brier,4) if brier is not None else None},
        "historial_parametros":historial,
        "descargo":("Sistema auto-adaptativo con calibración. Probabilidades por zona "
                    "para 7 días, NO predicción de día/hora/lugar. Oficial: SENAPRED, sismologia.cl."),
    }


if __name__ == "__main__":
    print("Escaneando, reaprendiendo y calibrando...")
    estado = correr()
    json.dump(estado, open("estado_aprendizaje.json","w"), ensure_ascii=False, indent=2)
    p=estado["parametros_aprendidos"]
    print(f"\nEscaneados {estado['n_eventos_escaneados']} sismos | b={p['b']:.2f}")
    print(f"\nRANKING de zonas por riesgo (7 días):")
    for i,z in enumerate(estado["zonas"],1):
        print(f"  {i}. [{z['nivel']:8s}] {z['zona']:26s} M5={z['prob_M5_7d']*100:3.0f}% M6={z['prob_M6_7d']*100:.0f}%")
    c=estado["calibracion"]
    print(f"\nCalibración: {c['evaluaciones']} semanas evaluadas, "
          f"tasa acierto top-3: {c.get('tasa_acierto_top3_pct','—')}%")
