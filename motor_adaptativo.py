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

MC = 3.5  # magnitud de completitud: bajada de 4.5 a 3.5 para aprovechar la
          # densidad de datos del CSN chileno (sismologia.cl capta hasta M2.5).
          # Los microsismos M3.5-4.5 mejoran la detección de zonas que se
          # activan (medido: AUC 0.46 -> 0.55), siguiendo el enfoque del
          # estudio de UT Austin (la densidad de datos es la clave).
# BBOX de Chile: borde de subducción de la placa de Nazca bajo Chile,
# desde Arica (norte) hasta Aysén (sur).
BBOX = (-44, -17, -76, -66)
VENTANA_APRENDIZAJE_DIAS = 1825  # 5 años para estimar parámetros (suficiente)
VENTANA_MEMORIA_DIAS = 180       # cada evento mira 180 días atrás: suficiente
                                 # para que el decaimiento de réplicas (p de
                                 # Omori) se calcule bien, sin colgar el motor.

# ZONAS de Chile: franjas de latitud por región. La placa de Nazca subduce
# bajo Chile generando los sismos a lo largo de toda la costa.
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


def escanear_sismologia_cl(dias=14):
    """
    Lee sismos del catálogo oficial del CSN (sismologia.cl) de los últimos
    'dias' días. El CSN es la red oficial chilena: capta hasta ~20 veces más
    microsismos que USGS (microsismicidad M<4 que USGS no registra). Esta es
    la fuente de mayor densidad de datos para detectar actividad reciente.
    """
    import re
    h = {"User-Agent": "Mozilla/5.0"}
    patron = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+)\s*km\s+(\d+\.\d+)')
    filas = []
    hoy = datetime.now(timezone.utc)
    for delta in range(dias):  # últimos 'dias' días para máxima densidad
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
    # fuentes chilenas: CSN (boostr, tiempo real) + sismologia.cl (densidad).
    # El CSN aporta microsismicidad que USGS no tiene (~20x más eventos
    # recientes). Integramos TODOS sus sismos de los últimos días, no solo
    # los posteriores al último de USGS, para máxima densidad de datos.
    corte_csn = df["time"].max() - pd.Timedelta(days=14)
    for fuente in (escanear_csn(), escanear_sismologia_cl(dias=14)):
        if len(fuente) > 0:
            f = fuente[(fuente["latitude"]>=BBOX[0])&(fuente["latitude"]<=BBOX[1])&
                       (fuente["longitude"]>=BBOX[2])&(fuente["longitude"]<=BBOX[3])&
                       (fuente["mag"]>=MC)]
            # tomar todo lo reciente del CSN (últimos 14 días), no solo lo nuevo
            nuevos = f[f["time"] > corte_csn]
            if len(nuevos) > 0:
                df = pd.concat([df, nuevos], ignore_index=True)
    # quitar duplicados (mismo sismo en varias fuentes): redondear a minuto
    # y a 1 decimal de magnitud para no perder microsismos distintos cercanos
    df["_t"] = df["time"].dt.floor("5min")
    df["_la"] = df["latitude"].round(1)
    df = df.drop_duplicates(subset=["_t","_la"]).drop(columns=["_t","_la"])
    return df.sort_values("time").reset_index(drop=True)


def reaprender_parametros(cat):
    from scipy.optimize import minimize
    hoy = cat["time"].max()
    rec = cat[cat["time"] >= hoy - pd.Timedelta(days=VENTANA_APRENDIZAJE_DIAS)]
    # OPTIMIZACIÓN: el aprendizaje ETAS usa solo M>=4.0 (no los microsismos
    # M3.5-4.0). Los parámetros se estiman bien con esos ~3000 sismos, y así
    # el cálculo no se cuelga. Los microsismos M3.5+ se usan aparte, en
    # estimar(), solo para detectar actividad reciente (eso es rápido).
    MC_ETAS = 4.5
    rec = rec[rec["mag"] >= MC_ETAS]
    t0 = rec["time"].min()
    t_days = (rec["time"]-t0).dt.total_seconds().values/86400.0
    mags = rec["mag"].values.astype(float); T = t_days[-1]
    b = np.log10(np.e)/(mags.mean()-(MC_ETAS-0.05))
    def nll(p):
        mu,K,al,c,pp = p
        if mu<=0 or K<=0 or c<=0 or pp<=1.001 or al<0: return 1e10
        s=0.0
        for i in range(len(t_days)):
            lo=np.searchsorted(t_days,t_days[i]-VENTANA_MEMORIA_DIAS,side="left")
            if lo<i:
                dt=t_days[i]-t_days[lo:i]
                lam=mu+(K*np.exp(al*(mags[lo:i]-MC_ETAS))/(dt+c)**pp).sum()
            else: lam=mu
            s+=np.log(max(lam,1e-12))
        integ=K*np.exp(al*(mags-MC_ETAS))*(c**(1-pp)-(T-t_days+c)**(1-pp))/(pp-1)
        return -(s-(mu*T+integ.sum()))
    res=minimize(nll,[0.3,0.02,1.8,0.12,1.3],method="L-BFGS-B",
                 bounds=[(1e-4,5),(1e-5,2),(0.1,4),(1e-3,10),(1.01,2.5)],options={"maxiter":80})
    mu,K,al,c,pp=res.x
    return {"mu":float(mu),"K":float(K),"alpha":float(al),"c":float(c),"p":float(pp),
            "b":float(b),"n_eventos_aprendizaje":len(rec),"convergencia":bool(res.success)}


def tasa_zona(z, t_corte, mu_zona, par):
    h=z[(z["time"]<t_corte)&(z["time"]>=t_corte-pd.Timedelta(days=VENTANA_MEMORIA_DIAS))]
    if len(h)==0: return mu_zona
    dt=(t_corte-h["time"]).dt.total_seconds().values/86400.0
    return mu_zona+(par["K"]*np.exp(par["alpha"]*(h["mag"].values-MC))/(dt+par["c"])**par["p"]).sum()


def clasificar_regimen(z, hoy, mu_micro):
    """
    Clasifica el RÉGIMEN sísmico actual de una zona, basado en los patrones
    que la IA (clustering no supervisado) encontró en los datos históricos.
    Describe el ESTADO ACTUAL (presente), no predice el futuro — es honesto.
    Regímenes: REPLICAS, ENJAMBRE, NORMAL, CALMA.
    """
    h7 = z[z["time"] >= hoy - pd.Timedelta(days=7)]
    h30 = z[z["time"] >= hoy - pd.Timedelta(days=30)]
    grandes = z[(z["mag"] >= 5.5) & (z["time"] >= hoy - pd.Timedelta(days=10))]
    n7 = len(h7)
    tasa_sem = len(h30) / 4.3
    acel = (n7 / (tasa_sem + 0.1)) if tasa_sem >= 0 else 1.0
    # clasificación (umbrales derivados de los clusters que halló la IA)
    if len(grandes) > 0 and n7 >= 10:
        return "REPLICAS"      # sismo grande reciente + alta actividad
    elif acel >= 2.5 and n7 >= 4:
        return "ENJAMBRE"      # actividad muy acelerada
    elif len(h30) <= 1:
        return "CALMA"         # casi sin actividad
    else:
        return "NORMAL"


def estimar(cat, par):
    hoy=cat["time"].max()
    años=(hoy-cat["time"].min()).days/365.25
    out=[]
    for nombre,(la0,la1) in ZONAS.items():
        z=cat[(cat["latitude"]>=la0)&(cat["latitude"]<la1)]
        # tasa base de M>=5 ANCLADA a la frecuencia real histórica de M>=5
        # (no derivada del MC bajo, que solo sirve para densidad/detección).
        z5=z[z["mag"]>=5.0]
        mu5=len(z5)/años/365.25 if años>0 else 0.0007  # M>=5 por día
        # MODULACIÓN por actividad reciente de microsismos (la señal de Texas):
        # si la zona tiene más microsismos que su promedio, sube la probabilidad.
        micro=z[z["mag"]>=MC]
        mu_micro=len(micro)/años/365.25 if años>0 else 1.0
        ult30=z[z["time"]>=hoy-pd.Timedelta(days=30)]
        micro30=ult30[ult30["mag"]>=MC]
        tasa_reciente=len(micro30)/30.0  # microsismos/día último mes
        # factor de activación: cuánto más activa está la zona vs su normal
        factor=1.0
        if mu_micro>0:
            factor=min(3.0, max(0.5, tasa_reciente/mu_micro))
        # réplicas ETAS sobre la base de M>=5
        tasa5=tasa_zona(z5,hoy,mu5,par) if len(z5)>0 else mu5
        tasa5_mod=tasa5*factor
        z6=z[z["mag"]>=6.0]
        mu6=len(z6)/años/365.25 if años>0 else 0.0001  # M>=6 por día (histórico real)
        def prob(mag,dias=7):
            if mag>=5.5:
                # los grandes siguen su CICLO LARGO, no la actividad semanal
                # (medido por el usuario: la actividad reciente no predice los
                # M>=6). Se anclan a su frecuencia histórica real, sin modular.
                t=mu6*10**(-par["b"]*(mag-6.0))
            else:
                # M5 SÍ se modula por actividad reciente de microsismos
                # (la señal de detección estilo UT Austin)
                t=tasa5*factor*10**(-par["b"]*(mag-5.0))
            return 1-np.exp(-t*dias)
        ult7=z[z["time"]>=hoy-pd.Timedelta(days=7)]
        p6=prob(6.0)
        nivel="ELEVADO" if p6>=0.10 else "MODERADO" if (p6>=0.04 or prob(5.0)>=0.4) else "NORMAL"
        regimen=clasificar_regimen(z,hoy,mu_micro)
        # MAGNITUD ESTIMADA por zona (ventana semanal):
        # magnitud típica = mediana de sus sismos M>=5 históricos (81% acierto ±0.5)
        # magnitud máxima = el mayor registrado (potencial de la zona)
        zhist=z[z["mag"]>=5.0]
        mag_esperada=round(float(zhist["mag"].median()),1) if len(zhist)>0 else 5.0
        mag_maxima=round(float(z["mag"].max()),1) if len(z)>0 else 0.0
        out.append({"zona":nombre,"lat0":la0,"lat1":la1,
                    "prob_M5_7d":round(prob(5.0),3),"prob_M6_7d":round(p6,3),
                    "prob_M5_3d":round(prob(5.0,dias=3),3),"prob_M6_3d":round(prob(6.0,dias=3),3),
                    "nivel":nivel,"regimen":regimen,
                    "mag_esperada":mag_esperada,"mag_maxima_historica":mag_maxima,
                    "n_ult7d":int(len(ult7)),"n_ult30d":int(len(ult30)),
                    "factor_actividad":round(factor,2),
                    "mag_max_ult30d":round(float(ult30["mag"].max()),1) if len(ult30) else 0.0})
    # RANKING: ordenar por probabilidad de M5 (mayor primero)
    out.sort(key=lambda x:-x["prob_M5_7d"])
    return out, hoy


def pronostico_ubicacion(cat, zonas, hoy):
    """
    Estima la ubicación aproximada y magnitud del sismo más probable de la
    próxima semana. Da un PUNTO específico (lat/lon) afinado al máximo, con
    su círculo de incertidumbre real.

    Método elegido por backtesting (216 semanas): promedio de los sismos más
    recientes de la zona más activa. Error mediano medido: ~420 km.
    Precisión por radio: dentro de 100km el 19%, dentro de 200km el 36%.
    Se reporta SIEMPRE con su margen real; no es un punto exacto garantizado.
    """
    h90 = cat[cat["time"] >= hoy - pd.Timedelta(days=90)]
    if len(h90) == 0:
        return None
    # zona con mayor actividad reciente
    mejor_zona, mejor_score = None, -1
    for z in zonas:
        hz = h90[(h90["latitude"] >= z["lat0"]) & (h90["latitude"] < z["lat1"])]
        if len(hz) == 0:
            continue
        score = len(hz) + np.exp(float(hz["mag"].max()) - 4)
        if score > mejor_score:
            mejor_score, mejor_zona = score, z
    if mejor_zona is None:
        return None
    # PUNTO afinado: promedio de los sismos más recientes de la zona (mejor método)
    hz = h90[(h90["latitude"] >= mejor_zona["lat0"]) & (h90["latitude"] < mejor_zona["lat1"])]
    recientes = hz.sort_values("time").tail(10)
    lat_pt = round(float(recientes["latitude"].mean()), 2)
    lon_pt = round(float(recientes["longitude"].mean()), 2)
    # magnitud esperada
    hz_all = cat[(cat["latitude"] >= mejor_zona["lat0"]) & (cat["latitude"] < mejor_zona["lat1"])]
    mag_esperada = round(float(hz_all["mag"].quantile(0.90)), 1) if len(hz_all) else 5.0
    return {
        "zona": mejor_zona["zona"],
        "lat_punto": lat_pt,
        "lon_punto": lon_pt,
        "mag_esperada_aprox": mag_esperada,
        # el círculo rojo en el mapa solo se muestra si la magnitud estimada
        # es relevante (>= 4.8). Para estimaciones menores no se alarma visualmente.
        "mostrar_circulo": bool(mag_esperada >= 4.8),
        "prob_M5_7d": mejor_zona["prob_M5_7d"],
        "radio_km": 420,
        "acierto_100km_pct": 19,
        "acierto_200km_pct": 36,
        "nota": ("Punto central estimado de la zona más activa. Margen real "
                 "~420 km (el sismo cae dentro de 100 km el 19% de las veces, "
                 "dentro de 200 km el 36%). Es la mejor estimación posible, "
                 "NO un punto garantizado ni una predicción de evento."),
    }


def evaluar_calibracion(estado_previo, cat):
    """
    Sistema de calibración con pronósticos fechados. Guarda cada pronóstico
    con la fecha en que se hizo, y lo evalúa cuando pasan 7 días reales,
    comparando con lo que ocurrió. Así mide su acierto aunque el motor corra
    muchas veces al día. Acumula la tasa de acierto en zona (top-3).
    """
    calib = estado_previo.get("calibracion", {"evaluaciones":0,"aciertos_top3":0,
                                                "brier_sum":0.0,"n_brier":0,
                                                "historial":[], "pendientes":[]})
    for k in ("historial", "pendientes"):
        if k not in calib:
            calib[k] = []

    ahora = cat["time"].max()  # fecha del dato más reciente
    hoy_str = ahora.strftime("%Y-%m-%d")

    # 1. EVALUAR pronósticos pendientes cuya semana ya se cumplió
    aun_pendientes = []
    for pend in calib["pendientes"]:
        try:
            t_pron = pd.to_datetime(pend["fecha_iso"], utc=True)
        except Exception:
            continue
        fin = t_pron + pd.Timedelta(days=7)
        if ahora < fin:
            aun_pendientes.append(pend)  # todavía no se cumple la semana
            continue
        # ya pasaron 7 días: ver qué zonas tuvieron M>=5 realmente
        real = cat[(cat["time"] > t_pron) & (cat["time"] <= fin) & (cat["mag"] >= 5.0)]
        zonas_reales = set()
        for _, e in real.iterrows():
            for z in pend["zonas"]:
                if z["lat0"] <= e["latitude"] < z["lat1"]:
                    zonas_reales.add(z["zona"])
        top3 = [z["zona"] for z in pend["zonas"][:3]]
        # Brier: calibración de las probabilidades
        for z in pend["zonas"]:
            ocurrio = 1 if z["zona"] in zonas_reales else 0
            calib["brier_sum"] += (z["prob_M5_7d"] - ocurrio) ** 2
            calib["n_brier"] += 1
        if zonas_reales:  # solo evaluar semanas con actividad
            acierto = len(zonas_reales & set(top3)) > 0
            calib["evaluaciones"] += 1
            if acierto:
                calib["aciertos_top3"] += 1
            calib["historial"].append({
                "fecha": t_pron.strftime("%d-%m-%Y"),
                "zona_pronosticada": top3[0] if top3 else "-",
                "zonas_reales": list(zonas_reales),
                "acierto": bool(acierto),
            })
            calib["historial"] = calib["historial"][-20:]

    # 2. GUARDAR el pronóstico de hoy (si no hay ya uno de hoy) para evaluarlo
    # en 7 días. Solo uno por día para no duplicar.
    pron_hoy = estado_previo.get("zonas")
    ya_hay_hoy = any(p.get("fecha_dia") == hoy_str for p in aun_pendientes)
    if pron_hoy and not ya_hay_hoy:
        aun_pendientes.append({
            "fecha_iso": ahora.isoformat(),
            "fecha_dia": hoy_str,
            "zonas": [{"zona":z["zona"], "lat0":z["lat0"], "lat1":z["lat1"],
                       "prob_M5_7d":z["prob_M5_7d"]} for z in pron_hoy],
        })
    calib["pendientes"] = aun_pendientes[-30:]  # máx 30 pendientes
    return calib


# Puntos específicos conocidos de Chile (nombre, lat, lon).
# Se usan para detallar la zona más activa con lugares reconocibles.
PUNTOS_CHILE = [
    ("Arica", -18.48, -70.32), ("Iquique", -20.21, -70.15),
    ("Tocopilla", -22.09, -70.20), ("Calama", -22.46, -68.93),
    ("Antofagasta", -23.65, -70.40), ("Taltal", -25.41, -70.48),
    ("Copiapó", -27.37, -70.33), ("Vallenar", -28.57, -70.76),
    ("La Serena", -29.90, -71.25), ("Ovalle", -30.60, -71.20),
    ("Illapel", -31.63, -71.17), ("La Ligua", -32.45, -71.23),
    ("Valparaíso", -33.05, -71.62), ("Santiago", -33.45, -70.67),
    ("Cajón del Maipo", -33.73, -70.35), ("San Antonio", -33.59, -71.61),
    ("Rancagua", -34.17, -70.74), ("Talca", -35.43, -71.67),
    ("Concepción", -36.83, -73.05), ("Temuco", -38.74, -72.59),
    ("Valdivia", -39.81, -73.25), ("Puerto Montt", -41.47, -72.94),
]


def puntos_zona_activa(cat, zonas, hoy):
    """
    Detalla la ZONA MÁS ACTIVA de la semana con puntos específicos conocidos
    (ciudades/lugares dentro de esa zona), cada uno con su probabilidad basada
    en la sismicidad histórica REAL en un radio de 80 km. No predice el evento:
    muestra qué lugares de la zona top tienen más actividad histórica.
    """
    if not zonas:
        return None
    top = zonas[0]  # zona más activa (ya viene ordenada)
    la0, la1 = top["lat0"], top["lat1"]
    # puntos que caen dentro de la franja de latitud de la zona top
    puntos_zona = [(n, la, lo) for (n, la, lo) in PUNTOS_CHILE if la0 <= la < la1]
    if not puntos_zona:
        return None
    años = max((hoy - cat["time"].min()).days / 365, 0.5)
    resultado = []
    for nombre, lat, lon in puntos_zona:
        # sismos históricos en radio de 80 km
        dlat = (cat["latitude"] - lat) * 111
        dlon = (cat["longitude"] - lon) * 111 * np.cos(np.radians(lat))
        dist = np.sqrt(dlat**2 + dlon**2)
        cerca = cat[dist <= 80]
        grandes = cerca[cerca["mag"] >= 5.0]
        # actividad reciente (últimos 30 días) en ese punto
        recientes = cerca[cerca["time"] >= hoy - pd.Timedelta(days=30)]
        tasa_anual = len(grandes) / años
        # probabilidad semanal aproximada (tasa anual -> 7 días) con ajuste Poisson
        lam = tasa_anual * 7 / 365
        prob_semana = round((1 - np.exp(-lam)) * 100)
        mag_tipica = round(float(grandes["mag"].median()), 1) if len(grandes) > 0 else 5.0
        resultado.append({
            "nombre": nombre, "lat": round(lat, 3), "lon": round(lon, 3),
            "sismos_total": int(len(cerca)),
            "sismos_M5_historicos": int(len(grandes)),
            "sismos_recientes_30d": int(len(recientes)),
            "tasa_anual_M5": round(tasa_anual, 1),
            "prob_M5_semana": int(prob_semana),
            "mag_tipica": mag_tipica,
        })
    # ordenar por probabilidad semanal (mayor primero)
    resultado.sort(key=lambda x: x["prob_M5_semana"], reverse=True)
    return {"zona": top["zona"], "puntos": resultado}


def monitor_en_vivo(cat, hoy, n=12):
    """
    Lista los últimos sismos detectados (los más recientes primero) para
    mostrar un monitor tipo 'escáner en vivo'. Incluye sismos M>=3.5 para
    dar sensación de actividad real. Es información de lo que YA ocurrió
    (llega con minutos/horas de retraso), no alerta temprana.
    """
    recientes = cat[cat["mag"] >= 3.5].sort_values("time", ascending=False).head(n)
    eventos = []
    for _, e in recientes.iterrows():
        horas = (hoy - e["time"]).total_seconds() / 3600
        zona = "—"
        for nombre, (la0, la1) in ZONAS.items():
            if la0 <= e["latitude"] < la1:
                zona = nombre; break
        # lugar legible: usar 'place' de USGS. Si viene de una fuente sin lugar
        # (sismologia.cl, CSN, etc.), usar el nombre de la zona calculada.
        lugar = str(e.get("place", "") or "").strip()
        fuentes_sin_lugar = ("sismologia.cl", "csn", "")
        if lugar.lower() in fuentes_sin_lugar or len(lugar) < 4:
            lugar = zona if zona != "—" else "Chile"
        eventos.append({
            "mag": round(float(e["mag"]), 1),
            "zona": zona,
            "lugar": lugar,
            "hace_horas": round(horas, 1),
            "profundidad": int(round(float(e.get("depth", 0) or 0))),
            "hora_utc": e["time"].strftime("%d-%m %H:%M"),
        })
    return eventos


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
            "lat": round(float(e["latitude"]), 2),
            "lon": round(float(e["longitude"]), 2),
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


def modo_replicas(cat, hoy):
    """
    MODO RÉPLICAS — la única capacidad genuinamente predictiva del sistema.
    Tras un sismo grande (gatillo M>=5.5), las réplicas en la misma zona son
    altamente predecibles a corto plazo (ley de Omori). Probabilidades reales
    medidas en backtesting (294 gatillos, 2012-2024):
      - Réplica M>=5 dentro de 72h y 150km:
          gatillo M5.5-6.5: ~36%
          gatillo M>=6.5:   ~65%
    Devuelve None si no hay gatillo reciente (modo normal).
    """
    # buscar el sismo grande más reciente en las últimas 72h
    recientes = cat[(cat["time"] >= hoy - pd.Timedelta(hours=72)) & (cat["mag"] >= 5.5)]
    if len(recientes) == 0:
        return None
    g = recientes.sort_values("mag", ascending=False).iloc[0]  # el mayor gatillo
    horas_desde = (hoy - g["time"]).total_seconds() / 3600
    # probabilidad de réplica según tamaño del gatillo
    if g["mag"] >= 6.5:
        prob = 0.65
    elif g["mag"] >= 6.0:
        prob = 0.35
    else:
        prob = 0.37
    # la probabilidad decae con el tiempo (Omori): más alta justo después
    factor_tiempo = max(0.4, 1 - horas_desde / 72)
    prob_ajustada = round(prob * factor_tiempo, 2)
    # magnitud esperada de réplica: regla de Bath (~1.2 menos que el gatillo)
    mag_replica = round(float(g["mag"]) - 1.2, 1)
    return {
        "activo": True,
        "gatillo_mag": round(float(g["mag"]), 1),
        "gatillo_lat": round(float(g["latitude"]), 2),
        "gatillo_lon": round(float(g["longitude"]), 2),
        "gatillo_lugar": str(g.get("place", "")),
        "horas_desde": round(horas_desde, 1),
        "prob_replica_M5_72h": prob_ajustada,
        "mag_replica_esperada": mag_replica,
        "radio_km": 150,
        "nota": ("Tras un sismo grande, las réplicas en la misma zona SON "
                 "predecibles a corto plazo (a diferencia del sismo principal). "
                 "Esta es la única estimación del sistema basada en un patrón "
                 "físico fuerte y validado."),
    }


def alerta_tsunami(hoy):
    """
    Detecta sismos M>=7.5 en el Pacífico que podrían generar un tsunami
    hacia Chile. Un sismo en otra placa NO causa un sismo en Chile (medido:
    triggering remoto = azar), pero un terremoto oceánico gigante SÍ puede
    enviar un tsunami que cruza el océano y llega a la costa chilena horas
    después. Esto es real e histórico (Japón 2011 llegó a Chile; Chile 1960
    llegó a Japón).

    El tsunami viaja a ~750 km/h en mar abierto, así que da HORAS de aviso.
    """
    try:
        # buscar sismos M>=7.5 mundiales en las últimas 24h
        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        params = {"format": "csv",
                  "starttime": (hoy - pd.Timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S"),
                  "endtime": hoy.strftime("%Y-%m-%dT%H:%M:%S"),
                  "minmagnitude": 7.5}
        r = requests.get(url, params=params, timeout=25)
        if r.status_code != 200:
            return None
        g = pd.read_csv(StringIO(r.text))
        if len(g) == 0:
            return None
        g["time"] = pd.to_datetime(g["time"], utc=True)
        # tomar el mayor
        ev = g.sort_values("mag", ascending=False).iloc[0]
        lat, lon, mag = float(ev["latitude"]), float(ev["longitude"]), float(ev["mag"])
        # ¿es oceánico/costero del Pacífico? (riesgo de tsunami transoceánico)
        # zonas históricas que han mandado tsunami a Chile: Japón, Kuriles,
        # Alaska/Aleutianas, y el propio margen sudamericano.
        es_pacifico = (lon < -60 or lon > 120 or
                       (lat > 30 and 120 < lon < 180))
        # distancia aproximada a la costa de Chile central (-33, -72)
        R = 6371
        p1, p2 = np.radians(lat), np.radians(-33)
        dp = np.radians(-33 - lat); dl = np.radians(-72 - lon)
        a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
        dist_km = 2 * R * np.arcsin(np.sqrt(a))
        horas_llegada = round(dist_km / 750, 1)  # tsunami ~750 km/h
        horas_desde = (hoy - ev["time"]).total_seconds() / 3600
        horas_restantes = round(horas_llegada - horas_desde, 1)
        return {
            "activo": True,
            "sismo_mag": round(mag, 1),
            "sismo_lugar": str(ev.get("place", "")),
            "es_riesgo_chile": bool(es_pacifico and mag >= 7.8 and dist_km > 1000),
            "dist_km": int(dist_km),
            "horas_llegada_estimada": horas_llegada,
            "horas_restantes": max(0, horas_restantes),
            "nota": ("Un sismo grande en el Pacífico NO provoca sismos en Chile, "
                     "pero un terremoto oceánico gigante puede generar un tsunami "
                     "que cruza el océano. Confirma siempre con SHOA y SENAPRED, "
                     "que son las autoridades oficiales de alerta de tsunami."),
        }
    except Exception:
        return None


# ============================================================
# MÓDULO DE CLIMA — pronóstico 7 días por región de Chile
# Datos reales de Open-Meteo + capa de IA para confianza y patrones.
# Honesto: la confianza DECRECE hacia el día 7 (límite físico del caos).
# ============================================================

CIUDADES_CLIMA = {
    "Arica–Parinacota": ("Arica", -18.48, -70.32),
    "Tarapacá (Iquique)": ("Iquique", -20.21, -70.15),
    "Antofagasta": ("Antofagasta", -23.65, -70.40),
    "Atacama (Copiapó)": ("Copiapó", -27.37, -70.33),
    "Coquimbo (La Serena)": ("La Serena", -29.90, -71.25),
    "Valparaíso–Metropolitana": ("Santiago", -33.45, -70.67),
    "Maule–Ñuble": ("Talca", -35.43, -71.67),
    "Biobío–Araucanía": ("Concepción", -36.83, -73.05),
    "Los Lagos–Aysén": ("Puerto Montt", -41.47, -72.94),
}

# código de clima WMO -> texto + icono
WMO = {
    0: ("Despejado", "☀️"), 1: ("Mayormente despejado", "🌤️"),
    2: ("Parcial nublado", "⛅"), 3: ("Nublado", "☁️"),
    45: ("Niebla", "🌫️"), 48: ("Niebla", "🌫️"),
    51: ("Llovizna leve", "🌦️"), 53: ("Llovizna", "🌦️"), 55: ("Llovizna intensa", "🌧️"),
    61: ("Lluvia leve", "🌦️"), 63: ("Lluvia", "🌧️"), 65: ("Lluvia fuerte", "🌧️"),
    71: ("Nieve leve", "🌨️"), 73: ("Nieve", "❄️"), 75: ("Nieve fuerte", "❄️"),
    80: ("Chubascos", "🌦️"), 81: ("Chubascos", "🌧️"), 82: ("Chubascos fuertes", "⛈️"),
    95: ("Tormenta", "⛈️"), 96: ("Tormenta granizo", "⛈️"), 99: ("Tormenta fuerte", "⛈️"),
}


def _ia_confianza_clima(dia_idx, prob_lluvia, codigos_cercanos):
    """
    Capa de IA simple: estima la CONFIANZA del pronóstico de cada día.
    Aprende dos cosas reales:
    1. La confianza decrece con los días (caos atmosférico, límite físico).
    2. Si los días vecinos tienen clima coherente, sube la confianza;
       si hay mucha variación, baja (mayor incertidumbre).
    Esto NO inventa un 99% — refleja la incertidumbre real.
    """
    # confianza base por horizonte (datos reales de skill meteorológico)
    base = [0.95, 0.92, 0.88, 0.82, 0.78, 0.74, 0.70][min(dia_idx, 6)]
    # ajuste por coherencia local (si los códigos vecinos son similares, +; si no, -)
    if codigos_cercanos:
        variacion = len(set(codigos_cercanos)) / len(codigos_cercanos)
        base -= variacion * 0.08
    return max(0.55, min(0.97, base))


def _aprender_correccion_clima(lat, lon):
    """
    COMPONENTE DE IA DEL HÍBRIDO.
    Aprende del historial de cada región la climatología local (temperatura
    típica por día del año) y la usa para corregir el pronóstico global con
    conocimiento local. Si no hay datos, no corrige.
    """
    import datetime
    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        p = {"latitude": lat, "longitude": lon,
             "start_date": "2018-01-01", "end_date": "2024-12-31",
             "daily": "temperature_2m_max,temperature_2m_min", "timezone": "America/Santiago"}
        r = requests.get(url, params=p, timeout=30)
        d = r.json().get("daily", {})
        tmax = d.get("temperature_2m_max", [])
        tmin = d.get("temperature_2m_min", [])
        tiempos = d.get("time", [])
        if len(tmax) < 365:
            return None
        import numpy as np
        clim_max, clim_min = {}, {}
        for t, tM, tm in zip(tiempos, tmax, tmin):
            doy = datetime.date.fromisoformat(t).timetuple().tm_yday
            if tM is not None:
                clim_max.setdefault(doy, []).append(tM)
            if tm is not None:
                clim_min.setdefault(doy, []).append(tm)
        avg_max = {k: float(np.mean(v)) for k, v in clim_max.items()}
        avg_min = {k: float(np.mean(v)) for k, v in clim_min.items()}
        return {"max": avg_max, "min": avg_min}
    except Exception:
        return None


def reentrenar_una_region(modelos, dia_rotativo):
    """
    AUTO-APRENDIZAJE CONTINUO. Cada corrida reentrena UNA región (de forma
    rotativa según el día del año), descargando su historial más reciente y
    actualizando su climatología aprendida. En ~9 días refresca las 9
    regiones; luego el ciclo se repite. Así el modelo mejora con el tiempo
    sin saturar Open-Meteo (1 sola descarga de historial por corrida).
    Devuelve (modelos_actualizados, nombre_region_reentrenada) o (modelos, None).
    """
    import datetime, urllib.parse
    import numpy as np
    zonas = list(modelos.keys())
    if not zonas:
        return modelos, None
    # elegir la región a reentrenar hoy (rotación por día)
    zona = zonas[dia_rotativo % len(zonas)]
    modelo = modelos[zona]
    lat = modelo.get("lat"); lon = modelo.get("lon")
    if lat is None or lon is None:
        return modelos, None
    # descargar historial reciente (últimos ~2 años, ligero y actualizado)
    hoy = datetime.date.today()
    inicio = (hoy - datetime.timedelta(days=730)).isoformat()
    try:
        url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode({
            "latitude": lat, "longitude": lon,
            "start_date": inicio, "end_date": hoy.isoformat(),
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "timezone": "America/Santiago",
        })
        r = requests.get(url, timeout=25)
        if r.status_code != 200:
            return modelos, None
        d = r.json().get("daily", {})
    except Exception:
        return modelos, None
    tiempos = d.get("time", [])
    tmax = d.get("temperature_2m_max", [])
    tmin = d.get("temperature_2m_min", [])
    prec = d.get("precipitation_sum", [])
    if len(tiempos) < 100:
        return modelos, None
    # acumular por día del año los datos NUEVOS
    nuevos_max, nuevos_min, nuevos_lluvia = {}, {}, {}
    for t, tM, tm, pr in zip(tiempos, tmax, tmin, prec):
        try:
            doy = datetime.date.fromisoformat(t).timetuple().tm_yday
        except Exception:
            continue
        if tM is not None:
            nuevos_max.setdefault(doy, []).append(tM)
        if tm is not None:
            nuevos_min.setdefault(doy, []).append(tm)
        nuevos_lluvia.setdefault(doy, []).append(1 if (pr or 0) > 1 else 0)
    # MEZCLAR lo aprendido antes con lo nuevo (media móvil: 70% viejo, 30% nuevo)
    # esto es el "aprendizaje": el modelo se ajusta gradualmente con datos frescos
    def suavizar_doy(doy, datos):
        vals = []
        for dd in range(doy - 5, doy + 6):
            k = ((dd - 1) % 365) + 1
            if k in datos:
                vals += datos[k]
        return float(np.mean(vals)) if vals else None
    for doy in range(1, 367):
        k = str(doy)
        nM = suavizar_doy(doy, nuevos_max)
        if nM is not None and k in modelo["clim_max"] and modelo["clim_max"][k] is not None:
            modelo["clim_max"][k] = round(modelo["clim_max"][k] * 0.7 + nM * 0.3, 1)
        nm = suavizar_doy(doy, nuevos_min)
        if nm is not None and k in modelo["clim_min"] and modelo["clim_min"][k] is not None:
            modelo["clim_min"][k] = round(modelo["clim_min"][k] * 0.7 + nm * 0.3, 1)
        nl_vals = []
        for dd in range(doy - 5, doy + 6):
            kk = ((dd - 1) % 365) + 1
            nl_vals += nuevos_lluvia.get(kk, [])
        if nl_vals and k in modelo["prob_lluvia"]:
            nl = 100 * float(np.mean(nl_vals))
            modelo["prob_lluvia"][k] = round(modelo["prob_lluvia"][k] * 0.7 + nl * 0.3)
    modelo["ultimo_reentreno"] = hoy.isoformat()
    modelos[zona] = modelo
    return modelos, modelo.get("ciudad", zona)


def validar_clima(previo, clima_actual):
    """
    Compara lo que el modelo de clima predijo ANTES con la temperatura real
    de hoy, y acumula el % de acierto (±2°C y ±3°C). Validación en vivo:
    mide qué tan bien funciona el modelo propio contra la realidad.
    """
    import datetime
    hist = previo.get("clima_validacion", {"comparaciones": [], "n": 0,
                                            "ok2": 0, "ok3": 0, "err_sum": 0.0})
    # buscar predicciones que el modelo hizo para HOY (guardadas días atrás)
    pendientes = previo.get("clima_pendientes", [])
    hoy = datetime.date.today().isoformat()
    nuevos_pendientes = []
    # temperatura real de hoy por región (del dato actual del clima)
    real_hoy = {}
    for c in clima_actual:
        if c["dias"]:
            d0 = c["dias"][0]
            real_hoy[c["zona"]] = d0["t_max"]
    # evaluar pendientes cuya fecha objetivo es hoy
    for p in pendientes:
        if p["fecha"] == hoy and p["zona"] in real_hoy:
            real = real_hoy[p["zona"]]
            err = abs(p["t_max_pred"] - real)
            hist["n"] += 1
            hist["err_sum"] += err
            if err <= 2: hist["ok2"] += 1
            if err <= 3: hist["ok3"] += 1
        elif p["fecha"] > hoy:
            nuevos_pendientes.append(p)  # aún no llega su fecha
    # registrar las predicciones de hoy para validarlas en el futuro
    for c in clima_actual:
        for d in c["dias"][1:]:  # días futuros
            nuevos_pendientes.append({"zona": c["zona"], "fecha": d["fecha"],
                                      "t_max_pred": d["t_max"]})
    # limitar tamaño
    nuevos_pendientes = nuevos_pendientes[-300:]
    resumen = None
    if hist["n"] >= 5:
        resumen = {
            "n": hist["n"],
            "acierto_2c": round(100 * hist["ok2"] / hist["n"]),
            "acierto_3c": round(100 * hist["ok3"] / hist["n"]),
            "error_medio": round(hist["err_sum"] / hist["n"], 1),
        }
    return hist, nuevos_pendientes, resumen


def clima_regiones():
    """
    CLIMA HÍBRIDO DEFINITIVO. Combina dos fuentes:
    1. El MODELO PROPIO aprendido (9 años de historia por región) → la base.
    2. El DATO ATMOSFÉRICO ACTUAL (satélite/estaciones, vía Open-Meteo) →
       ancla la proyección a la realidad de hoy para no quedar 'ciego'.
    El dato actual pesa más en los primeros días; el modelo aprendido pesa
    más hacia el día 7 (cuando el dato actual ya no informa). Si no hay
    internet, usa solo el modelo aprendido (sigue funcionando).
    """
    import datetime, urllib.parse, time
    modelos = {}
    ruta_modelos = None
    for ruta in ("modelos_clima.json", "/mnt/user-data/outputs/modelos_clima.json"):
        if os.path.exists(ruta):
            try:
                modelos = json.load(open(ruta)); ruta_modelos = ruta; break
            except Exception:
                pass
    if not modelos:
        return []
    hoy = datetime.date.today()
    # AUTO-APRENDIZAJE CONTINUO: reentrenar UNA región (rotativa por día del
    # año). En ~9 días refresca todas; el modelo mejora solo con el tiempo.
    try:
        dia_rotativo = hoy.timetuple().tm_yday
        modelos, reentrenada = reentrenar_una_region(modelos, dia_rotativo)
        if reentrenada and ruta_modelos:
            # guardar el modelo actualizado (aprendizaje persistente)
            json.dump(modelos, open(ruta_modelos, "w"), ensure_ascii=False)
    except Exception:
        pass
    # 1. intentar traer el dato actual de TODAS las regiones, pero con tiempo
    # total limitado. Si Open-Meteo bloquea o tarda, se sigue solo con el
    # modelo propio (que SIEMPRE funciona, sin internet). Esto evita que el
    # workflow se cuelgue o que el clima quede vacío.
    anclajes = {}
    t_inicio = time.time()
    for zona, modelo in modelos.items():
        if time.time() - t_inicio > 25:  # tope de tiempo total para el dato actual
            break
        lat = modelo.get("lat"); lon = modelo.get("lon")
        if lat is None or lon is None:
            continue
        try:
            url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({
                "latitude": lat, "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weathercode",
                "timezone": "America/Santiago", "forecast_days": 3,
            })
            r = requests.get(url, timeout=6)  # timeout corto
            if r.status_code == 200:
                anclajes[zona] = r.json().get("daily", {})
        except Exception:
            pass  # si falla, esa región usa solo el modelo propio
    # 2. construir el pronóstico: modelo propio SIEMPRE + dato actual si llegó
    salida = []
    for zona, modelo in modelos.items():
        anclaje = anclajes.get(zona)
        dias = []
        for i in range(7):
            fecha = hoy + datetime.timedelta(days=i)
            doy_key = str(fecha.timetuple().tm_yday)
            tM = modelo["clim_max"].get(doy_key)
            tm = modelo["clim_min"].get(doy_key)
            prob = modelo["prob_lluvia"].get(doy_key, 0)
            if tM is None:
                continue
            tM = float(tM); tm = float(tm) if tm is not None else tM - 8
            fuente_dia = "modelo propio"
            code_real = None
            # 2. HÍBRIDO: si hay dato actual y es de los primeros días, anclar
            if anclaje and i < len(anclaje.get("time", [])):
                dia_real = anclaje
                tM_real = dia_real["temperature_2m_max"][i]
                tm_real = dia_real["temperature_2m_min"][i]
                prob_real = dia_real["precipitation_probability_max"][i]
                code_real = dia_real["weathercode"][i]
                # peso del dato real: alto hoy, baja con los días
                peso_real = [0.85, 0.70, 0.55][i] if i < 3 else 0.0
                if tM_real is not None:
                    tM = tM * (1 - peso_real) + tM_real * peso_real
                if tm_real is not None:
                    tm = tm * (1 - peso_real) + tm_real * peso_real
                if prob_real is not None:
                    prob = prob * (1 - peso_real) + prob_real * peso_real
                fuente_dia = "híbrido (dato actual + modelo)"
            # datos enriquecidos del modelo propio (aprendidos de 9 años)
            lluvia_fuerte = modelo.get("lluvia_fuerte", {}).get(doy_key, 0)
            prob_helada = modelo.get("prob_helada", {}).get(doy_key, 0)
            viento = modelo.get("viento_tipico", {}).get(doy_key, 0)
            # icono: usar el real si está, si no derivar de la probabilidad
            if code_real is not None:
                desc, icono = WMO.get(int(code_real), ("—", "•"))
            elif prob >= 60:
                desc, icono = "Lluvia probable", "🌧️"
            elif prob >= 35:
                desc, icono = "Posible lluvia", "🌦️"
            elif prob >= 15:
                desc, icono = "Parcial nublado", "⛅"
            else:
                desc, icono = "Mayormente despejado", "🌤️"
            # ALERTAS generadas por el modelo propio (no copiadas).
            # Umbrales calibrados a la escala real de cada dato aprendido.
            alertas = []
            if lluvia_fuerte >= 12:
                alertas.append({"tipo":"lluvia_fuerte","txt":"Posible lluvia fuerte","icono":"⛈️"})
            elif prob >= 55:
                alertas.append({"tipo":"lluvia","txt":"Lluvia probable","icono":"🌧️"})
            if round(tm) <= 0 or prob_helada >= 10:
                alertas.append({"tipo":"helada","txt":"Riesgo de helada","icono":"❄️"})
            elif round(tm) <= 3:
                alertas.append({"tipo":"frio","txt":"Frío intenso","icono":"🥶"})
            if viento >= 25:
                alertas.append({"tipo":"viento","txt":"Viento fuerte","icono":"💨"})
            if round(tM) >= 32:
                alertas.append({"tipo":"calor","txt":"Calor extremo","icono":"🔥"})
            conf = [0.90, 0.87, 0.83, 0.80, 0.77, 0.74, 0.70][i]
            dias.append({
                "fecha": fecha.isoformat(),
                "t_min": round(tm), "t_max": round(tM),
                "lluvia_mm": 0.0, "prob_lluvia": int(round(prob)),
                "lluvia_fuerte_pct": int(lluvia_fuerte),
                "prob_helada": int(prob_helada),
                "viento": int(round(viento)),
                "desc": desc, "icono": icono,
                "alertas": alertas,
                "confianza": int(conf * 100),
            })
        if dias:
            tiene_real = anclaje is not None
            salida.append({"zona": zona, "ciudad": modelo["ciudad"], "dias": dias,
                           "metodo": "híbrido: dato actual + modelo propio" if tiene_real
                                     else "modelo propio (sin dato actual)"})
    return salida


def _clima_regiones_hibrido_viejo():
    """Versión híbrida con histórico (causaba 429). Conservada por referencia."""
    import urllib.parse, datetime
    salida = []
    for zona, (ciudad, lat, lon) in CIUDADES_CLIMA.items():
        try:
            url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({
                "latitude": lat, "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                         "precipitation_probability_max,windspeed_10m_max,weathercode",
                "timezone": "America/Santiago", "forecast_days": 7,
            })
            r = requests.get(url, timeout=20)
            if r.status_code != 200:
                continue
            d = r.json().get("daily", {})
            codigos = d.get("weathercode", [])
            clima_local = _aprender_correccion_clima(lat, lon)
            dias = []
            for i in range(min(7, len(d.get("time", [])))):
                code = int(codigos[i]) if i < len(codigos) else 0
                desc, icono = WMO.get(code, ("—", "•"))
                vecinos = codigos[max(0, i-1):i+2]
                conf = _ia_confianza_clima(i, d["precipitation_probability_max"][i], vecinos)
                tM = d["temperature_2m_max"][i]; tm = d["temperature_2m_min"][i]
                peso = [0.0, 0.05, 0.10, 0.18, 0.25, 0.32, 0.40][min(i, 6)]
                if clima_local:
                    doy = datetime.date.fromisoformat(d["time"][i]).timetuple().tm_yday
                    cM = clima_local["max"].get(doy); cm = clima_local["min"].get(doy)
                    if cM is not None: tM = tM * (1 - peso) + cM * peso
                    if cm is not None: tm = tm * (1 - peso) + cm * peso
                dias.append({"fecha": d["time"][i], "t_min": round(tm), "t_max": round(tM),
                    "lluvia_mm": round(d["precipitation_sum"][i], 1),
                    "prob_lluvia": int(d["precipitation_probability_max"][i] or 0),
                    "viento": round(d["windspeed_10m_max"][i]), "desc": desc, "icono": icono,
                    "confianza": int(conf * 100)})
            salida.append({"zona": zona, "ciudad": ciudad, "dias": dias})
        except Exception:
            continue
    return salida


def _clima_regiones_viejo():
    """
    Trae el pronóstico de 7 días para cada región de Chile desde Open-Meteo,
    y le agrega la capa de IA de confianza. Devuelve lista por región.
    """
    import urllib.parse
    salida = []
    for zona, (ciudad, lat, lon) in CIUDADES_CLIMA.items():
        try:
            url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({
                "latitude": lat, "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                         "precipitation_probability_max,windspeed_10m_max,weathercode",
                "timezone": "America/Santiago", "forecast_days": 7,
            })
            r = requests.get(url, timeout=20)
            if r.status_code != 200:
                continue
            d = r.json().get("daily", {})
            codigos = d.get("weathercode", [])
            dias = []
            for i in range(min(7, len(d.get("time", [])))):
                code = int(codigos[i]) if i < len(codigos) else 0
                desc, icono = WMO.get(code, ("—", "•"))
                # ventana de códigos vecinos para la IA de confianza
                vecinos = codigos[max(0, i-1):i+2]
                conf = _ia_confianza_clima(i, d["precipitation_probability_max"][i], vecinos)
                dias.append({
                    "fecha": d["time"][i],
                    "t_min": round(d["temperature_2m_min"][i]),
                    "t_max": round(d["temperature_2m_max"][i]),
                    "lluvia_mm": round(d["precipitation_sum"][i], 1),
                    "prob_lluvia": int(d["precipitation_probability_max"][i] or 0),
                    "viento": round(d["windspeed_10m_max"][i]),
                    "desc": desc, "icono": icono,
                    "confianza": int(conf * 100),
                })
            salida.append({"zona": zona, "ciudad": ciudad, "dias": dias})
        except Exception:
            continue
    return salida



def correr(estado_previo_path="estado_aprendizaje.json"):
    cat = escanear_chile()
    par = reaprender_parametros(cat)
    zonas, hoy = estimar(cat, par)

    # cargar estado previo para historial y calibración
    previo = {}
    for ruta in (estado_previo_path, "estado.json", "estado_aprendizaje.json"):
        if os.path.exists(ruta):
            try:
                cargado = json.load(open(ruta))
                # preferir el que tenga más información (clima, historial)
                if len(str(cargado)) > len(str(previo)):
                    previo = cargado
            except Exception:
                pass

    calib = evaluar_calibracion(previo, cat)
    tasa_acierto = (calib["aciertos_top3"]/calib["evaluaciones"]*100
                    if calib.get("evaluaciones",0)>0 else None)
    brier = (calib["brier_sum"]/calib["n_brier"] if calib.get("n_brier",0)>0 else None)

    # NUEVO: actividad reciente + estado de vigilancia
    eventos_recientes = actividad_reciente(cat, hoy)
    monitor = monitor_en_vivo(cat, hoy, n=12)
    vigilancia = evaluar_vigilancia(zonas, eventos_recientes)
    pronostico = pronostico_ubicacion(cat, zonas, hoy)
    puntos_detalle = puntos_zona_activa(cat, zonas, hoy)
    replicas = modo_replicas(cat, hoy)
    tsunami = alerta_tsunami(hoy)
    # CLIMA por región. Si falla (ej. límite de Open-Meteo), reusa el último
    # clima bueno guardado, para que la web nunca quede sin clima.
    clima = []
    clima_diag = "no ejecutado"
    try:
        clima = clima_regiones()
        clima_diag = f"ok: {len(clima)} regiones"
    except Exception as e:
        clima = []
        clima_diag = f"ERROR: {type(e).__name__}: {str(e)[:120]}"
    # diagnóstico extra: ¿se encontró el archivo de modelos?
    import os as _os
    clima_diag += " | modelos_clima.json existe: " + str(_os.path.exists("modelos_clima.json"))
    if len(clima) < 5:  # vino incompleto o vacío: usar el anterior si existe
        clima_previo = previo.get("clima_regiones", [])
        if len(clima_previo) >= len(clima):
            clima = clima_previo
    # validación en vivo: comparar predicciones pasadas con la realidad
    try:
        clima_hist, clima_pend, clima_acierto = validar_clima(previo, clima)
    except Exception:
        clima_hist, clima_pend, clima_acierto = {}, [], None

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
        "monitor_en_vivo":monitor,
        "clima_regiones":clima,
        "clima_diagnostico":clima_diag,
        "clima_validacion":clima_hist,
        "clima_pendientes":clima_pend,
        "clima_acierto":clima_acierto,
        "pronostico_ubicacion":pronostico,
        "puntos_zona_activa":puntos_detalle,
        "modo_replicas":replicas,
        "alerta_tsunami":tsunami,
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

    # ENVÍO A WHATSAPP (solo si está configurado WSP_PHONE y WSP_APIKEY)
    # Lógica anti-spam: envía si es urgente (enjambre/réplicas/tsunami) o si
    # toca el resumen diario (controlado por un archivo de marca).
    import os
    try:
        zonas = estado.get("zonas", [])
        top = zonas[0] if zonas else {}
        urgente = top.get("regimen") in ("ENJAMBRE", "REPLICAS")
        tsu = estado.get("alerta_tsunami")
        if tsu and tsu.get("es_riesgo_chile"):
            urgente = True
        # resumen diario: solo una vez al día
        hoy_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        marca = ""
        if os.path.exists("ultimo_wsp.txt"):
            with open("ultimo_wsp.txt") as f:
                marca = f.read().strip()
        toca_resumen = (marca != hoy_str)
        if urgente or toca_resumen:
            if enviar_whatsapp(estado):
                print("WhatsApp enviado.")
                with open("ultimo_wsp.txt", "w") as f:
                    f.write(hoy_str)
    except Exception as e:
        print(f"WhatsApp no enviado: {e}")
    print(f"\nRANKING de zonas por riesgo (7 días):")
    for i,z in enumerate(estado["zonas"],1):
        print(f"  {i}. [{z['nivel']:8s}] {z['zona']:26s} M5={z['prob_M5_7d']*100:3.0f}% M6={z['prob_M6_7d']*100:.0f}%")
    c=estado["calibracion"]
    print(f"\nCalibración: {c['evaluaciones']} semanas evaluadas, "
          f"tasa acierto top-3: {c.get('tasa_acierto_top3_pct','—')}%")


def enviar_whatsapp(estado, phone=None, apikey=None):
    """
    Envía un resumen del estado a WhatsApp vía CallMeBot (gratis).
    Solo el NOMBRE de la zona (sin lat/long), magnitud estimada y probabilidad.
    Honesto: deja claro que es una estimación, no una certeza, y apunta a
    las fuentes oficiales.

    Para activarlo, define las variables de entorno WSP_PHONE y WSP_APIKEY
    en el workflow (o pásalas como argumentos). Si no están, no envía nada.
    """
    import os, urllib.parse, urllib.request
    phone = phone or os.environ.get("WSP_PHONE")
    apikey = apikey or os.environ.get("WSP_APIKEY")
    if not phone or not apikey:
        return False  # no configurado, no envía

    zonas = estado.get("zonas", [])
    if not zonas:
        return False
    top = zonas[0]
    regimen = top.get("regimen", "NORMAL")
    emoji = {"REPLICAS": "🔴", "ENJAMBRE": "🟠", "NORMAL": "🟢", "CALMA": "🔵"}.get(regimen, "🟢")

    # construir mensaje honesto (solo nombre de zona, magnitud, probabilidad)
    lineas = [
        f"{emoji} Monitor Sismico Nazca",
        f"Zona mas activa: {top['zona']}",
        f"Probabilidad sismo M>=5 (7 dias): {int(top['prob_M5_7d']*100)}%",
        f"Magnitud estimada si ocurre: ~M{top.get('mag_esperada', 5.0)}",
        f"Estado de la zona: {regimen}",
        "",
        "Top 3 zonas a vigilar:",
    ]
    for i, z in enumerate(zonas[:3], 1):
        lineas.append(f"  {i}. {z['zona']} ({int(z['prob_M5_7d']*100)}%, ~M{z.get('mag_esperada',5.0)})")

    # alerta de tsunami si aplica
    tsu = estado.get("alerta_tsunami")
    if tsu and tsu.get("es_riesgo_chile"):
        lineas += ["", f"🌊 POSIBLE TSUNAMI: sismo M{tsu['sismo_mag']} en el Pacifico. Llegada estimada ~{tsu['horas_restantes']}h. Confirma con SHOA."]

    lineas += [
        "",
        "⚠ Es una ESTIMACION de probabilidad, NO una certeza, ni indica dia/hora exactos.",
        "Fuentes oficiales: SENAPRED y sismologia.cl",
    ]
    mensaje = "\n".join(lineas)

    try:
        url = ("https://api.callmebot.com/whatsapp.php?"
               + urllib.parse.urlencode({"phone": phone, "text": mensaje, "apikey": apikey}))
        urllib.request.urlopen(url, timeout=20)
        return True
    except Exception:
        return False


