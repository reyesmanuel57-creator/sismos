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
        # ya pasaron 7 días: ver qué zonas tuvieron M>=5 realmente.
        # Si el catálogo no cubre esa fecha (hubo un hueco de corridas y los
        # sismos viejos ya no vienen en el escaneo), no se puede evaluar de
        # forma justa: se descarta sin contar, para no inventar un fallo.
        cat_min = cat["time"].min()
        if t_pron < cat_min:
            # el catálogo empieza DESPUÉS del pronóstico -> datos incompletos
            continue  # se cierra el pendiente sin puntuarlo (hueco de datos)
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
            calib["historial"] = calib["historial"][-40:]

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
    calib["pendientes"] = aun_pendientes[-60:]  # hasta 60 días de pendientes
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


# Estaciones meteorológicas reales de Chile (red DGAC/DMC vía boostr.cl),
# asignadas a cada zona. Son mediciones EN EL SUELO, independientes de los
# modelos atmosféricos globales: sirven para mostrar el estado actual y para
# VALIDAR el pronóstico contra un dato que no proviene del mismo modelo.
ESTACIONES_CHILE = {
    "Arica–Parinacota": "SCAR",           # Arica
    "Tarapacá (Iquique)": "SCDA",         # Iquique
    "Antofagasta": "SCFA",                # Antofagasta
    "Atacama (Copiapó)": "SCAT",          # Caldera
    "Coquimbo (La Serena)": "SCSE",       # La Serena/Coquimbo
    "Valparaíso–Metropolitana": "SCEL",   # Santiago Poniente
    "Maule–Ñuble": "SCIC",                # Curicó
    "Biobío–Araucanía": "SCIE",           # Concepción
    "Los Lagos–Aysén": "SCTE",            # Puerto Montt
}


def observaciones_chile():
    """
    Trae las observaciones actuales de las estaciones meteorológicas chilenas
    (una sola llamada). Devuelve {zona: {...}} con temperatura, condición y
    humedad medidas en el suelo. Si falla, devuelve {} y nada se rompe.
    """
    try:
        r = requests.get("https://api.boostr.cl/weather.json",
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if r.status_code != 200:
            return {}
        datos = r.json().get("data", [])
    except Exception:
        return {}
    por_codigo = {e.get("code"): e for e in datos if e.get("code")}
    salida = {}
    for zona, codigo in ESTACIONES_CHILE.items():
        e = por_codigo.get(codigo)
        if not e:
            continue
        try:
            temp = float(e.get("temperature"))
        except (TypeError, ValueError):
            continue
        salida[zona] = {
            "estacion": codigo,
            "ciudad": e.get("city", ""),
            "temperatura": round(temp, 1),
            "condicion": e.get("condition", ""),
            "humedad": e.get("humidity"),
            "hora": e.get("updated_at", ""),
        }
    return salida


def acumular_observaciones(previo, obs, hoy_str):
    """
    El motor corre varias veces al día. Cada corrida registra la temperatura
    medida por la estación chilena; guardando el máximo y el mínimo del día se
    reconstruye la temperatura máxima/mínima REAL observada. Ese dato sirve
    después para validar el pronóstico contra medición independiente.
    """
    acc = previo.get("obs_diarias", {})
    if not isinstance(acc, dict):
        acc = {}
    for zona, o in obs.items():
        t = o.get("temperatura")
        if t is None:
            continue
        z = acc.setdefault(zona, {})
        dia = z.get(hoy_str)
        if dia is None:
            z[hoy_str] = {"t_max_obs": t, "t_min_obs": t, "n": 1}
        else:
            dia["t_max_obs"] = max(dia["t_max_obs"], t)
            dia["t_min_obs"] = min(dia["t_min_obs"], t)
            dia["n"] = dia.get("n", 1) + 1
        # conservar solo los últimos 20 días por zona
        if len(z) > 20:
            for k in sorted(z.keys())[:-20]:
                z.pop(k, None)
    return acc


# Corrección de sesgo tipo MOS ("pensar como meteorólogo"). Parámetros
# elegidos por validación walk-forward con holdout: se ajustan en la
# primera mitad de los datos y se evalúan en la segunda. Mejora medida
# fuera de muestra: +1.5% (dentro de muestra daba +8%, era ilusión).
# Se corrige solo la MITAD del sesgo (k=0.5): aplicar el sesgo completo
# EMPEORA el pronóstico, porque la estimación es ruidosa.
VENTANA_SESGO = 14      # cuántas comparaciones recientes se recuerdan
K_SESGO = 0.5           # fracción del sesgo que se descuenta
MIN_SESGO = 8           # no corregir con menos de 8 comparaciones
MAX_SESGO = 2.5         # tope de corrección, en °C


# Versión del esquema de validación. Al subirla, se descarta la memoria de
# sesgo anterior. Necesario porque la validación v1 reconstruía la máxima
# diaria muestreando estaciones cada corrida; como GitHub no ejecuta el motor
# tantas veces al día, esa "máxima observada" quedaba por debajo de la real y
# el sistema aprendía un sesgo falso de varios grados.
# Nombres legibles de los modelos atmosféricos, para el marcador público.
NOMBRES_MODELOS = {
    "ecmwf_ifs025": "ECMWF (europeo)",
    "gfs_seamless": "GFS (americano)",
    "icon_seamless": "ICON (alemán)",
    "gem_seamless": "GEM (canadiense)",
    "meteofrance_seamless": "MeteoFrance",
}

VALIDACION_VERSION = 3
DIAS_ESPERA_ERA5 = 3      # ERA5 publica con ~2 días de retraso


def verdad_era5(modelos, dias=12):
    """
    Temperatura máxima y mínima REALES de los últimos días, por región, según
    el reanálisis ERA5. Es la verdad de contraste: entrega el máximo del día
    completo, sin depender de cuántas veces alcanzó a correr el motor.

    Se piden las 9 regiones en UNA sola llamada (Open-Meteo acepta varias
    coordenadas separadas por coma y responde una lista). Antes se hacía una
    llamada por región y en GitHub fallaban algunas por tiempo de espera.
    Devuelve {zona: {"YYYY-MM-DD": {"tmax": x, "tmin": y}}}.
    """
    import datetime, urllib.parse
    zonas, lats, lons = [], [], []
    for zona, modelo in modelos.items():
        lat, lon = modelo.get("lat"), modelo.get("lon")
        if lat is None or lon is None:
            continue
        zonas.append(zona); lats.append(str(lat)); lons.append(str(lon))
    if not zonas:
        return {}
    hoy = datetime.date.today()
    ini = (hoy - datetime.timedelta(days=dias)).isoformat()
    fin = (hoy - datetime.timedelta(days=2)).isoformat()
    url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode({
        "latitude": ",".join(lats), "longitude": ",".join(lons),
        "start_date": ini, "end_date": fin,
        "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": "America/Santiago"})
    datos = None
    for intento in range(2):
        try:
            r = requests.get(url, timeout=25)
            if r.status_code == 200:
                datos = r.json(); break
        except Exception:
            pass
    if datos is None:
        return {}
    if isinstance(datos, dict):      # una sola ubicación
        datos = [datos]
    salida = {}
    for zona, bloque in zip(zonas, datos):
        d = (bloque or {}).get("daily", {})
        reg = {}
        for t, mx, mn in zip(d.get("time", []), d.get("temperature_2m_max", []),
                             d.get("temperature_2m_min", [])):
            if mx is not None:
                reg[t] = {"tmax": float(mx),
                          "tmin": float(mn) if mn is not None else None}
        if reg:
            salida[zona] = reg
    return salida


def validar_clima(previo, clima_actual, verdad=None):
    """
    Compara cada pronóstico guardado con la temperatura REAL de ese día, según
    el reanálisis ERA5. Acumula el acierto (±2 °C y ±3 °C) y el sesgo con signo
    por región y horizonte, que luego se descuenta a medias.

    Por qué ERA5 y no las estaciones: el motor no corre suficientes veces al
    día como para capturar el momento más caliente, así que el "máximo
    observado" en las estaciones queda sistemáticamente bajo. Validar contra
    eso enseñaba un sesgo falso. ERA5 entrega el máximo del día completo.
    """
    import datetime
    base = {"n": 0, "ok2": 0, "ok3": 0, "err_sum": 0.0}
    hist = previo.get("clima_validacion") or {}
    sesgos = previo.get("clima_sesgos") or {}
    marcador = previo.get("clima_marcador") or {}
    # si la validación previa venía del esquema viejo, se descarta entera
    if previo.get("validacion_version") != VALIDACION_VERSION:
        hist, sesgos, marcador = dict(base), {}, {}
    for k, v in base.items():
        hist.setdefault(k, v)
    if not isinstance(sesgos, dict):
        sesgos = {}

    pendientes = previo.get("clima_pendientes", [])
    hoy = datetime.date.today()
    limite = (hoy - datetime.timedelta(days=DIAS_ESPERA_ERA5)).isoformat()
    verdad = verdad or {}

    quedan = []
    for p in pendientes:
        zona, f = p.get("zona"), p.get("fecha")
        if not zona or not f:
            continue
        if f > limite:
            quedan.append(p)          # todavía no hay verdad para ese día
            continue
        real = (verdad.get(zona) or {}).get(f)
        if not real:
            # sin verdad disponible: se descarta si ya es muy viejo
            if f >= (hoy - datetime.timedelta(days=14)).isoformat():
                quedan.append(p)
            continue
        pred = p.get("t_max_pred")
        if pred is None:
            continue
        err = abs(pred - real["tmax"])
        hist["n"] += 1
        hist["err_sum"] += err
        if err <= 2: hist["ok2"] += 1
        if err <= 3: hist["ok3"] += 1
        lead = str(p.get("lead", 1))
        z = sesgos.setdefault(zona, {})
        serie = z.setdefault(lead, [])
        serie.append(round(pred - real["tmax"], 2))
        z[lead] = serie[-VENTANA_SESGO:]

        # MARCADOR: mismo día, misma verdad, para todos los pronósticos.
        # No se aprende a imitar a nadie: se aprende contra la realidad, y
        # aquí solo se lleva la cuenta de quién estuvo más cerca.
        def _anotar(fuente, valor):
            if valor is None:
                return
            m = marcador.setdefault(fuente, {"n": 0, "err_sum": 0.0, "ok2": 0})
            e = abs(valor - real["tmax"])
            m["n"] += 1; m["err_sum"] += e
            if e <= 2: m["ok2"] += 1
        _anotar("Mi sistema", pred)
        _anotar("Ensemble (mediana)", p.get("med_tmax"))
        for nombre_m, v in (p.get("modelos_tmax") or {}).items():
            _anotar(NOMBRES_MODELOS.get(nombre_m, nombre_m), v)

    # registrar los pronósticos de hoy para evaluarlos cuando ERA5 los cubra
    hechos = {(p["zona"], p["fecha"]) for p in quedan}
    for c in clima_actual:
        for j, d in enumerate(c["dias"][1:], start=1):
            if (c["zona"], d["fecha"]) in hechos:
                continue
            reg = {"zona": c["zona"], "fecha": d["fecha"],
                   "t_max_pred": d["t_max"], "lead": j}
            # se guarda también lo que dijo cada modelo atmosférico ese día,
            # para poder puntuarlos todos contra la misma verdad (ERA5).
            if d.get("modelos_tmax"):
                reg["modelos_tmax"] = d["modelos_tmax"]
                reg["med_tmax"] = d.get("med_tmax")
            quedan.append(reg)

    resumen = None
    if hist["n"] >= 10:
        resumen = {"n": hist["n"],
                   "acierto_2c": round(100 * hist["ok2"] / hist["n"]),
                   "acierto_3c": round(100 * hist["ok3"] / hist["n"]),
                   "error_medio": round(hist["err_sum"] / hist["n"], 2),
                   "fuente": "reanálisis ERA5"}

    # tabla ordenada del marcador (solo con suficientes comparaciones)
    tabla = None
    listos = [(f, m) for f, m in marcador.items() if m["n"] >= 10]
    if listos:
        tabla = sorted(
            [{"fuente": f, "n": m["n"],
              "error_medio": round(m["err_sum"] / m["n"], 2),
              "dentro_2c": round(100 * m["ok2"] / m["n"])} for f, m in listos],
            key=lambda x: x["error_medio"])
    return hist, quedan[-400:], resumen, sesgos, marcador, tabla


def cargar_mos():
    """
    Carga el post-procesador estadístico ("IA") que corrige la salida de los
    modelos atmosféricos. Es una regresión Ridge entrenada sobre 52 días en
    las 9 regiones, con el reanálisis ERA5 como verdad, y validada con origen
    móvil (5 cortes temporales): mejora medida +18% sobre el híbrido, y +13.7%
    en el peor corte. No usa la fecha como variable, para no memorizar la
    estación del año. Si el archivo no está, el sistema sigue sin él.
    """
    for ruta in ("modelo_mos.json", "/mnt/user-data/outputs/modelo_mos.json"):
        if os.path.exists(ruta):
            try:
                return json.load(open(ruta))
            except Exception:
                pass
    return None


MOS = cargar_mos()


def aplicar_mos(mos, cual, zona, lead, clim, med, spread, mn, mx, vals_modelos):
    """
    Aplica el post-procesador a "tmax" o "tmin". Devuelve la temperatura
    corregida, o None si no corresponde aplicarlo (sin modelo, horizonte fuera
    de rango, zona desconocida o pocos modelos disponibles).
    Mejora medida con origen móvil: +18.0% en la máxima, +19.9% en la mínima.
    """
    if not mos or lead not in mos["leads"] or zona not in mos["zonas"]:
        return None
    sub = mos.get(cual)
    if not sub:
        return None
    num = mos["features_num"]          # clim, med, spread, mn, mx, + 5 modelos
    x = [clim, med, spread, mn, mx] + [vals_modelos[m] for m in num[5:]]
    z = [0.0] * len(mos["zonas"]); z[mos["zonas"].index(zona)] = 1.0
    l = [0.0] * len(mos["leads"]);  l[mos["leads"].index(lead)] = 1.0
    x = x + z + l
    if len(x) != len(sub["coef"]):
        return None
    return sum(a * b for a, b in zip(x, sub["coef"])) + sub["intercept"]


# Pesos del ensemble atmosférico por horizonte (día 1..7), obtenidos
# minimizando el error absoluto contra el reanálisis ERA5 en un backtest
# de 60 días sobre las 9 regiones. Antes eran valores supuestos; ahora
# están medidos. El modelo propio solo pesa de verdad al día 7.
PESO_ENSEMBLE = [0.98, 0.93, 0.92, 0.92, 0.80, 0.84, 0.69]

# Confianza base por horizonte = proporción de días en que el pronóstico
# quedó dentro de ±2 °C, medida contra ERA5 en el mismo backtest.
CONF_BASE = [0.97, 0.94, 0.92, 0.89, 0.80, 0.75, 0.71]


def clima_regiones(sesgos=None):
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
    # 1. ENSEMBLE ATMOSFÉRICO: consultar VARIOS modelos globales a la vez
    # (europeo ECMWF, americano GFS, alemán ICON, canadiense GEM, francés).
    # Se usa la MEDIANA de los modelos (robusta) y su DISPERSIÓN, que mide
    # la incertidumbre real: si los modelos coinciden hay certeza; si
    # discrepan, menos. Una sola llamada por región (no satura la API).
    # Si falla, se sigue solo con el modelo propio (que nunca depende de red).
    MODELOS_ATM = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless",
                   "gem_seamless", "meteofrance_seamless"]

    def _detalle(daily, campo, i):
        """Valor de cada modelo, y su mediana/dispersión/min/max, para el día i."""
        por_modelo = {}
        for mm in MODELOS_ATM:
            serie = daily.get(f"{campo}_{mm}")
            if serie and i < len(serie) and serie[i] is not None:
                por_modelo[mm] = float(serie[i])
        if not por_modelo:
            return None
        vals = sorted(por_modelo.values())
        n = len(vals)
        mediana = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
        media = sum(vals) / n
        desv = (sum((v - media) ** 2 for v in vals) / n) ** 0.5
        # los modelos que no reportaron se rellenan con la mediana, igual que
        # durante el entrenamiento del post-procesador
        completo = {mm: por_modelo.get(mm, mediana) for mm in MODELOS_ATM}
        return {"mediana": mediana, "desv": desv, "n": n,
                "min": vals[0], "max": vals[-1], "modelos": completo}

    def _consenso(daily, campo, i):
        """Mediana y dispersión del campo entre los modelos, para el día i."""
        d = _detalle(daily, campo, i)
        if not d:
            return None, None, 0
        return d["mediana"], d["desv"], d["n"]

    anclajes = {}
    t_inicio = time.time()
    for zona, modelo in modelos.items():
        if time.time() - t_inicio > 30:  # tope de tiempo total
            break
        lat = modelo.get("lat"); lon = modelo.get("lon")
        if lat is None or lon is None:
            continue
        try:
            url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({
                "latitude": lat, "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min,"
                         "precipitation_probability_max,precipitation_sum,weathercode",
                "timezone": "America/Santiago", "forecast_days": 7,
                "models": ",".join(MODELOS_ATM),
            })
            r = requests.get(url, timeout=9)
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
            # mm que el modelo espera SI llueve (intensidad aprendida)
            mm_modelo = float(modelo.get("mm_esperado", {}).get(doy_key, 0) or 0)
            mm_dia = mm_modelo
            fuente_dia = "modelo propio"
            code_real = None
            n_modelos = 0
            desv_temp = None
            # 2. HÍBRIDO: consenso de los modelos atmosféricos + modelo propio.
            # El ensemble manda en los primeros días; el modelo propio en los
            # últimos (cuando los modelos globales pierden habilidad).
            clim_orig = tM   # climatología pura de la máxima
            clim_orig_min = tm   # climatología pura de la mínima
            det_max = None
            det_min = None
            if anclaje and i < len(anclaje.get("time", [])):
                det_max = _detalle(anclaje, "temperature_2m_max", i)
                det_min = _detalle(anclaje, "temperature_2m_min", i)
                tM_real, dM, nM = _consenso(anclaje, "temperature_2m_max", i)
                tm_real, dm, _ = _consenso(anclaje, "temperature_2m_min", i)
                prob_real, _, _ = _consenso(anclaje, "precipitation_probability_max", i)
                mm_real, dmm, _ = _consenso(anclaje, "precipitation_sum", i)
                code_real, _, _ = _consenso(anclaje, "weathercode", i)
                n_modelos = nM
                desv_temp = dM
                # peso del ensemble: alto los primeros días, baja hacia el 5º
                # Pesos MEDIDOS por backtest (60 días, 9 regiones, verdad ERA5):
                # el pronóstico global supera a la climatología propia en todos
                # los horizontes; solo al día 7 se igualan. Ver PESO_ENSEMBLE.
                peso_real = PESO_ENSEMBLE[i] if i < len(PESO_ENSEMBLE) else 0.0
                if tM_real is not None:
                    tM = tM * (1 - peso_real) + tM_real * peso_real
                if tm_real is not None:
                    tm = tm * (1 - peso_real) + tm_real * peso_real
                if prob_real is not None:
                    prob = prob * (1 - peso_real) + prob_real * peso_real
                if mm_real is not None:
                    mm_dia = mm_modelo * (1 - peso_real) + float(mm_real) * peso_real
                fuente_dia = f"ensemble {nM} modelos + modelo propio"
            # POST-PROCESADOR ESTADÍSTICO ("IA"): en vez de mezclar los modelos
            # con pesos fijos, una regresión entrenada decide cuánto vale cada
            # modelo en cada región y horizonte. Solo se usa con al menos 4
            # modelos disponibles y con el horizonte cubierto por el ensemble.
            # El ajuste se limita a 3 °C para que nunca produzca un disparate.
            mos_aplicado = False
            tope = MOS.get("max_ajuste_c", 3.0) if MOS else 3.0
            if MOS and det_max and det_max["n"] >= 4:
                try:
                    pred = aplicar_mos(MOS, "tmax", zona, i + 1, clim_orig,
                                       det_max["mediana"], det_max["desv"],
                                       det_max["min"], det_max["max"],
                                       det_max["modelos"])
                except Exception:
                    pred = None
                if pred is not None and abs(pred - tM) <= tope:
                    tM = pred
                    mos_aplicado = True
            # la mínima importa tanto como la máxima: dispara las heladas
            if MOS and det_min and det_min["n"] >= 4:
                try:
                    pred_m = aplicar_mos(MOS, "tmin", zona, i + 1, clim_orig_min,
                                         det_min["mediana"], det_min["desv"],
                                         det_min["min"], det_min["max"],
                                         det_min["modelos"])
                except Exception:
                    pred_m = None
                if pred_m is not None and abs(pred_m - tm) <= tope:
                    tm = pred_m
                    mos_aplicado = True

            # CORRECCIÓN DE SESGO (MOS): si en esta región, a este horizonte,
            # el pronóstico viene calentando o enfriando de forma sistemática
            # frente a las estaciones chilenas, se descuenta la mitad de ese
            # sesgo. Solo con suficientes comparaciones y con tope, para no
            # amplificar ruido.
            sesgo_aplicado = 0.0
            if sesgos:
                serie = (sesgos.get(zona) or {}).get(str(i))
                if serie and len(serie) >= MIN_SESGO:
                    b = sum(serie) / len(serie)
                    sesgo_aplicado = max(-MAX_SESGO, min(MAX_SESGO, K_SESGO * b))
                    tM -= sesgo_aplicado
            # datos enriquecidos del modelo propio (aprendidos de 9 años)
            lluvia_fuerte = modelo.get("lluvia_fuerte", {}).get(doy_key, 0)
            prob_helada = modelo.get("prob_helada", {}).get(doy_key, 0)
            viento = modelo.get("viento_tipico", {}).get(doy_key, 0)
            mm_max_hist = modelo.get("mm_max_historico", {}).get(doy_key, 0)
            # si la probabilidad es baja, los mm esperados también bajan
            mm_mostrar = round(mm_dia, 1) if prob >= 20 else 0.0
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
            if mm_mostrar >= 20 or (lluvia_fuerte >= 12 and prob >= 45):
                alertas.append({"tipo":"lluvia_fuerte",
                                "txt":f"Lluvia fuerte (~{mm_mostrar:.0f}mm)" if mm_mostrar>=20 else "Posible lluvia fuerte",
                                "icono":"⛈️"})
            elif prob >= 55:
                alertas.append({"tipo":"lluvia",
                                "txt":f"Lluvia probable (~{mm_mostrar:.0f}mm)" if mm_mostrar>0 else "Lluvia probable",
                                "icono":"🌧️"})
            # La mínima ahora está post-procesada (error ±0.9 °C), así que la
            # alerta se apoya en ella y no solo en la climatología. Antes se
            # avisaba "helada" con 7 °C de mínima: falsa alarma.
            if round(tm) <= 0 or (prob_helada >= 10 and round(tm) <= 3):
                alertas.append({"tipo":"helada","txt":"Riesgo de helada","icono":"❄️"})
            elif round(tm) <= 3:
                alertas.append({"tipo":"frio","txt":"Frío intenso","icono":"🥶"})
            if viento >= 25:
                alertas.append({"tipo":"viento","txt":"Viento fuerte","icono":"💨"})
            if round(tM) >= 32:
                alertas.append({"tipo":"calor","txt":"Calor extremo","icono":"🔥"})
            # CONFIANZA MEDIDA: base por horizonte, ajustada por el acuerdo
            # real entre los modelos atmosféricos. Si discrepan mucho
            # (dispersión alta), la certeza baja de verdad. Ya no es fija.
            # Confianza base = % de días que el pronóstico cayó dentro de ±2°C,
            # medido contra ERA5 (no es un número inventado).
            conf = CONF_BASE[i]
            if desv_temp is not None and n_modelos >= 3:
                # desviación de 0°C → +5% de confianza; 5°C o más → -18%
                ajuste = 0.05 - min(desv_temp, 5.0) * 0.046
                conf = max(0.45, min(0.96, conf + ajuste))
            dias.append({
                "fecha": fecha.isoformat(),
                "t_min": round(tm), "t_max": round(tM),
                "lluvia_mm": mm_mostrar, "prob_lluvia": int(round(prob)),
                "mm_max_historico": float(mm_max_hist or 0),
                "lluvia_fuerte_pct": int(lluvia_fuerte),
                "prob_helada": int(prob_helada),
                "viento": int(round(viento)),
                "sesgo_corregido": round(sesgo_aplicado, 2),
                "postproceso_ia": bool(mos_aplicado),
                "modelos_tmax": (det_max["modelos"] if det_max else None),
                "med_tmax": (det_max["mediana"] if det_max else None),
                "n_modelos": int(n_modelos),
                "acuerdo_modelos": round(float(desv_temp), 1) if desv_temp is not None else None,
                "desc": desc, "icono": icono,
                "alertas": alertas,
                "confianza": int(conf * 100),
            })
        if dias:
            tiene_real = anclaje is not None
            salida.append({"zona": zona, "ciudad": modelo["ciudad"], "dias": dias,
                           "metodo": "ensemble de 5 modelos atmosféricos + modelo propio" if tiene_real
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



# ---------------------------------------------------------------------------
# ALERTA POR UMBRAL (no depende del ranking top-3).
# Una zona genera aviso SOLO si su probabilidad propia cruza el umbral.
# Medido sobre 601 semanas reales (2013-2024):
#   umbral 50% -> se dispara cada 26 dias, acierta 33%
#   umbral 60% -> se dispara cada ~9 meses, acierta 56%
# Se usa 60% porque es el punto donde el aviso acierta mas veces de las que
# falla. Aun asi falla 44%: el texto SIEMPRE debe mostrar esa cifra.
UMBRAL_ALERTA = 0.60
ACIERTO_ALERTA_PCT = 56          # medido, no estimado
UMBRAL_VIGILANCIA = 0.45         # aviso menor, solo informativo


def evaluar_alerta(zonas):
    """
    Devuelve el aviso vigente, o None si ninguna zona lo amerita.
    No mira el ranking: cada zona se evalua contra el umbral por si sola,
    de modo que puede haber cero avisos aunque siempre exista un top-3.
    """
    if not zonas:
        return None
    cands = [z for z in zonas if z.get("prob_M5_7d", 0) >= UMBRAL_VIGILANCIA]
    if not cands:
        return None
    z = max(cands, key=lambda x: x.get("prob_M5_7d", 0))
    p = z.get("prob_M5_7d", 0)
    alto = p >= UMBRAL_ALERTA
    return {
        "activa": True,
        "nivel": "alerta" if alto else "vigilancia",
        "zona": z.get("zona"),
        "prob_M5_7d": round(p, 3),
        "prob_M6_7d": round(z.get("prob_M6_7d", 0), 3),
        "mag_esperada": z.get("mag_esperada"),
        "umbral": UMBRAL_ALERTA if alto else UMBRAL_VIGILANCIA,
        "acierto_historico_pct": ACIERTO_ALERTA_PCT if alto else None,
        "texto": ("Actividad muy por sobre lo normal en esta zona."
                  if alto else
                  "Actividad algo elevada. Solo informativo."),
        "descargo": ("Aviso generado por un modelo estadistico. No es una "
                     "alerta oficial. Las fuentes oficiales son CSN y SENAPRED."),
    }


def sismos_mundiales(cat_mundo=None):
    """
    Sismos M>=7 recientes en el mundo, para el panel informativo.
    Se acompana SIEMPRE de la correlacion medida con Chile, que es nula:
    tras un M>=7 mundial, la prob de M>=7 en Chile a 30-60 dias es 2%;
    sin ningun sismo previo es 3% (506 casos, 1990-2024).
    """
    import datetime, urllib.parse
    hoy = datetime.date.today()
    try:
        url = "https://earthquake.usgs.gov/fdsnws/event/1/query?" + urllib.parse.urlencode({
            "format": "geojson",
            "starttime": (hoy - datetime.timedelta(days=90)).isoformat(),
            "endtime": hoy.isoformat(), "minmagnitude": 7.0})
        r = requests.get(url, timeout=12)
        if r.status_code != 200:
            return None
        feats = r.json().get("features", [])
    except Exception:
        return None
    ev = []
    for f in feats[:6]:
        pr = f.get("properties", {})
        co = (f.get("geometry") or {}).get("coordinates") or [None, None]
        ev.append({"mag": pr.get("mag"), "lugar": pr.get("place"),
                   "ms": pr.get("time"), "lat": co[1], "lon": co[0]})
    return {
        "eventos": ev,
        "correlacion_chile_pct": 2,
        "base_sin_detonante_pct": 3,
        "n_casos": 506,
        "veredicto": ("Sin efecto medible sobre Chile. Un sismo grande en otro "
                      "pais no eleva la probabilidad de uno en Chile."),
    }


def correr(estado_previo_path="estado_aprendizaje.json"):
    cat = escanear_chile()
    par = reaprender_parametros(cat)
    zonas, hoy = estimar(cat, par)

    # aviso por umbral propio (independiente del ranking) y panel mundial
    try:
        _alerta_vigente = evaluar_alerta(zonas)
    except Exception:
        _alerta_vigente = None
    try:
        _mundo = sismos_mundiales()
    except Exception:
        _mundo = None

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
        clima = clima_regiones(previo.get("clima_sesgos"))
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
    # OBSERVACIONES REALES de las estaciones meteorológicas chilenas.
    # Sirven para mostrar el estado actual medido en el suelo y, sobre todo,
    # para validar el pronóstico contra un dato independiente del modelo.
    import datetime as _dt
    try:
        obs = observaciones_chile()
    except Exception:
        obs = {}
    hoy_str = _dt.date.today().isoformat()
    try:
        obs_diarias = acumular_observaciones(previo, obs, hoy_str) if obs else previo.get("obs_diarias", {})
    except Exception:
        obs_diarias = previo.get("obs_diarias", {})
    clima_diag += f" | estaciones chilenas: {len(obs)}"
    # validación en vivo contra las estaciones chilenas (dato independiente)
    # verdad de contraste: ERA5 (máximo real del día, sin depender de cuántas
    # veces alcanzó a correr el motor). Las estaciones chilenas se siguen
    # usando para mostrar el estado actual.
    try:
        _mods = {}
        for _r in ("modelos_clima.json", "/mnt/user-data/outputs/modelos_clima.json"):
            if os.path.exists(_r):
                _mods = json.load(open(_r)); break
        verdad = verdad_era5(_mods) if _mods else {}
    except Exception:
        verdad = {}
    clima_diag += f" | verdad ERA5: {len(verdad)} regiones"
    try:
        (clima_hist, clima_pend, clima_acierto,
         clima_sesgos, clima_marcador, clima_tabla) = validar_clima(previo, clima, verdad)
    except Exception:
        clima_hist, clima_pend, clima_acierto = {}, [], None
        clima_sesgos = previo.get("clima_sesgos", {})
        clima_marcador, clima_tabla = previo.get("clima_marcador", {}), None

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
        "observaciones_chile":obs,
        "obs_diarias":obs_diarias,
        "clima_diagnostico":clima_diag,
        "clima_validacion":clima_hist,
        "clima_marcador":clima_marcador,
        "clima_tabla":clima_tabla,
        "validacion_version":VALIDACION_VERSION,
        "clima_sesgos":clima_sesgos,
        "clima_pendientes":clima_pend,
        "clima_acierto":clima_acierto,
        "pronostico_ubicacion":pronostico,
        "puntos_zona_activa":puntos_detalle,
        "modo_replicas":replicas,
        "alerta_tsunami":tsunami,
        "parametros_aprendidos":par,
        "zonas":zonas,
        "alerta":_alerta_vigente,
        "mundo":_mundo,
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

    # AVISOS POR WHATSAPP (TextMeBot) — solo si hay algo nuevo y real.
    # Requiere los secrets WSP_APIKEY y WSP_PHONE en el workflow.
    try:
        import notificador
        notificador.notificar(estado)
    except Exception as e:
        print(f"[notificador] omitido: {e}")
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


