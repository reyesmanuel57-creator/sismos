"""
Notificador WhatsApp para el sistema sísmico — vía TextMeBot.
Envía SOLO cuando hay algo nuevo y real. Nunca en bucle (WhatsApp bloquea).
Guarda lo ya enviado en aviso_enviados.json para no repetir.

Uso desde el motor:
    from notificador import notificar
    notificar(estado, apikey, telefono)

apikey y telefono se leen de variables de entorno en GitHub Actions:
    TEXTMEBOT_APIKEY  y  WHATSAPP_TELEFONO
"""
import os, json, time, datetime, urllib.parse, urllib.request

ARCHIVO_ENVIADOS = "aviso_enviados.json"
MAX_HIST = 200

def _cargar_enviados():
    try:
        return json.load(open(ARCHIVO_ENVIADOS))
    except Exception:
        return {"ids": []}

def _guardar_enviados(d):
    d["ids"] = d["ids"][-MAX_HIST:]
    try:
        json.dump(d, open(ARCHIVO_ENVIADOS, "w"))
    except Exception:
        pass

def _enviar(texto, apikey, telefono):
    """Envía un WhatsApp por TextMeBot. Devuelve True si ok."""
    if not apikey or not telefono:
        print("[notificador] falta apikey o teléfono — no se envía")
        return False
    url = ("https://api.textmebot.com/send.php?"
           + urllib.parse.urlencode({
               "recipient": telefono,
               "apikey": apikey,
               "text": texto,
               "json": "no"}))
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            resp = r.read().decode(errors="ignore")
        ok = "Success" in resp or "success" in resp
        print(f"[notificador] envío: {'OK' if ok else 'respuesta: '+resp[:80]}")
        return ok
    except Exception as e:
        print(f"[notificador] error al enviar: {e}")
        return False

def notificar(estado, apikey=None, telefono=None):
    """
    Revisa el estado y envía los avisos NUEVOS. Tres tipos:
      1) sismo M>=5 recién ocurrido
      2) zona en vigilancia (>=45%) o alerta (>=60%)
      3) réplicas esperadas tras un M>=6 (si el estado trae 'replicas')
    """
    apikey = apikey or os.environ.get("WSP_APIKEY")
    telefono = telefono or os.environ.get("WSP_PHONE")
    env = _cargar_enviados()
    ids = set(env["ids"])
    nuevos = []
    enviados_ahora = 0

    # --- 1) SISMOS M>=5 recientes ---
    for s in (estado.get("ultimos_sismos") or [])[:20]:
        mag = s.get("mag") or s.get("magnitude") or 0
        if mag < 5.0:
            continue
        sid = str(s.get("id") or f"{s.get('time','')}-{mag}")
        if sid in ids:
            continue
        lugar = s.get("lugar") or s.get("place") or "Chile"
        prof = s.get("depth") or s.get("prof") or "?"
        txt = (f"\U0001F534 SISMO M{mag} en Chile\n"
               f"{lugar}\n"
               f"Profundidad: {prof} km\n"
               f"(Reporte USGS/CSN, minutos despues del evento)")
        if _enviar(txt, apikey, telefono):
            ids.add(sid); nuevos.append(sid); enviados_ahora += 1
            time.sleep(6)  # respetar el límite de TextMeBot

    # --- 2) ALERTA / VIGILANCIA por umbral ---
    al = estado.get("alerta")
    if al and al.get("activa"):
        zona = al.get("zona"); prob = round(al.get("prob_M5_7d", 0) * 100)
        nivel = al.get("nivel")
        # id por día para no repetir el mismo aviso cada 30 min
        hoy = datetime.date.today().isoformat()
        aid = f"alerta-{nivel}-{zona}-{hoy}"
        if aid not in ids:
            if nivel == "alerta":
                txt = (f"\u26A0\uFE0F VIGILANCIA ELEVADA - {zona}\n"
                       f"Probabilidad de sismo M>=5 en 7 dias: {prob}%\n"
                       f"Este aviso acierta {al.get('acierto_historico_pct',56)}% de las veces "
                       f"(medido en 12 anios). No es alerta oficial. Fuentes: CSN, SENAPRED.")
            else:
                txt = (f"\U0001F7E1 Actividad algo elevada - {zona}\n"
                       f"Probabilidad M>=5 en 7 dias: {prob}%\n"
                       f"Solo informativo. Fuentes oficiales: CSN, SENAPRED.")
            if _enviar(txt, apikey, telefono):
                ids.add(aid); nuevos.append(aid); enviados_ahora += 1
                time.sleep(6)

    # --- 3) RÉPLICAS tras M>=6 (si el motor las calculó) ---
    rep = estado.get("replicas")
    if rep and rep.get("activo"):
        rid = f"replicas-{rep.get('sismo_id','')}"
        if rid not in ids:
            txt = (f"\U0001F4CA Tras el M{rep.get('mag')} de {rep.get('lugar','')}:\n"
                   f"Se esperan ~{rep.get('replicas_hoy','?')} replicas M>=3 hoy, "
                   f"decayendo en 2-3 semanas.\n"
                   f"Las replicas son normales tras un sismo grande.")
            if _enviar(txt, apikey, telefono):
                ids.add(rid); nuevos.append(rid); enviados_ahora += 1

    env["ids"] = list(ids)
    _guardar_enviados(env)
    print(f"[notificador] avisos enviados esta corrida: {enviados_ahora}")
    return enviados_ahora
