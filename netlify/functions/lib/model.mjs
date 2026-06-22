// Modelo sísmico auto-adaptativo para Chile.
// Descarga el catálogo reciente del USGS y recalcula, en cada ejecución,
// el valor-b, las probabilidades por zona y los parámetros tipo ETAS
// (productividad K, alpha, Omori c/p) a partir de los datos reales.
//
// Nota honesta: las probabilidades son estimaciones Poisson + Gutenberg-Richter
// alimentadas con datos en vivo del USGS. NO son predicciones de día, hora ni
// lugar exacto. Fuente oficial de emergencias: SENAPRED y sismologia.cl.

const LOG10E = Math.LOG10E; // 0.4342944819...

// Caja envolvente de Chile (incluye zona de subducción frente a la costa).
export const CHILE = {
  minlat: -57,
  maxlat: -17,
  minlon: -77,
  maxlon: -65,
};

// Zonas por latitud (norte = menos negativo).
export const ZONES = [
  { zona: "Norte Grande (Arica–Antofagasta)", latMin: -26, latMax: -17 },
  { zona: "Norte Chico (Atacama–Coquimbo)", latMin: -32, latMax: -26 },
  { zona: "Centro (Valparaíso–Maule)", latMin: -36, latMax: -32 },
  { zona: "Sur (Biobío–Los Lagos)", latMin: -44, latMax: -36 },
  { zona: "Austral (Aysén–Magallanes)", latMin: -57, latMax: -44 },
];

const MC = 4.5; // magnitud de completitud aproximada del catálogo USGS en Chile
const AFT_MIN = 4.0; // umbral del catálogo: réplicas contadas desde M4.0 (más señal)
const WINDOW_YEARS = 6; // ventana de aprendizaje
const DAY_MS = 86400000;

function isoDaysAgo(now, days) {
  return new Date(now - days * DAY_MS).toISOString().slice(0, 10);
}

// Descarga el catálogo del USGS (FDSN) para la ventana de aprendizaje.
// Devuelve eventos {t (ms), mag, lat, lon}.
export async function fetchCatalog(nowMs) {
  const start = isoDaysAgo(nowMs, WINDOW_YEARS * 365);
  const params = new URLSearchParams({
    format: "geojson",
    starttime: start,
    minlatitude: String(CHILE.minlat),
    maxlatitude: String(CHILE.maxlat),
    minlongitude: String(CHILE.minlon),
    maxlongitude: String(CHILE.maxlon),
    minmagnitude: "4.0",
    orderby: "time",
    limit: "20000",
  });
  const url = `https://earthquake.usgs.gov/fdsnws/event/1/query?${params}`;
  const res = await fetch(url, { headers: { "User-Agent": "monitor-sismico-chile" } });
  if (!res.ok) throw new Error(`USGS respondió ${res.status}`);
  const gj = await res.json();
  const events = [];
  for (const f of gj.features || []) {
    const c = f.geometry && f.geometry.coordinates;
    const mag = f.properties && f.properties.mag;
    const t = f.properties && f.properties.time;
    if (!c || mag == null || t == null) continue;
    events.push({ t, mag, lon: c[0], lat: c[1] });
  }
  // orden ascendente por tiempo
  events.sort((a, b) => a.t - b.t);
  return events;
}

function zoneOf(lat) {
  for (const z of ZONES) if (lat >= z.latMin && lat < z.latMax) return z.zona;
  return null;
}

// Valor-b por máxima verosimilitud (Aki 1965), con corrección de binning 0.1.
function bValue(mags) {
  const m = mags.filter((x) => x >= MC);
  if (m.length < 30) return { b: 1.0, n: m.length, ok: false };
  const mean = m.reduce((s, x) => s + x, 0) / m.length;
  const b = LOG10E / (mean - (MC - 0.05));
  return { b: clamp(b, 0.6, 1.8), n: m.length, ok: true };
}

function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

function haversineDeg(aLat, aLon, bLat, bLon) {
  // distancia angular aproximada en grados (suficiente para agrupar réplicas)
  const dLat = aLat - bLat;
  const dLon = (aLon - bLon) * Math.cos((((aLat + bLat) / 2) * Math.PI) / 180);
  return Math.sqrt(dLat * dLat + dLon * dLon);
}

// Productividad de réplicas: regresa log10(N_réplicas) vs (Mm - Mc)
// para obtener alpha y K. Identifica sismos principales (M>=5) sin un evento
// mayor en los 7 días previos dentro de ~2°.
function productivity(events) {
  const mains = [];
  for (let i = 0; i < events.length; i++) {
    const e = events[i];
    if (e.mag < 5.0) continue;
    let isMain = true;
    for (let j = i - 1; j >= 0; j--) {
      if (e.t - events[j].t > 7 * DAY_MS) break;
      if (events[j].mag > e.mag && haversineDeg(e.lat, e.lon, events[j].lat, events[j].lon) < 2)
        { isMain = false; break; }
    }
    if (isMain) mains.push({ idx: i, e });
  }
  const xs = [], ys = [];
  for (const { idx, e } of mains) {
    let n = 0;
    for (let j = idx + 1; j < events.length; j++) {
      if (events[j].t - e.t > 10 * DAY_MS) break;
      if (events[j].mag >= AFT_MIN && events[j].mag < e.mag &&
          haversineDeg(e.lat, e.lon, events[j].lat, events[j].lon) < 1.5) n++;
    }
    xs.push(e.mag - MC);
    ys.push(Math.log10(n + 0.1));
  }
  if (xs.length < 8) return { K: 0.008, alpha: 1.5, ok: false, nMains: xs.length };
  const { slope, intercept } = linreg(xs, ys);
  // si la regresión no es físicamente plausible (productividad debe crecer con la
  // magnitud), usamos valores estándar de subducción y marcamos no-convergencia.
  if (slope < 0.3) return { K: 0.008, alpha: 1.5, ok: false, nMains: xs.length };
  const alpha = clamp(slope, 0.6, 2.5);
  const K = clamp(Math.pow(10, intercept), 1e-4, 1);
  return { K, alpha, ok: true, nMains: xs.length };
}

// Ley de Omori-Utsu (p, c) sobre la mayor secuencia de réplicas del catálogo.
function omori(events) {
  // sismo principal más grande
  let big = null;
  for (const e of events) if (!big || e.mag > big.mag) big = e;
  if (!big || big.mag < 5.5) return { p: 1.1, c: 0.01, ok: false, nAft: 0 };
  const aft = [];
  for (const e of events) {
    const dt = (e.t - big.t) / DAY_MS;
    if (dt > 0 && dt <= 30 && e.mag >= AFT_MIN && e.mag < big.mag &&
        haversineDeg(big.lat, big.lon, e.lat, e.lon) < 1.5) aft.push(dt);
  }
  if (aft.length < 30) return { p: 1.1, c: 0.01, ok: false, nAft: aft.length };
  // tasa por bins logarítmicos: n(t) ∝ (t+c)^-p  → log(rate) = -p*log(t+c)+k
  const c = 0.01;
  const edges = [];
  for (let e = -2; e <= 1.5; e += 0.25) edges.push(Math.pow(10, e));
  const xs = [], ys = [];
  for (let i = 0; i < edges.length - 1; i++) {
    const lo = edges[i], hi = edges[i + 1];
    const cnt = aft.filter((t) => t >= lo && t < hi).length;
    if (cnt === 0) continue;
    const mid = Math.sqrt(lo * hi);
    const rate = cnt / (hi - lo);
    xs.push(Math.log10(mid + c));
    ys.push(Math.log10(rate));
  }
  if (xs.length < 3) return { p: 1.1, c, ok: false, nAft: aft.length };
  const { slope } = linreg(xs, ys);
  const p = clamp(-slope, 0.8, 1.8);
  return { p, c, ok: true, nAft: aft.length };
}

function linreg(xs, ys) {
  const n = xs.length;
  const mx = xs.reduce((s, x) => s + x, 0) / n;
  const my = ys.reduce((s, y) => s + y, 0) / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    num += (xs[i] - mx) * (ys[i] - my);
    den += (xs[i] - mx) * (xs[i] - mx);
  }
  const slope = den === 0 ? 0 : num / den;
  return { slope, intercept: my - slope * mx };
}

function nivelFor(probM5, probM6, magMax30) {
  if (probM6 >= 0.06 || probM5 >= 0.6 || magMax30 >= 6.5) return "ELEVADO";
  if (probM6 >= 0.025 || probM5 >= 0.3) return "MODERADO";
  return "NORMAL";
}

// Construye el estado completo a partir del catálogo y del historial previo.
export function computeState(events, nowMs, prevHistory = []) {
  const spanDays = events.length
    ? Math.max(1, (nowMs - events[0].t) / DAY_MS)
    : WINDOW_YEARS * 365;

  const bv = bValue(events.map((e) => e.mag));
  const prod = productivity(events);
  const om = omori(events);

  // tasa base nacional de M>=5 (eventos/día)
  const nM5 = events.filter((e) => e.mag >= 5.0).length;
  const mu = nM5 / spanDays;

  const zonas = ZONES.map((z) => {
    const inZone = events.filter((e) => e.lat >= z.latMin && e.lat < z.latMax);
    const m5all = inZone.filter((e) => e.mag >= 5.0).length;
    const last30 = inZone.filter((e) => nowMs - e.t <= 30 * DAY_MS);
    const last7 = inZone.filter((e) => nowMs - e.t <= 7 * DAY_MS);
    const magMax30 = last30.reduce((mx, e) => Math.max(mx, e.mag), 0);

    // tasa base de M>=5 por día en la zona
    const lambda5bg = m5all / spanDays;

    // factor de actividad reciente (proxy de disparo tipo ETAS):
    // compara la tasa de los últimos 30 días con la tasa histórica de la zona
    const rate30 = inZone.filter((e) => nowMs - e.t <= 30 * DAY_MS).length / 30;
    const baseRate = inZone.length / spanDays;
    let factor = baseRate > 0 ? rate30 / baseRate : 1;
    factor = clamp(factor, 0.3, 6);

    const lambda5 = lambda5bg * factor;
    const lambda6 = lambda5 * Math.pow(10, -bv.b); // Gutenberg-Richter de M5 a M6
    const probM5 = 1 - Math.exp(-lambda5 * 7);
    const probM6 = 1 - Math.exp(-lambda6 * 7);

    return {
      zona: z.zona,
      prob_M5_7d: round(probM5, 3),
      prob_M6_7d: round(probM6, 3),
      nivel: nivelFor(probM5, probM6, magMax30),
      n_ult7d: last7.length,
      n_ult30d: last30.length,
      mag_max_ult30d: round(magMax30, 1),
    };
  });

  const ultimoSismo = events.length ? new Date(events[events.length - 1].t).toISOString() : null;
  const convergencia = bv.ok && prod.ok && om.ok;

  const parametros = {
    mu: round6(mu),
    K: round6(prod.K),
    alpha: round6(prod.alpha),
    c: round6(om.c),
    p: round6(om.p),
    b: round6(bv.b),
    n_eventos_aprendizaje: bv.n,
    convergencia,
  };

  const nowIso = new Date(nowMs).toISOString();
  const historial = [...prevHistory];
  historial.push({
    fecha: nowIso,
    mu: round(mu, 4),
    K: round(prod.K, 4),
    alpha: round(prod.alpha, 3),
    p: round(om.p, 3),
    b: round(bv.b, 3),
    n_total: events.length,
  });
  // conserva las últimas 60 actualizaciones
  while (historial.length > 60) historial.shift();

  return {
    actualizado: nowIso,
    ultimo_sismo: ultimoSismo,
    n_eventos_escaneados: events.length,
    parametros_aprendidos: parametros,
    zonas,
    historial_parametros: historial,
    descargo:
      "Sistema auto-adaptativo alimentado con datos en vivo del USGS. " +
      "Probabilidades por zona para 7 días (Poisson + Gutenberg-Richter), " +
      "NO predicción de día/hora/lugar. Oficial: SENAPRED, sismologia.cl.",
  };
}

function round(x, d) {
  const f = Math.pow(10, d);
  return Math.round(x * f) / f;
}
function round6(x) {
  return round(x, 6);
}

// Descarga + cómputo en un solo paso.
export async function buildState(nowMs, prevHistory = []) {
  const events = await fetchCatalog(nowMs);
  return computeState(events, nowMs, prevHistory);
}
