// Endpoint que entrega el estado vigente del monitor sísmico.
// Lee el resultado guardado en Netlify Blobs. Si no existe o está vencido
// (más de ~6 horas), recalcula al vuelo con datos en vivo del USGS y lo guarda.
// Así la página funciona desde el primer despliegue, sin esperar al cron.
import { getStore } from "@netlify/blobs";
import { buildState } from "./lib/model.mjs";

const MAX_EDAD_MS = 6.5 * 60 * 60 * 1000;

export default async () => {
  const store = getStore("sismos");
  let estado = await store.get("estado", { type: "json" }).catch(() => null);

  const nowMs = Date.parse(new Date().toISOString());
  const vencido = !estado || !estado.actualizado ||
    nowMs - Date.parse(estado.actualizado) > MAX_EDAD_MS;

  if (vencido) {
    try {
      const prevHistory = (estado && estado.historial_parametros) || [];
      estado = await buildState(nowMs, prevHistory);
      await store.setJSON("estado", estado);
    } catch (err) {
      // si el USGS falla, servimos lo último que tengamos
      if (!estado) {
        return new Response(JSON.stringify({ error: "sin datos disponibles" }), {
          status: 503,
          headers: { "content-type": "application/json" },
        });
      }
    }
  }

  return new Response(JSON.stringify(estado), {
    headers: {
      "content-type": "application/json",
      "cache-control": "public, max-age=300",
    },
  });
};

export const config = {
  path: "/api/estado",
};
