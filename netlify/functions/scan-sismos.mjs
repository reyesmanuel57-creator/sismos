// Función programada: escanea el USGS y recalcula el modelo cada 6 horas.
// Guarda el resultado en Netlify Blobs para que la página lo muestre actualizado.
import { getStore } from "@netlify/blobs";
import { buildState } from "./lib/model.mjs";

export default async () => {
  const store = getStore("sismos");
  const prev = await store.get("estado", { type: "json" }).catch(() => null);
  const prevHistory = (prev && prev.historial_parametros) || [];

  const nowMs = Date.parse(new Date().toISOString());
  const estado = await buildState(nowMs, prevHistory);
  await store.setJSON("estado", estado);

  return new Response(
    JSON.stringify({ ok: true, actualizado: estado.actualizado, eventos: estado.n_eventos_escaneados }),
    { headers: { "content-type": "application/json" } }
  );
};

export const config = {
  schedule: "0 */6 * * *",
};
