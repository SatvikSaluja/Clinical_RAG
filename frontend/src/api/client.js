// In dev, Vite proxies "/api" to localhost:8000 (see vite.config.js). In a
// static prod build there's no proxy, so VITE_API_BASE must point at the
// deployed backend's own URL (set at build time).
const BASE = `${import.meta.env.VITE_API_BASE || ""}/api`;

async function handle(resp) {
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw new Error(`API error ${resp.status}: ${text}`);
  }
  return resp.json();
}

export async function listPatients() {
  return handle(await fetch(`${BASE}/patients`));
}

export async function getPatient(patientId) {
  return handle(await fetch(`${BASE}/patients/${patientId}`));
}

export async function runQuery(patientId, question, config) {
  return handle(
    await fetch(`${BASE}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ patient_id: patientId, question, config }),
    })
  );
}

export async function getDashboard() {
  return handle(await fetch(`${BASE}/dashboard`));
}
