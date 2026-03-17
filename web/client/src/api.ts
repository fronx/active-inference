import type { SimulationParams, TimeStep } from "./types";

const API = "http://localhost:3001";

const cache = new Map<string, TimeStep[]>();

async function hashParams(params: SimulationParams): Promise<string> {
  const sorted = JSON.stringify(params, Object.keys(params).sort());
  const buf = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(sorted),
  );
  return Array.from(new Uint8Array(buf).slice(0, 8))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

export async function simulate(
  params: SimulationParams,
  onPoint: (points: TimeStep[]) => void,
  onStatus: (status: string) => void,
  onDone: () => void,
  onError: (msg: string) => void,
): Promise<void> {
  const hash = await hashParams(params);
  const cached = cache.get(hash);
  if (cached) {
    onStatus("cached");
    for (let i = 0; i < cached.length; i++) {
      onPoint(cached.slice(0, i + 1));
    }
    onDone();
    return;
  }

  onStatus("sending");

  const res = await fetch(`${API}/api/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });

  if (!res.ok || !res.body) {
    onError(`Server error: ${res.status}`);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const points: TimeStep[] = [];

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const payload = line.slice(6);
      if (payload === "[DONE]") {
        cache.set(hash, points);
        onDone();
        return;
      }
      const parsed = JSON.parse(payload);
      if (parsed.error) {
        onError(parsed.error);
        return;
      }
      if (parsed.status) {
        onStatus(parsed.status);
        continue;
      }
      points.push(parsed as TimeStep);
      onPoint([...points]);
      if (points.length === 1) onStatus("streaming");
    }
  }

  if (points.length > 0) {
    cache.set(hash, points);
  }
  onDone();
}
