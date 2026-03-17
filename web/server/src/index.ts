import express from "express";
import cors from "cors";
import { startWorker, runSimulation } from "./octave.js";
import { getCache, setCache, paramHash } from "./cache.js";
import type { SimulationParams } from "./types.js";

const app = express();
app.use(cors());
app.use(express.json());

function log(msg: string) {
  console.log(`[server] ${msg}`);
}

const PRESETS: Record<string, SimulationParams> = {
  healthy: {
    beliefPrior: [1.0, 0.5, 0.3],
    M2V: -2,
    M1V: 3,
    N: 48,
  },
  depressed: {
    beliefPrior: [0.6, 0.2, 0.8],
    M2V: 4,
    M1V: 0,
    N: 48,
  },
  manic: {
    beliefPrior: [1.4, 0.8, 0.2],
    M2V: 4,
    M1V: 0,
    N: 48,
  },
};

app.get("/api/presets", (_req, res) => {
  res.json(PRESETS);
});

// SSE endpoint: streams results one timestep at a time
app.post("/api/simulate", async (req, res) => {
  const params: SimulationParams = req.body;

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders();

  try {
    const hash = paramHash(params);
    const cached = getCache(hash);

    if (cached) {
      log(`Cache hit (${hash})`);
    } else {
      log(`Cache miss (${hash}) — running simulation...`);
      res.write(`data: ${JSON.stringify({ status: "computing" })}\n\n`);
    }

    const result = cached ?? (await runSimulation(params));
    if (!cached) {
      setCache(hash, result);
      log(`Simulation complete, cached as ${hash}`);
    }

    // stream one timestep at a time
    const N = result.timesteps.length;
    for (let t = 0; t < N; t++) {
      const point = {
        t: result.timesteps[t],
        freeEnergy: result.freeEnergy[t],
        beliefs: {
          energy: result.beliefs.energy[t],
          reward: result.beliefs.reward[t],
          fatigue: result.beliefs.fatigue[t],
        },
        energy: result.energy[t],
        reward: result.reward[t],
        fatigue: result.fatigue[t],
      };
      res.write(`data: ${JSON.stringify(point)}\n\n`);
    }

    res.write("data: [DONE]\n\n");
    res.end();
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    log(`Error: ${message}`);
    res.write(`data: ${JSON.stringify({ error: message })}\n\n`);
    res.end();
  }
});

const PORT = 3001;

async function main() {
  log("Booting...");
  await startWorker();
  app.listen(PORT, () => {
    log(`Listening on http://localhost:${PORT}`);
  });
}

main().catch((err) => {
  console.error("Failed to start:", err);
  process.exit(1);
});
