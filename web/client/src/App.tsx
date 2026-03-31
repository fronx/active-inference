import { useState, useCallback, useEffect, useRef } from "react";
import { ParameterPanel } from "./components/ParameterPanel";
import { ChartGrid } from "./components/ChartGrid";
import { simulate } from "./api";
import { PRESETS } from "./presets";
import type { SimulationParams, PresetName, TimeStep } from "./types";

const STATUS_LABELS: Record<string, string> = {
  sending: "Sending to Octave...",
  computing: "Running simulation...",
  streaming: "Receiving results...",
  cached: "Loading from cache...",
};

function deepEqual(a: SimulationParams, b: SimulationParams): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

function detectPreset(params: SimulationParams): PresetName {
  for (const [name, preset] of Object.entries(PRESETS)) {
    if (deepEqual(params, preset)) return name as PresetName;
  }
  return "custom";
}

function paramsToSearch(params: SimulationParams): string {
  const preset = detectPreset(params);
  if (preset !== "custom") return `?p=${preset}`;
  const b = params.beliefPrior.map((v) => v.toFixed(2)).join(",");
  return `?b=${b}&pv=${params.M2V}&sv=${params.M1V}&n=${params.N}`;
}

function paramsFromSearch(search: string): SimulationParams | null {
  const u = new URLSearchParams(search);
  const p = u.get("p");
  if (p && p in PRESETS) return PRESETS[p as keyof typeof PRESETS];
  const b = u.get("b");
  if (!b) return null;
  const parts = b.split(",").map(Number);
  if (parts.length !== 3 || parts.some(isNaN)) return null;
  const pv = Number(u.get("pv"));
  const sv = Number(u.get("sv"));
  const n = Number(u.get("n"));
  if ([pv, sv, n].some(isNaN)) return null;
  return { beliefPrior: [parts[0], parts[1], parts[2]], M2V: pv, M1V: sv, N: n };
}

export default function App() {
  const initial = paramsFromSearch(location.search) ?? PRESETS.healthy;
  const [params, setParams] = useState<SimulationParams>(initial);
  const [preset, setPreset] = useState<PresetName>(detectPreset(initial));
  const [data, setData] = useState<TimeStep[]>([]);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const runId = useRef(0);
  const skipPush = useRef(false);

  function runSim(p: SimulationParams, pushHistory: boolean) {
    if (pushHistory && !skipPush.current) {
      const search = paramsToSearch(p);
      if (location.search !== search) history.pushState(null, "", search);
    }
    skipPush.current = false;

    const id = ++runId.current;
    setRunning(true);
    setStatus("sending");
    setError(null);
    setData([]);
    simulate(
      p,
      (points) => { if (runId.current === id) setData(points); },
      (s) => { if (runId.current === id) setStatus(s); },
      () => { if (runId.current === id) { setRunning(false); setStatus(null); } },
      (msg) => { if (runId.current === id) { setError(msg); setRunning(false); setStatus(null); } },
    );
  }

  // run on first load
  useEffect(() => {
    runSim(initial, false);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // handle back/forward
  useEffect(() => {
    function onPopState() {
      const p = paramsFromSearch(location.search) ?? PRESETS.healthy;
      setParams(p);
      setPreset(detectPreset(p));
      skipPush.current = true;
      runSim(p, false);
    }
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleParamChange = useCallback((p: SimulationParams) => {
    setParams(p);
    setPreset(detectPreset(p));
  }, []);

  const handlePreset = useCallback((name: PresetName) => {
    if (name === "custom") return;
    const p = PRESETS[name];
    setParams(p);
    setPreset(name);
    runSim(p, true);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleRun = useCallback(() => {
    runSim(params, true);
  }, [params]); // eslint-disable-line react-hooks/exhaustive-deps

  const statusLabel = status ? STATUS_LABELS[status] ?? status : null;

  return (
    <div className="app">
      <header>
        <h1>Active Inference: Psychology</h1>
      </header>
      <div className="main">
        <ParameterPanel
          params={params}
          preset={preset}
          running={running}
          onChange={handleParamChange}
          onPreset={handlePreset}
          onRun={handleRun}
        />
        <div className="chart-area">
          {error && <div className="error">{error}</div>}
          <ChartGrid data={data} running={running} statusLabel={statusLabel} />
        </div>
      </div>
    </div>
  );
}
