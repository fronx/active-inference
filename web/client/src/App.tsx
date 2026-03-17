import { useState, useCallback, useRef } from "react";
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

export default function App() {
  const [params, setParams] = useState<SimulationParams>(PRESETS.healthy);
  const [preset, setPreset] = useState<PresetName>("healthy");
  const [data, setData] = useState<TimeStep[]>([]);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const runId = useRef(0);

  const handleParamChange = useCallback((p: SimulationParams) => {
    setParams(p);
    setPreset(detectPreset(p));
  }, []);

  const handlePreset = useCallback((name: PresetName) => {
    setPreset(name);
    if (name !== "custom") {
      setParams(PRESETS[name]);
    }
  }, []);

  const handleRun = useCallback(() => {
    const id = ++runId.current;
    setRunning(true);
    setStatus("sending");
    setError(null);
    setData([]);

    simulate(
      params,
      (points) => {
        if (runId.current === id) setData(points);
      },
      (s) => {
        if (runId.current === id) setStatus(s);
      },
      () => {
        if (runId.current === id) {
          setRunning(false);
          setStatus(null);
        }
      },
      (msg) => {
        if (runId.current === id) {
          setError(msg);
          setRunning(false);
          setStatus(null);
        }
      },
    );
  }, [params]);

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
