import { type SimulationParams, type PresetName } from "../types";

interface Props {
  params: SimulationParams;
  preset: PresetName;
  running: boolean;
  onChange: (params: SimulationParams) => void;
  onPreset: (name: PresetName) => void;
  onRun: () => void;
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  suffix,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  suffix?: string;
}) {
  return (
    <div className="slider-row">
      <label>
        <span className="slider-label">{label}</span>
        <span className="slider-value">
          {value.toFixed(1)}
          {suffix}
        </span>
      </label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </div>
  );
}

function setBelief(
  params: SimulationParams,
  idx: number,
  val: number,
): SimulationParams {
  const arr = [...params.beliefPrior] as [number, number, number];
  arr[idx] = val;
  return { ...params, beliefPrior: arr };
}

export function ParameterPanel({
  params,
  preset,
  running,
  onChange,
  onPreset,
  onRun,
}: Props) {
  function set<K extends keyof SimulationParams>(
    key: K,
    value: SimulationParams[K],
  ) {
    onChange({ ...params, [key]: value });
  }

  return (
    <div className="panel">
      <h2>Parameters</h2>

      <div className="preset-row">
        <label>Preset</label>
        <select
          value={preset}
          onChange={(e) => onPreset(e.target.value as PresetName)}
        >
          <option value="healthy">Healthy</option>
          <option value="depressed">Depressed</option>
          <option value="manic">Manic</option>
          <option value="custom">Custom</option>
        </select>
      </div>

      <fieldset>
        <legend>Beliefs about the world</legend>
        <Slider
          label="Expected energy"
          value={params.beliefPrior[0]}
          min={0}
          max={2}
          step={0.1}
          onChange={(v) => onChange(setBelief(params, 0, v))}
        />
        <Slider
          label="Expected reward"
          value={params.beliefPrior[1]}
          min={0}
          max={1.5}
          step={0.1}
          onChange={(v) => onChange(setBelief(params, 1, v))}
        />
        <Slider
          label="Expected fatigue"
          value={params.beliefPrior[2]}
          min={0}
          max={1.5}
          step={0.1}
          onChange={(v) => onChange(setBelief(params, 2, v))}
        />
      </fieldset>

      <fieldset>
        <legend>Precision (how rigid vs flexible)</legend>
        <Slider
          label="Prior precision"
          value={params.M2V}
          min={-4}
          max={8}
          step={0.5}
          onChange={(v) => set("M2V", v)}
          suffix={` (exp=${Math.exp(params.M2V).toFixed(1)})`}
        />
        <Slider
          label="Sensory precision"
          value={params.M1V}
          min={-2}
          max={8}
          step={0.5}
          onChange={(v) => set("M1V", v)}
          suffix={` (exp=${Math.exp(params.M1V).toFixed(1)})`}
        />
      </fieldset>

      <fieldset>
        <legend>Simulation</legend>
        <Slider
          label="Time steps"
          value={params.N}
          min={16}
          max={96}
          step={8}
          onChange={(v) => set("N", v)}
        />
      </fieldset>

      <button className="run-btn" onClick={onRun} disabled={running}>
        {running ? "Running..." : "Run simulation"}
      </button>
    </div>
  );
}
