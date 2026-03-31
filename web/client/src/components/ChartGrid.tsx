import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import type { TimeStep } from "../types";

interface Props {
  data: TimeStep[];
  running: boolean;
  statusLabel: string | null;
}

const FREE_ENERGY_DOMAIN: [number, number] = [-250, 500];
const FREE_ENERGY_TICKS = [-200, 0, 200, 400];
const BELIEF_DOMAIN: [number, number] = [-1.5, 2.5];
const BELIEF_TICKS = [-1, 0, 1, 2];
const OUTPUT_DOMAIN: [number, number] = [-3, 1.75];
const OUTPUT_TICKS = [-3, -2, -1, 0, 1];
const BODY_DOMAIN: [number, number] = [0, 2.5];
const BODY_TICKS = [0, 0.5, 1, 1.5, 2, 2.5];

export function ChartGrid({ data, running, statusLabel }: Props) {
  if (data.length === 0 && !running) {
    return (
      <div className="charts empty">
        <p>Select parameters and run the simulation.</p>
      </div>
    );
  }

  const chartData = data
    .filter((d) => d.beliefs)
    .map((d) => ({
      t: d.t,
      freeEnergy: d.freeEnergy,
      believedEnergy: d.beliefs.energy,
      believedReward: d.beliefs.reward,
      believedFatigue: d.beliefs.fatigue,
      energy: d.energy,
      feltEnergy: d.feltEnergy,
      reward: d.reward,
      fatigue: d.fatigue,
      reserves: d.reserves,
      capacity: d.capacity,
      activation: d.activation,
      effort: d.effort,
      actionTarget: d.actionTarget,
      fatigueState: d.fatigueState,
      opportunity: d.opportunity,
    }));

  return (
    <div className="charts">
      {running && data.length === 0 && (
        <div className="loading-overlay">{statusLabel ?? "Waiting..."}</div>
      )}

      <div className="chart-cell">
        <h3>Free energy</h3>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="t" label={{ value: "time", position: "bottom" }} />
            <YAxis domain={FREE_ENERGY_DOMAIN} ticks={FREE_ENERGY_TICKS} />
            <Tooltip />
            <Line
              type="monotone"
              dataKey="freeEnergy"
              stroke="#333"
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-cell">
        <h3>Beliefs about the world</h3>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="t" label={{ value: "time", position: "bottom" }} />
            <YAxis domain={BELIEF_DOMAIN} ticks={BELIEF_TICKS} />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="believedEnergy"
              name="expected vitality"
              stroke="#333"
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="believedReward"
              name="expected reward"
              stroke="#16a34a"
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="believedFatigue"
              name="expected fatigue"
              stroke="#dc2626"
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-cell">
        <h3>Output and felt energy</h3>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="t" label={{ value: "time", position: "bottom" }} />
            <YAxis domain={OUTPUT_DOMAIN} ticks={OUTPUT_TICKS} />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="energy"
              name="realized energy"
              stroke="#333"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="feltEnergy"
              name="felt energy"
              stroke="#7c3aed"
              strokeWidth={1.75}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="actionTarget"
              name="action target"
              stroke="#ea580c"
              strokeDasharray="8 3"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="reserves"
              name="reserves"
              stroke="#2563eb"
              strokeDasharray="6 3"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="activation"
              name="activation"
              stroke="#a855f7"
              strokeDasharray="2 2"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="opportunity"
              name="opportunity"
              stroke="#111827"
              strokeDasharray="10 4"
              strokeWidth={1.25}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-cell">
        <h3>Body state and cost memory</h3>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="t" label={{ value: "time", position: "bottom" }} />
            <YAxis domain={BODY_DOMAIN} ticks={BODY_TICKS} />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="capacity"
              name="capacity"
              stroke="#06b6d4"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="reserves"
              name="reserves"
              stroke="#2563eb"
              strokeDasharray="6 3"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="fatigue"
              name="observed fatigue"
              stroke="#dc2626"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="fatigueState"
              name="fatigue state"
              stroke="#1a1a1a"
              strokeDasharray="6 3"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-cell wide">
        <h3>Variable guide</h3>
        <div className="glossary">
          <div className="glossary-item">
            <code>expected vitality</code>
            <p>Intuitive: how energized the agent expects itself to feel.</p>
            <p>Mechanical: the first high-level belief prior, mixed into predicted <code>feltEnergy</code>.</p>
          </div>
          <div className="glossary-item">
            <code>expected reward</code>
            <p>Intuitive: how worth it the agent expects action to be.</p>
            <p>Mechanical: the second high-level belief prior, mixed into predicted opportunity-gated payoff.</p>
          </div>
          <div className="glossary-item">
            <code>expected fatigue</code>
            <p>Intuitive: how costly or draining the agent expects action to be.</p>
            <p>Mechanical: the third high-level belief prior, mixed into predicted fatigue observations.</p>
          </div>
          <div className="glossary-item">
            <code>opportunity</code>
            <p>Intuitive: whether there is something salient to act on right now.</p>
            <p>Mechanical: an exogenous pulse train that gates reward and body adaptation.</p>
          </div>
          <div className="glossary-item">
            <code>action target</code>
            <p>Intuitive: how strongly the agent is trying to mobilize.</p>
            <p>Mechanical: the raw action variable before body-limited leverage turns it into realized activation.</p>
          </div>
          <div className="glossary-item">
            <code>activation</code>
            <p>Intuitive: the latent mobilization state of the body.</p>
            <p>Mechanical: the fast hidden state <code>x(3)</code>, shaped by opportunity, beliefs, fatigue, and reserves.</p>
          </div>
          <div className="glossary-item">
            <code>felt energy</code>
            <p>Intuitive: how energized the person feels.</p>
            <p>Mechanical: an interoceptive observation derived from reserve-scaled activation, opportunity boost, and fatigue penalty.</p>
          </div>
          <div className="glossary-item">
            <code>realized energy</code>
            <p>Intuitive: how much output the body is actually producing.</p>
            <p>Mechanical: approximately <code>reserves × activation</code>.</p>
          </div>
          <div className="glossary-item">
            <code>reward</code>
            <p>Intuitive: the payoff or value of acting.</p>
            <p>Mechanical: opportunity-gated output benefit minus a fatigue penalty.</p>
          </div>
          <div className="glossary-item">
            <code>reserves</code>
            <p>Intuitive: currently available expendable energy.</p>
            <p>Mechanical: a slow body state that depletes under output and recovers toward capacity during rest.</p>
          </div>
          <div className="glossary-item">
            <code>capacity</code>
            <p>Intuitive: the body&apos;s longer-term ability to restore reserves.</p>
            <p>Mechanical: a slower body state shaped by homeostasis, opportunity-gated use, atrophy, and damage.</p>
          </div>
          <div className="glossary-item">
            <code>fatigue state</code>
            <p>Intuitive: accumulated internal tiredness.</p>
            <p>Mechanical: the latent fatigue state that builds with output and decays with rest.</p>
          </div>
          <div className="glossary-item">
            <code>observed fatigue</code>
            <p>Intuitive: how tired or strained the body feels.</p>
            <p>Mechanical: fatigue state plus an extra output-dependent load term.</p>
          </div>
          <div className="glossary-item">
            <code>free energy</code>
            <p>Intuitive: how badly the current model is explaining or controlling what is happening.</p>
            <p>Mechanical: the plotted <code>-DEM.J</code> time series from the ADEM solver, not a guaranteed monotone training loss.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
