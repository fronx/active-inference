import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import type { TimeStep } from "../types";

interface Props {
  data: TimeStep[];
  running: boolean;
  statusLabel: string | null;
}

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
      reward: d.reward,
      fatigue: d.fatigue,
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
            <YAxis />
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
            <YAxis />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="believedEnergy"
              name="expected energy"
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
        <h3>Energy expenditure</h3>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="t" label={{ value: "time", position: "bottom" }} />
            <YAxis />
            <Tooltip />
            <ReferenceLine y={1.0} stroke="#999" strokeDasharray="3 3" />
            <Line
              type="monotone"
              dataKey="energy"
              stroke="#333"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-cell">
        <h3>Outcomes</h3>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="t" label={{ value: "time", position: "bottom" }} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="reward"
              stroke="#16a34a"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="fatigue"
              stroke="#dc2626"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
