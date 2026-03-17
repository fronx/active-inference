import type { SimulationParams, PresetName } from "./types";

export const PRESETS: Record<Exclude<PresetName, "custom">, SimulationParams> =
  {
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
