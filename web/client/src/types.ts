export interface SimulationParams {
  beliefPrior: [number, number, number]; // [expected_energy, expected_reward, expected_fatigue]
  M2V: number; // prior precision exponent
  M1V: number; // sensory precision exponent
  N: number; // time steps
}

export interface TimeStep {
  t: number;
  freeEnergy: number;
  beliefs: {
    energy: number;
    reward: number;
    fatigue: number;
  };
  energy: number;
  feltEnergy: number;
  reward: number;
  fatigue: number;
  reserves: number;
  capacity: number;
  activation: number;
  effort: number;
  fatigueState: number;
  actionTarget: number;
  opportunity: number;
}

export type PresetName = "healthy" | "depressed" | "manic" | "custom";
