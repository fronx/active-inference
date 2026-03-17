export interface SimulationParams {
  beliefPrior: [number, number, number]; // [expected_energy, expected_reward, expected_fatigue]
  M2V: number; // prior precision exponent
  M1V: number; // sensory precision exponent
  N: number; // time steps
}

export interface SimulationResult {
  timesteps: number[];
  freeEnergy: number[];
  beliefs: {
    energy: number[];
    reward: number[];
    fatigue: number[];
  };
  energy: number[];
  reward: number[];
  fatigue: number[];
}
