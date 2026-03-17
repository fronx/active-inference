import { createHash } from "crypto";
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import type { SimulationParams, SimulationResult } from "./types.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const CACHE_DIR = join(__dirname, "..", ".cache");

mkdirSync(CACHE_DIR, { recursive: true });

export function paramHash(params: SimulationParams): string {
  const sorted = JSON.stringify(params, Object.keys(params).sort());
  return createHash("sha256").update(sorted).digest("hex").slice(0, 16);
}

export function getCache(hash: string): SimulationResult | null {
  const path = join(CACHE_DIR, `${hash}.json`);
  if (!existsSync(path)) return null;
  return JSON.parse(readFileSync(path, "utf-8"));
}

export function setCache(hash: string, data: SimulationResult): void {
  const path = join(CACHE_DIR, `${hash}.json`);
  writeFileSync(path, JSON.stringify(data));
}
