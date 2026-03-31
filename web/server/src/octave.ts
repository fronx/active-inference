import { spawn, execSync, type ChildProcess } from "child_process";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import { writeFileSync, unlinkSync } from "fs";
import type { SimulationParams, SimulationResult } from "./types.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = join(__dirname, "..", "..", "..");
const FIFO_PATH = "/tmp/octave_psychology_worker";

let worker: ChildProcess | null = null;
let ready = false;
let queue: Array<{
  params: SimulationParams;
  resolve: (r: SimulationResult) => void;
  reject: (e: Error) => void;
}> = [];
let processing = false;
let stdoutBuffer = "";

function log(msg: string) {
  console.log(`[octave] ${msg}`);
}

function createFifo() {
  try {
    unlinkSync(FIFO_PATH);
  } catch {}
  execSync(`mkfifo ${FIFO_PATH}`);
}

export function startWorker(): Promise<void> {
  return new Promise((resolve, reject) => {
    log("Starting persistent Octave worker...");
    createFifo();

    const cmd = `addpath('spm12','spm12/toolbox/DEM','psychology'); psychology_worker('${FIFO_PATH}')`;
    worker = spawn("octave", ["--no-gui", "--eval", cmd], {
      cwd: PROJECT_ROOT,
      stdio: ["ignore", "pipe", "pipe"],
    });

    worker.stderr!.on("data", (chunk) => {
      const text = chunk.toString().trim();
      if (text) log(`stderr: ${text.slice(0, 200)}`);
    });

    worker.on("close", (code) => {
      log(`Worker exited with code ${code}`);
      ready = false;
      worker = null;
      try {
        unlinkSync(FIFO_PATH);
      } catch {}
      for (const item of queue) {
        item.reject(new Error("Octave worker exited unexpectedly"));
      }
      queue = [];
    });

    worker.on("error", (err) => {
      log(`Worker error: ${err.message}`);
      reject(err);
    });

    worker.stdout!.on("data", (chunk: Buffer) => {
      const raw = chunk.toString();
      stdoutBuffer += raw;

      const lines = stdoutBuffer.split("\n");
      stdoutBuffer = lines.pop() ?? "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;

        if (trimmed === "__READY__") {
          ready = true;
          log("Worker ready");
          resolve();
        } else if (trimmed === "__DONE__") {
          log("Simulation complete");
          processing = false;
          processQueue();
        } else if (trimmed.startsWith("{")) {
          log(`Received result (${trimmed.length} chars)`);
          const current = queue[0];
          if (current) {
            queue.shift();
            try {
              const result = JSON.parse(trimmed);
              if (result.error) {
                current.reject(new Error(result.error));
              } else {
                current.resolve(result as SimulationResult);
              }
            } catch {
              current.reject(
                new Error(
                  `Bad JSON from Octave: ${trimmed.slice(0, 200)}`,
                ),
              );
            }
          }
        } else {
          log(`stdout: ${trimmed.slice(0, 100)}`);
        }
      }
    });
  });
}

function processQueue() {
  if (processing || queue.length === 0 || !ready || !worker) return;
  processing = true;

  const item = queue[0];
  const json = JSON.stringify(item.params);
  log(`Running simulation (N=${item.params.N})...`);
  // writeFileSync blocks until Octave opens the FIFO for reading,
  // which happens immediately since the worker loops on fopen
  writeFileSync(FIFO_PATH, json + "\n");
}

export function stopWorker() {
  if (worker) {
    log("Killing Octave worker...");
    worker.kill();
    worker = null;
    ready = false;
  }
  try { unlinkSync(FIFO_PATH); } catch {}
}

export function runSimulation(
  params: SimulationParams,
): Promise<SimulationResult> {
  return new Promise((resolve, reject) => {
    if (!worker || !ready) {
      reject(new Error("Octave worker not ready"));
      return;
    }
    queue.push({ params, resolve, reject });
    processQueue();
  });
}
