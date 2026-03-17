import { spawn, type ChildProcess } from "child_process";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import type { SimulationParams, SimulationResult } from "./types.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = join(__dirname, "..", "..", "..");

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

export function startWorker(): Promise<void> {
  return new Promise((resolve, reject) => {
    log("Starting persistent Octave worker...");

    const evalCmd =
      "addpath('spm12','spm12/toolbox/DEM','psychology'); psychology_worker()";
    worker = spawn("octave", ["--no-gui", "--eval", evalCmd], {
      cwd: PROJECT_ROOT,
      stdio: ["pipe", "pipe", "pipe"],
    });

    worker.stderr!.on("data", (chunk) => {
      // Octave prints warnings to stderr — log but don't crash
      const text = chunk.toString().trim();
      if (text) log(`stderr: ${text}`);
    });

    worker.on("close", (code) => {
      log(`Worker exited with code ${code}`);
      ready = false;
      worker = null;
      // reject any pending requests
      for (const item of queue) {
        item.reject(new Error("Octave worker exited unexpectedly"));
      }
      queue = [];
    });

    worker.on("error", (err) => {
      log(`Worker error: ${err.message}`);
      reject(err);
    });

    // wait for __READY__ signal
    const onData = (chunk: Buffer) => {
      stdoutBuffer += chunk.toString();
      const lines = stdoutBuffer.split("\n");
      stdoutBuffer = lines.pop() ?? "";

      for (const line of lines) {
        if (line.trim() === "__READY__") {
          ready = true;
          log("Worker ready");
          resolve();
          return;
        }
      }
    };
    worker.stdout!.on("data", onData);

    // after ready, switch to the request handler
    const switchToRequestHandler = () => {
      worker!.stdout!.removeListener("data", onData);
      worker!.stdout!.on("data", handleStdout);
    };

    // resolve switches handler
    const origResolve = resolve;
    resolve = () => {
      switchToRequestHandler();
      origResolve();
    };
  });
}

function handleStdout(chunk: Buffer) {
  stdoutBuffer += chunk.toString();
  const lines = stdoutBuffer.split("\n");
  stdoutBuffer = lines.pop() ?? "";

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed === "__DONE__") {
      processing = false;
      processQueue();
    } else if (trimmed.startsWith("{")) {
      // JSON result
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
        } catch (e) {
          current.reject(new Error(`Bad JSON from Octave: ${trimmed.slice(0, 200)}`));
        }
      }
    }
  }
}

function processQueue() {
  if (processing || queue.length === 0 || !ready || !worker) return;
  processing = true;

  const item = queue[0];
  const json = JSON.stringify(item.params);
  log(`Sending params (N=${item.params.N})...`);
  worker.stdin!.write(json + "\n");
}

export function runSimulation(params: SimulationParams): Promise<SimulationResult> {
  return new Promise((resolve, reject) => {
    if (!worker || !ready) {
      reject(new Error("Octave worker not ready"));
      return;
    }
    log("Queued simulation request");
    queue.push({ params, resolve, reject });
    processQueue();
  });
}
