# Dataflow Graph Generation Instructions

Generated from conversation with GPT-5.2 via PAL MCP server.

## Overview

Created a schema discovery system for MATLAB/Octave code that:
1. Instruments key functions to log data transformations
2. Captures {name, class, size} for inputs/outputs (minimal schema)
3. Deduplicates signatures to create static dataflow diagram
4. Exports to JSON, then converts to Graphviz DOT

## Implementation

### Modified Files

**DEM_morphogenesis.m** - Instrumented with:
- Global trace variables (after `clear global`)
- Schema dump call (before `return`)
- Instrumentation in `morphogenesis`, `observe`, `expect`
- Helper functions for tracing and JSON export

**scripts/schema_to_dot.py** - Converts JSON schema to DOT diagram

### Usage

```bash
# 1. Run simulation to capture schema
octave --eval "DEM_morphogenesis"

# 2. Convert schema to DOT
python3 scripts/schema_to_dot.py dem_morphogenesis_schema.json --out schema.dot

# 3. Render diagram
dot -Tpng schema.dot -o schema.png
# or
dot -Tsvg schema.dot -o schema.svg
```

## Key Design Decisions

**1. Schema Discovery vs Runtime Tracing**
- Chose schema discovery: run once to capture type signatures
- Not per-timestep tracing (would be noisy)
- Deduplication ensures unique transformations only

**2. Instrumentation Points**
- `morphogenesis(position, signal, y) -> sense`
- `observe` with internal `spm_unvec` and `morphogenesis` calls
- `expect` with internal `spm_softmax` transformation
- Captures both high-level functions and key internal steps

**3. Schema Format**
- Minimal: {name, class, size} only
- No sparsity, no numeric ranges
- Structs represented as `struct{field1,field2,...}`

**4. Edge Inference**
- Heuristic: match output {class,size} to input {class,size}
- Connects producers to consumers automatically
- Avoids self-loops

## Implementation Details

### Trace Infrastructure

**Globals:**
```matlab
global TRACE_SCHEMA TRACE_SCHEMA_KEYS
TRACE_SCHEMA      = {};           % Array of event records
TRACE_SCHEMA_KEYS = struct();     % Deduplication registry
```

**Trace Call Pattern:**
```matlab
trace_sig('function_name', ...
    {'input1', value1; 'input2', value2}, ...  % inputs
    {'output1', result});                       % outputs
```

**Helper Functions:**
- `trace_sig()` - Main entry point, deduplicates by signature
- `trace_describe_kv()` - Extract descriptors from values
- `trace_describe_value()` - Get {name, class, size} for one value
- `trace_dump_schema()` - Write JSON to file
- `trace_jsonencode()` - Use jsonencode or fallback
- `trace_json_fallback()` - Minimal JSON encoder for Octave

### Python Schema→DOT Converter

**Node Creation:**
- One node per transformation event
- Record format showing inputs and outputs
- Unique ID from hash of event content

**Edge Creation:**
- Match outputs to inputs by {class, size} signature
- Create edges showing data flow
- Label with variable names

## Alternatives Considered

**Static Analysis (regex/mtree):**
- Pro: No code modification
- Con: Can't capture runtime shapes, complex to implement correctly
- Decision: Rejected for this use case

**Runtime Tracing (all timesteps):**
- Pro: See execution dynamics
- Con: Too much data, not needed for schema discovery
- Decision: Rejected, use deduplication instead

**Refactor to Separate Files:**
- Pro: Easier to wrap functions
- Con: Changes SPM demo structure, path management issues
- Decision: Rejected, in-place instrumentation cleaner

**Shadow spm_* Functions:**
- Pro: No changes to demo file
- Con: Risky (SPM uses these everywhere), hard to maintain
- Decision: Rejected except as future option

## Future Enhancements

**Better Edge Inference:**
- Add optional `event.notes` or `event.tags`
- Explicitly mark "this output is identityBelief"
- More deterministic graph wiring

**Multi-Demo Support:**
- Generalize tracing to work across SPM demos
- Shadow key spm_* functions with opt-in wrappers

**Interactive Visualization:**
- Web-based graph explorer
- Zoom into transformations
- Show sample data shapes

**Test Coverage:**
- Verify schema captures all key transformations
- Regression tests for schema format stability

## Context and Goal

**Original Goal:** Understand active inference mechanics deeply enough to port concepts to:
- Psychological phenomena (first-person energy dynamics)
- Cult dynamics (unifying/self-policing forces)
- Relationship dynamics (mutual investment patterns)
- Startup growth (specialization, alignment as free energy minimization)

**Why Dataflow Matters:**
Seeing concrete data shapes and transformations helps understand:
- How identity beliefs (logits → softmax → identityBelief) drive behavior
- How morphogen fields (position, signal → sense) enable communication
- How predictions (expect) and observations (observe) couple in active inference loop

**Key Insight from GPT-5.2:**
"Active inference never requires the agent to have access to the generative process as an objective, true model. It only requires that the agent can act to make its sensations conform to its own prior preferences."

Understanding the dataflow shows how this works mechanistically: beliefs transform to predictions, actions transform to observations, and prediction errors drive updates.
