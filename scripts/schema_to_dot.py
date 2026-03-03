#!/usr/bin/env python3
import json
import hashlib
import argparse
from pathlib import Path


def short_id(s: str, n: int = 10) -> str:
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return h[:n]


def port_label(d):
    # d has: name, class, size
    size = d.get("size", [])
    if isinstance(size, list):
        sz = "x".join(str(int(x)) for x in size) if size else ""
    else:
        sz = str(size)
    # Escape special characters for Graphviz
    name = str(d.get("name","?")).replace("{", "\\{").replace("}", "\\}")
    cls = str(d.get("class","?")).replace("{", "\\{").replace("}", "\\}")
    return f'{name}: {cls} [{sz}]'


def struct_label(title, items):
    if not items:
        return f"{title}|(none)"
    lines = "\\l".join(port_label(d) for d in items) + "\\l"
    return f"{title}|{lines}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("schema_json", type=Path)
    ap.add_argument("--out", type=Path, default=Path("schema.dot"))
    args = ap.parse_args()

    data = json.loads(args.schema_json.read_text(encoding="utf-8"))
    events = data.get("events", [])

    # Build DOT
    dot = []
    dot.append("digraph schema {")
    dot.append('  rankdir=LR;')
    dot.append('  node [shape=record, fontsize=10, fontname="Helvetica"];')
    dot.append('  edge [fontsize=9, fontname="Helvetica"];')

    # For each event, create a transformation node
    for idx, ev in enumerate(events):
        name = ev.get("name", f"event_{idx}")
        ins = ev.get("inputs", [])
        outs = ev.get("outputs", [])

        ev_key = json.dumps(ev, sort_keys=True)
        nid = f"n_{short_id(name + '|' + ev_key)}"

        label = struct_label(name, ins) + "|" + struct_label("returns", outs)
        dot.append(f'  {nid} [label="{label}"];')

    # Add edges based on matching output descriptors to input descriptors (by class+size)
    # This is a heuristic: connect any producer whose output matches a consumer input.
    def sig(d):
        size = d.get("size", [])
        size_t = tuple(size) if isinstance(size, list) else (size,)
        return (d.get("class", ""), size_t)

    nodes = []
    for idx, ev in enumerate(events):
        name = ev.get("name", f"event_{idx}")
        ev_key = json.dumps(ev, sort_keys=True)
        nid = f"n_{short_id(name + '|' + ev_key)}"
        nodes.append((nid, ev))

    # index producers by output signature
    producers = {}
    for nid, ev in nodes:
        for od in ev.get("outputs", []) or []:
            producers.setdefault(sig(od), []).append((nid, ev, od))

    # connect to consumers
    edge_count = 0
    for cnid, cev in nodes:
        for idesc in cev.get("inputs", []) or []:
            s = sig(idesc)
            for pnid, pev, odesc in producers.get(s, []):
                # avoid self-loops unless you want them
                if pnid == cnid:
                    continue
                dot.append(f'  {pnid} -> {cnid} [label="{odesc.get("name","out")}→{idesc.get("name","in")}"];')
                edge_count += 1

    dot.append("}")
    args.out.write_text("\n".join(dot), encoding="utf-8")
    print(f"Wrote {args.out} with {len(events)} nodes and {edge_count} edges.")


if __name__ == "__main__":
    main()
