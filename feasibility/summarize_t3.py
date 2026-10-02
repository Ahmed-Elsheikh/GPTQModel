"""Test 3c: 360M/135M latency ratio per benchmarking protocol from the Hydra multirun reports."""
import glob, json, os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "t3b")
rows = {}
for f in glob.glob(os.path.join(R, "**", "benchmark.json"), recursive=True):
    b = json.load(open(f)); c, r = b["config"], b["report"]
    model = os.path.basename(c["backend"]["model"])
    sc = c["scenario"]; key = (sc["warmup_runs"], sc["input_shapes"]["batch_size"], sc["input_shapes"]["sequence_length"])
    lat = lambda k, s="p50": (r.get(k) or {}).get("latency", {}) and r[k]["latency"][s]
    rows.setdefault(key, {})[model] = {"prefill_p50": lat("prefill"), "per_token_p50": lat("per_token"),
                                       "generate_mean": lat("generate", "mean"), "first_generate": lat("first_generate", "mean")}
out = []
for key in sorted(rows):
    m = rows[key]
    if len(m) < 2: continue
    a, b = m["synth-SmolLM2-135M"], m["synth-SmolLM2-360M"]
    out.append({"warmup": key[0], "bs": key[1], "seq": key[2], **{f"ratio_{k}": round(b[k] / a[k], 3) for k in a if a[k]},
                "135M": a, "360M": b})
json.dump(out, open(os.path.join(R, "..", "t3c_ratios.json"), "w"), indent=2)
for k in ("ratio_prefill_p50", "ratio_per_token_p50", "ratio_generate_mean", "ratio_first_generate"):
    v = [o[k] for o in out if k in o]
    if v: print(f"{k}: min={min(v)} max={max(v)} n={len(v)}")
for o in out:
    print(o["warmup"], o["bs"], o["seq"], {k: v for k, v in o.items() if k.startswith("ratio")})
