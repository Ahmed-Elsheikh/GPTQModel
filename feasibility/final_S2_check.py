"""Bit-identity check for S2 determinism repeats.
usage: final_S2_check.py JSONL RUN_ID_REGEX N  -> exit 0 if N successful runs exist and all have the same ppl
(exact float repr) AND the same checkpoint sha256; exit 1 otherwise. Prints a one-line verdict."""
import json, re, sys
f, pat, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = [json.loads(l) for l in open(f)] if __import__("os").path.exists(f) else []
rs = [r for r in rows if re.fullmatch(pat, r["run_id"])]
ok = [r for r in rs if r["rc"] == 0 and (r.get("result") or {}).get("ppl_quant") is not None]
ppl = sorted({repr(r["result"]["ppl_quant"]) for r in ok})
sha = sorted({(r["result"].get("config") or {}).get("ckpt_sha256") for r in ok})
same = len(ok) >= n and len(ppl) == 1 and len(sha) == 1 and sha[0] is not None
print(f"[check] {pat}: runs={len(rs)} ok={len(ok)} distinct_ppl={ppl} distinct_sha={[s[:16] if s else s for s in sha]} -> {'IDENTICAL' if same else 'NOT IDENTICAL'}", flush=True)
sys.exit(0 if same else 1)
