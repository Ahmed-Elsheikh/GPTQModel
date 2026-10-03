"""Compare determinism-probe runs: ppl (repr), checkpoint sha256, GPTQ per-module losses (layer 0 o_proj + first divergence)."""
import glob, hashlib, json, os, re, sys
OUT = os.environ.get("STEP2_OUT", "/tmp/step2_out")
def losses(log):
    return re.findall(r"'layer': (\d+), 'module': '([a-z_.]+)'.*?'loss': '([0-9.e-]+)'", open(log, errors="replace").read())
def ckhash(d):
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))): h.update(open(f, "rb").read())
    return h.hexdigest()[:16] if glob.glob(os.path.join(d, "*.safetensors")) else None
runs = [("step1 (Session B)", "logs/real_gptq_w4_135m_s0.log", None, "results/real_step1.jsonl", "gptq_w4_135m_s0"),
        ("rebuild1", "logs/real_s2_gptq_w4_135m_wt_s0_rebuild.log", f"{OUT}/ckpt/s2_gptq_w4_135m_wt_s0_rebuild",
         "results/real_step2_c4val.jsonl", "s2_gptq_w4_135m_wt_s0_rebuild")]
for l in open("results/real_step2_det.jsonl"):
    r = json.loads(l); runs.append((r["run_id"], r["log"], f"{OUT}/ckpt/{r['run_id']}", "results/real_step2_det.jsonl", r["run_id"]))
ref = {n: losses(lg) for n, lg, *_ in runs}
base_s1, base_r1 = ref["step1 (Session B)"], ref["rebuild1"]
def first_div(a, b):
    return next(((x[0], x[1], x[2], y[2]) for x, y in zip(a, b) if x != y), None)
rows = []
for name, log, ck, jl, rid in runs:
    rec = next(json.loads(l) for l in open(jl) if json.loads(l)["run_id"] == rid)
    x = rec.get("result") or {}
    L = ref[name]; op = next((v for la, m, v in L if la == "0" and m == "self_attn.o_proj"), None)
    det = x.get("det", {})
    row = dict(run=name, rc=rec["rc"], ppl=repr(x.get("ppl_quant")), ckpt_sha=ckhash(ck) if ck else None, n_losses=len(L),
               l0_o_proj_loss=op, first_div_vs_step1=first_div(L, base_s1), first_div_vs_rebuild1=first_div(L, base_r1),
               n_diff_vs_rebuild1=sum(a != b for a, b in zip(L, base_r1)), setting=det.get("setting"),
               attn=det.get("attn_impl_at_load"), threads=x.get("args", {}).get("threads"),
               quantize_s=x.get("phases", {}).get("quantize", {}).get("wall_s"), wall_s=rec["wall_s"])
    rows.append(row); print(json.dumps(row))
json.dump(rows, open(sys.argv[1] if len(sys.argv) > 1 else "/dev/null", "w"), indent=1)
