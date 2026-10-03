"""Diagnosis probe (no quantization): squared L2 norm of the mlp.down_proj INPUT per token position in the dense
SmolLM2-135M (fp32), on the first 8 calibration windows of seed 0 at L=512 and at L=2048. GPTQ's Hessian is the
mean of x x^T over calibration tokens, so a token's weight in H is proportional to ||x||^2 / N. Reports the share
of the Hessian trace contributed by position 0 and positions 0-3 for each window length."""
import json, sys, torch, calib
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
M = "HuggingFaceTB/SmolLM2-135M"
tok = AutoTokenizer.from_pretrained(M); model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32).eval()
LAYERS = [2, 11, 20, 28, 29]
cap = {}
def hook(i):
    def f(mod, inp): cap[i] = inp[0].detach().pow(2).sum(-1)[0]  # (L,)
    return f
for i in LAYERS: model.model.layers[i].mlp.down_proj.register_forward_pre_hook(hook(i))
out = {}
for L in (512, 2048):
    wins = calib.draw_windows("wikitext2", tok, n=8, L=L, seed=0)
    acc = {i: torch.zeros(L) for i in LAYERS}
    with torch.no_grad():
        for w in wins:
            model(input_ids=torch.tensor([w["input_ids"]]))
            for i in LAYERS: acc[i] += cap[i]
    out[L] = {}
    for i in LAYERS:
        a = acc[i]; tot = a.sum().item()
        out[L][i] = {"share_pos0_pct": 100 * a[0].item() / tot, "share_pos0_3_pct": 100 * a[:4].sum().item() / tot,
                     "pos0_over_median": a[0].item() / a[1:].median().item(), "max_pos": int(a.argmax())}
print(json.dumps(out, indent=1))
json.dump(out, open("results/real_diag_probe_positions.json", "w"), indent=2)
