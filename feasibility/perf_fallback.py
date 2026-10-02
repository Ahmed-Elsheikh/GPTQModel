"""Test 3 fallback / cross-check: plain torch.perf_counter latency of prefill + greedy decode."""
import sys, time, json, statistics, torch
from transformers import AutoModelForCausalLM
torch.set_num_threads(4); torch.manual_seed(0)
path, bs, seq, new, warm, iters = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
ids = torch.randint(2, m.config.vocab_size, (bs, seq))
gen = lambda n: m.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=n, min_new_tokens=n, do_sample=False, pad_token_id=0)
with torch.inference_mode():
    for _ in range(warm): gen(2)
    pre, full = [], []
    for _ in range(iters):
        t = time.perf_counter(); gen(1); pre.append(time.perf_counter() - t)
        t = time.perf_counter(); gen(new); full.append(time.perf_counter() - t)
dec = [(f - p) / (new - 1) for f, p in zip(full, pre)]
print(json.dumps({"model": path.split("/")[-1], "bs": bs, "seq": seq, "warmup": warm, "prefill_s_median": statistics.median(pre),
                  "decode_s_per_tok_median": statistics.median(dec), "prefill_all": pre, "torch": torch.__version__}))
