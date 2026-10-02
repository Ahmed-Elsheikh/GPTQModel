"""Run locuslab/wanda main.py unmodified (except patches/wanda_cpu.patch) with its HF-Hub data loaders
replaced IN THE DRIVER by offline synthetic token streams (huggingface.co is blocked in this VM).
Calibration: nsamples windows of model.seqlen drawn with random.Random(seed) (wanda's own c4 loader
also uses random.seed(seed)). Eval: first EVAL_WINDOWS non-overlapping windows of the synthetic test stream.
usage: python wanda_synth_driver.py <wanda main.py args...>"""
import os, random, sys, time, resource
HERE = os.path.dirname(os.path.abspath(__file__))
WANDA = os.path.join(os.path.dirname(HERE), "third_party", "wanda")
sys.path[:0] = [WANDA, HERE]
os.chdir(WANDA)
import torch
torch.set_num_threads(4)
import calib
import lib.data, lib.prune, lib.eval
EVAL_WINDOWS = int(os.environ.get("WANDA_EVAL_WINDOWS", "40"))

class _Enc:
    def __init__(self, ids): self.input_ids = ids

def synth_get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    vocab = len(tokenizer)
    rng = random.Random(seed)
    train = calib._synthetic_stream(vocab, calib.SYN_TRAIN_SEED)
    loader = []
    for _ in range(nsamples):
        s = rng.randint(0, len(train) - seqlen - 1)
        inp = torch.tensor([train[s:s + seqlen]]); tar = inp.clone(); tar[:, :-1] = -100
        loader.append((inp, tar))
    test = calib._synthetic_stream(vocab, calib.SYN_TEST_SEED)[: EVAL_WINDOWS * seqlen]
    print(f"[driver] synthetic loader for '{name}': {nsamples}x{seqlen} calib, {len(test)} eval tokens", flush=True)
    return loader, _Enc(torch.tensor([test]))

lib.data.get_loaders = lib.prune.get_loaders = lib.eval.get_loaders = synth_get_loaders
import main as wanda_main
t = time.time()
sys.argv = ["main.py"] + sys.argv[1:]
try:
    wanda_main.main()
finally:
    print(f"[driver] wall_s={time.time()-t:.1f} peak_rss_gb={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.2f}", flush=True)
