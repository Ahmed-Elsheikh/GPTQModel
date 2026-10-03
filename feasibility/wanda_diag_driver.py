"""Diagnosis step 4: run wanda_real_driver.py unchanged, but with Wanda's calibration window length set by
WANDA_SEQLEN (the patched main.get_llm fixes model.seqlen = min(512, max_pos)). Only model.seqlen is changed:
Wanda uses it for the calibration windows and the 128 x seqlen x hidden activation buffer; perplexity is the
driver's Step 1 protocol (40 x 2048 WikiText-2 test windows) regardless of seqlen.
Token ids come from calib.py's cache (filled by the main venv), so windows match the GPTQ runs exactly.
usage: WANDA_SEQLEN=2048 python wanda_diag_driver.py OUT.json <wanda main.py args...>"""
import json, os, runpy, sys
HERE = os.path.dirname(os.path.abspath(__file__))
WANDA = os.path.join(os.path.dirname(HERE), "third_party", "wanda")
sys.path[:0] = [WANDA, HERE]
os.chdir(WANDA)
import main as wanda_main  # the same module object the driver will import

L = int(os.environ.get("WANDA_SEQLEN", "512"))
_get_llm = wanda_main.get_llm
def get_llm(*a, **k):
    m = _get_llm(*a, **k)
    m.seqlen = L
    return m
wanda_main.get_llm = get_llm

out = sys.argv[1]
sys.argv = [os.path.join(HERE, "wanda_real_driver.py")] + sys.argv[1:]
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    if os.path.exists(out):
        import calib
        r = json.load(open(out)); r["wanda_seqlen"] = L; r["token_cache"] = calib.cache_info()
        json.dump(r, open(out, "w"), indent=2)
