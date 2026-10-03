"""A.7 item 2 runner: quant_eval.py with patches/diagnostic/gptq_exclude_pos0.py installed (position 0 excluded
from every GPTQ Hessian). Same CLI as quant_eval.py; the hook statistics are added to the --out JSON."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "patches", "diagnostic"), HERE]
import gptq_exclude_pos0 as hook
L = int(sys.argv[sys.argv.index("--L") + 1])
hook.install(L)
import quant_eval
out = sys.argv[sys.argv.index("--out") + 1]
try:
    quant_eval.main()
finally:
    if os.path.exists(out):
        r = json.load(open(out))
        r["diagnostic_hook"] = {"name": "gptq_exclude_pos0", **hook.STATS}
        json.dump(r, open(out, "w"), indent=2)
    print("[diag hook]", json.dumps({k: v for k, v in hook.STATS.items() if k != "shapes"}), hook.STATS["shapes"], flush=True)
