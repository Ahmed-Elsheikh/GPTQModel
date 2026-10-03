"""Step 4 of the S2 task: add a determinism caveat paragraph at the end of REPORT.md sections A.3, A.5 and A.6.
Idempotent (marker '<!-- det-caveat -->'). Reads the confirmed configuration from REPORT_final_S2.md."""
import os, re
H = os.path.dirname(os.path.abspath(__file__)); p = os.path.join(H, "REPORT.md"); s = open(p).read()
MARK = "<!-- det-caveat -->"
BASE = ("**Determinism caveat (added later; see REPORT_step2.md §8 and REPORT_final_S2.md \"Determinism (confirmed)\").** "
        "The runs in this section used GPTQModel's default CPU path (SDPA attention, 4 threads), which is not run-to-run "
        "deterministic: the identical SmolLM2 W4 seed-0 command gave 17.744843 and 17.811016 in repeats, a gap of 0.066 ppl. ")
TEXT = {
 "A.3": BASE + "The seed SDs above therefore include run-to-run noise (≥ 0.066 between repeats at W4, larger than the W4 five-seed "
        "range of 0.054), so the W4 seed SD cannot be read as pure calibration-seed sensitivity; it is an upper bound on "
        "both together. Run-to-run noise at W3 was not measured on the default path; the W3 seed SD (0.72) is about 11× the "
        "W4 repeat gap. The conclusion that the seed effect is negligible at W4 and small at W3 relative to the dense-to-W4 "
        "and W4-to-W3 gaps stands, because those gaps (3.26 and 20.2 ppl) are far larger than the noise.",
 "A.5": BASE + "Both arms' seed SDs include this run-to-run noise. The window-length effect itself (+5.15 ppl at W3, every 2048-token "
        "run above every 512-token run) is about 78× the 0.066 repeat gap and stands.",
 "A.6": BASE + "The SDs and the per-run values in this section include that noise. The diagnostic effects are far larger than it: "
        "W3 128×2048 vs 128×512 +5.15, 512×512 vs 128×2048 −5.17, W4 2048 vs 512 +1.15 ppl. They stand, and so does the "
        "conclusion that the effect follows window length, not token count. Wanda runs a separate code path, and its "
        "results were bit-identical across containers. The single-seed Qwen2.5-0.5B contrast (+0.47) is within the scale "
        "where this noise and seed variance matter; it is superseded by the 5 + 3-seed study with the confirmed "
        "deterministic configuration in REPORT_final_S2.md.",
}
heads = [m.start() for m in re.finditer(r"^(## |# )", s, re.M)]
for sec in ("A.6", "A.5", "A.3"):  # back to front so offsets stay valid
    m = re.search(rf"^## {re.escape(sec)} .*$", s, re.M)
    assert m, sec
    nxt = min([h for h in [x.start() for x in re.finditer(r"^(## |# )", s, re.M)] if h > m.start()], default=len(s))
    body = s[m.start():nxt]
    if MARK in body:
        continue
    s = s[:nxt].rstrip("\n") + "\n\n" + MARK + "\n" + TEXT[sec] + "\n\n" + s[nxt:]
open(p, "w").write(s)
print("caveats present:", s.count(MARK))
