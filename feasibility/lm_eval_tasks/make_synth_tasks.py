"""OFFLINE stand-ins for arc_easy / hellaswag / wikitext (HF Hub blocked).
Same output_type, prompt template, #choices and approximate token lengths as the real tasks; the
synthetic WordLevel tokenizer maps each 'tN' word to one token. Accuracy numbers are meaningless;
these exist to time the harness and to check that EXP3 factors are plumbed through flags."""
import json, os, random
D = os.path.dirname(os.path.abspath(__file__))
r = random.Random(0)
def words(n, vocab=49152): return " ".join(f"t{r.randrange(2, vocab)}" for _ in range(n))
def dump(name, rows):
    with open(os.path.join(D, name), "w") as f:
        for x in rows: f.write(json.dumps(x) + "\n")
# arc_easy: test 2376, train 2251; question ~25 tok, 4 choices ~5 tok
arc = lambda: {"question": words(25), "choices": {"text": [words(r.randint(2, 8)) for _ in range(4)], "label": list("ABCD")}, "answerKey": r.choice("ABCD")}
dump("arc_train.jsonl", [arc() for _ in range(600)]); dump("arc_test.jsonl", [arc() for _ in range(600)])
# hellaswag: validation 10042; ctx ~ 60 tok, 4 endings ~ 25 tok
hs = lambda: {"query": words(60), "choices": [words(r.randint(15, 35)) for _ in range(4)], "label": r.randrange(4)}
dump("hs_train.jsonl", [hs() for _ in range(600)]); dump("hs_val.jsonl", [hs() for _ in range(600)])
# wikitext-2 test: 62 documents, ~ 280k tokens (SmolLM2 tokenizer); ~4.5k tok/doc
dump("wt_test.jsonl", [{"page": words(r.randint(2500, 6500))} for _ in range(62)])
