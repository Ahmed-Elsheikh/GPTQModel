"""Build randomly-initialised stand-ins for the target models (HF Hub is blocked in this VM).

Architectures are reconstructed FROM MEMORY of the public config.json files and are
NOT verified against the Hub. Weights are random (seeded), so perplexities are
meaningless as quality numbers; these models exist only to measure wall time, memory,
determinism and API compatibility. A trivial WordLevel tokenizer is attached so that
tools that insist on a tokenizer can load the directory.
"""
import json, os, sys, torch
from transformers import LlamaConfig, Qwen2Config, AutoModelForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
SPECS = {
    "synth-SmolLM2-135M": (LlamaConfig, dict(hidden_size=576, intermediate_size=1536, num_hidden_layers=30,
        num_attention_heads=9, num_key_value_heads=3, vocab_size=49152, max_position_embeddings=8192,
        rope_theta=100000.0, rms_norm_eps=1e-5, tie_word_embeddings=True, bos_token_id=0, eos_token_id=0)),
    "synth-SmolLM2-360M": (LlamaConfig, dict(hidden_size=960, intermediate_size=2560, num_hidden_layers=32,
        num_attention_heads=15, num_key_value_heads=5, vocab_size=49152, max_position_embeddings=8192,
        rope_theta=100000.0, rms_norm_eps=1e-5, tie_word_embeddings=True, bos_token_id=0, eos_token_id=0)),
    "synth-Qwen2.5-0.5B": (Qwen2Config, dict(hidden_size=896, intermediate_size=4864, num_hidden_layers=24,
        num_attention_heads=14, num_key_value_heads=2, vocab_size=151936, max_position_embeddings=32768,
        rope_theta=1000000.0, rms_norm_eps=1e-6, tie_word_embeddings=True, bos_token_id=151643, eos_token_id=151643)),
}

def make_tokenizer(vocab_size):
    vocab = {f"t{i}": i for i in range(vocab_size)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="t1"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="t1", eos_token="t0", bos_token="t0", pad_token="t0")

def main(names):
    torch.manual_seed(0)
    for name in names:
        path = os.path.join(OUT, name)
        if os.path.exists(os.path.join(path, "config.json")):
            print("exists", path); continue
        cls, kw = SPECS[name]
        cfg = cls(**kw, torch_dtype="bfloat16")
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
        model.save_pretrained(path)
        make_tokenizer(kw["vocab_size"]).save_pretrained(path)
        n = sum(p.numel() for p in model.parameters())
        print(name, f"{n/1e6:.1f}M params ->", path)

if __name__ == "__main__":
    main(sys.argv[1:] or list(SPECS))
