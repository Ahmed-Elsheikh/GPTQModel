"""DIAGNOSTIC ONLY (REPORT.md A.7 item 2) - not a fix, not for production runs.

Runtime wrapper (no change to GPTQModel's source): replaces gptqmodel.quantization.gptq.GPTQ.add_batch with a
version that drops sequence position 0 from the calibration activations before they enter the Hessian
H = (2/N) * sum x x^T. Only the Hessian statistics change: the layer forward passes, the propagated hidden states
and the attention at position 0 are untouched (add_batch is an observer hook on each module's input).

Preconditions, checked at runtime: batch_size=1 calibration with equal-length, unpadded windows, so every
non-embedding add_batch call sees inp of shape (1, L, hidden) and inp[:, 0] is the window's first token. Any other
shape raises, instead of silently dropping the wrong rows.

STATS records calls, dropped rows and input shapes; quant_eval_diag.py writes them into the result JSON.
"""
import torch
import torch.nn as nn
from gptqmodel.quantization import gptq as _gptq

EXPECTED_L = None          # set by the runner (calibration window length)
STATS = {"calls": 0, "dropped_rows": 0, "kept_rows": 0, "shapes": {}, "embedding_calls": 0}
_orig_add_batch = _gptq.GPTQ.add_batch


def add_batch_excluding_pos0(self, inp, out, batch_index=None):
    if isinstance(self.module, nn.Embedding):
        STATS["embedding_calls"] += 1
        return _orig_add_batch(self, inp, out, batch_index)
    shape = tuple(inp.shape)
    STATS["shapes"][str(shape)] = STATS["shapes"].get(str(shape), 0) + 1
    if inp.dim() != 3 or inp.shape[0] != 1 or (EXPECTED_L is not None and inp.shape[1] != EXPECTED_L):
        raise RuntimeError(f"gptq_exclude_pos0: unexpected calibration input shape {shape} (expected (1, {EXPECTED_L}, h))")
    STATS["calls"] += 1
    STATS["dropped_rows"] += 1
    STATS["kept_rows"] += inp.shape[1] - 1
    return _orig_add_batch(self, inp[:, 1:, :], out, batch_index)


def install(expected_L):
    global EXPECTED_L
    EXPECTED_L = expected_L
    _gptq.GPTQ.add_batch = add_batch_excluding_pos0
