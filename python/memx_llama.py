"""Real Llama-architecture forward + greedy generation with MemX-hosted weights.

Bit-exact contract: every weight byte round-trips through MemX compression
unchanged, so the managed forward must reproduce a plain-torch reference
token-for-token (greedy argmax) and logits within bf16 rounding.

This is NOT a synthetic digest harness. It implements RMSNorm, RoPE, grouped
-query attention with a real KV cache, SiLU gated MLP, and a token loop.
"""
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def load_config(model_dir):
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    return cfg


def load_state_dict(model_dir):
    from safetensors import safe_open
    model_dir = Path(model_dir)
    idx_file = model_dir / "model.safetensors.index.json"
    tensors = {}
    if idx_file.exists():
        weight_map = json.loads(idx_file.read_text())["weight_map"]
        files = sorted(set(weight_map.values()))
        for fn in files:
            with safe_open(str(model_dir / fn), framework="pt") as f:
                for k in f.keys():
                    tensors[k] = f.get_tensor(k)
    else:
        with safe_open(str(model_dir / "model.safetensors"), framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
    return tensors


class LlamaWeights:
    """Holds per-layer weight tensors. Source can be plain dict or MemX views."""

    def __init__(self, cfg, get_tensor):
        self.cfg = cfg
        self.hidden = cfg["hidden_size"]
        self.layers = cfg["num_hidden_layers"]
        self.heads = cfg["num_attention_heads"]
        self.kv_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg.get("head_dim", self.hidden // self.heads)
        self.inter = cfg["intermediate_size"]
        self.eps = cfg["rms_norm_eps"]
        self.theta = cfg["rope_theta"]
        self.vocab = cfg["vocab_size"]
        self.group = self.heads // self.kv_heads

        self.embed = get_tensor("model.embed_tokens.weight")
        self.norm = get_tensor("model.norm.weight")
        self.lm_head = get_tensor("lm_head.weight")
        self.L = []
        for i in range(self.layers):
            p = f"model.layers.{i}."
            self.L.append({
                "ln1": get_tensor(p + "input_layernorm.weight"),
                "q": get_tensor(p + "self_attn.q_proj.weight"),
                "k": get_tensor(p + "self_attn.k_proj.weight"),
                "v": get_tensor(p + "self_attn.v_proj.weight"),
                "o": get_tensor(p + "self_attn.o_proj.weight"),
                "ln2": get_tensor(p + "post_attention_layernorm.weight"),
                "gate": get_tensor(p + "mlp.gate_proj.weight"),
                "up": get_tensor(p + "mlp.up_proj.weight"),
                "down": get_tensor(p + "mlp.down_proj.weight"),
            })

    def rope_cache(self, seq_len, device, dtype):
        hd = self.head_dim
        inv = 1.0 / (self.theta ** (torch.arange(0, hd, 2, device=device).float() / hd))
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, inv)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        return cos, sin


def rms_norm(x, w, eps):
    dtype = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    return xf.to(dtype) * w


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin, pos):
    c = cos[pos].unsqueeze(0).unsqueeze(0)
    s = sin[pos].unsqueeze(0).unsqueeze(0)
    q = q * c + rotate_half(q) * s
    k = k * c + rotate_half(k) * s
    return q, k


class KVCache:
    def __init__(self, layers, kv_heads, head_dim, max_len, dtype, device):
        self.k = [torch.zeros(1, kv_heads, max_len, head_dim, dtype=dtype, device=device) for _ in range(layers)]
        self.v = [torch.zeros(1, kv_heads, max_len, head_dim, dtype=dtype, device=device) for _ in range(layers)]
        self.len = 0

    def append(self, layer, k, v):
        n = k.shape[2]
        self.k[layer][:, :, self.len:self.len + n] = k
        self.v[layer][:, :, self.len:self.len + n] = v

    def get(self, layer):
        return self.k[layer][:, :, :self.len], self.v[layer][:, :, :self.len]


def forward_layer(w, i, x, cache, cos, sin, pos):
    L = w.L[i]
    b, n, _ = x.shape
    h = rms_norm(x, L["ln1"], w.eps)
    q = F.linear(h, L["q"]).view(b, n, w.heads, w.head_dim).transpose(1, 2)
    k = F.linear(h, L["k"]).view(b, n, w.kv_heads, w.head_dim).transpose(1, 2)
    v = F.linear(h, L["v"]).view(b, n, w.kv_heads, w.head_dim).transpose(1, 2)
    q, k = apply_rope(q, k, cos, sin, pos)
    cache.append(i, k, v)
    kk, vv = cache.get(i)
    if w.group > 1:
        kk = kk.repeat_interleave(w.group, dim=1)
        vv = vv.repeat_interleave(w.group, dim=1)
    scale = 1.0 / math.sqrt(w.head_dim)
    scores = torch.matmul(q, kk.transpose(-1, -2)) * scale
    total = kk.shape[2]
    mask = torch.full((n, total), float("-inf"), device=x.device, dtype=scores.dtype)
    mask = torch.triu(mask, diagonal=total - n + 1)
    scores = scores + mask
    attn = F.softmax(scores.float(), dim=-1).to(q.dtype)
    out = torch.matmul(attn, vv)
    out = out.transpose(1, 2).contiguous().view(b, n, w.heads * w.head_dim)
    x = x + F.linear(out, L["o"])
    h2 = rms_norm(x, L["ln2"], w.eps)
    x = x + F.linear(F.silu(F.linear(h2, L["gate"])) * F.linear(h2, L["up"]), L["down"])
    return x


def forward(w, ids, cache, cos, sin, start_pos):
    b, n = ids.shape
    reset = getattr(w, "reset_stream", None)
    if reset is not None:
        reset()
    x = F.embedding(ids, w.embed)
    pos = list(range(start_pos, start_pos + n))
    base = cache.len
    for i in range(w.layers):
        cache.len = base
        x = forward_layer(w, i, x, cache, cos, sin, pos)
    cache.len = base + n
    x = rms_norm(x, w.norm, w.eps)
    logits = F.linear(x, w.lm_head)
    return logits


@torch.no_grad()
def generate(w, prompt_ids, max_new_tokens, device="cpu", dtype=torch.bfloat16, max_len=None):
    if max_len is None:
        max_len = prompt_ids.shape[1] + max_new_tokens + 8
    cache = KVCache(w.layers, w.kv_heads, w.head_dim, max_len, dtype, device)
    cos, sin = w.rope_cache(max_len, device, dtype)
    ids = prompt_ids.to(device)
    logits = forward(w, ids, cache, cos, sin, 0)
    out = []
    nxt = logits[0, -1].argmax().item()
    out.append(nxt)
    for _ in range(max_new_tokens - 1):
        cur = torch.tensor([[nxt]], device=device)
        logits = forward(w, cur, cache, cos, sin, cache.len)
        nxt = logits[0, -1].argmax().item()
        out.append(nxt)
    return out


LAYER_KEYS = ("ln1", "q", "k", "v", "o", "ln2", "gate", "up", "down")
LAYER_NAME = {
    "ln1": "input_layernorm.weight",
    "q": "self_attn.q_proj.weight",
    "k": "self_attn.k_proj.weight",
    "v": "self_attn.v_proj.weight",
    "o": "self_attn.o_proj.weight",
    "ln2": "post_attention_layernorm.weight",
    "gate": "mlp.gate_proj.weight",
    "up": "mlp.up_proj.weight",
    "down": "mlp.down_proj.weight",
}


class StreamedWeights(LlamaWeights):
    """Layer-streaming weights: only a small working set lives on the compute
    device; each layer is materialized from MemX on demand, with the next
    layer prefetched into a second slot (double buffering)."""

    def __init__(self, cfg, ctx, allocs, meta, device="mps", page=16384):
        self.cfg = cfg
        self.hidden = cfg["hidden_size"]
        self.layers = cfg["num_hidden_layers"]
        self.heads = cfg["num_attention_heads"]
        self.kv_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg.get("head_dim", self.hidden // self.heads)
        self.inter = cfg["intermediate_size"]
        self.eps = cfg["rms_norm_eps"]
        self.theta = cfg["rope_theta"]
        self.vocab = cfg["vocab_size"]
        self.group = self.heads // self.kv_heads
        self.ctx = ctx
        self.allocs = allocs
        self.meta = meta
        self.device = device
        self.page = page
        self._slots = [{}, {}]
        self._slot_layer = [-1, -1]
        self._next = 0
        self.mat_bytes = 0
        self.mat_calls = 0
        self.embed = self._materialize("model.embed_tokens.weight")
        self.norm = self._materialize("model.norm.weight")
        self.lm_head = self._materialize("lm_head.weight")
        self._load_layer(0, 0)
        if self.layers > 1:
            self._load_layer(1, 1)

    @property
    def L(self):
        return self

    def _align(self, n):
        return ((n + self.page - 1) // self.page) * self.page

    def _materialize(self, name):
        a = self.allocs[name]
        shape, dtype, nb = self.meta[name]
        sz = self._align(nb)
        buf = torch.empty(sz, dtype=torch.uint8)
        self.ctx.materialize_range(a, 0, sz, buf.data_ptr(), sz)
        self.mat_bytes += sz
        self.mat_calls += 1
        return buf[:nb].view(dtype).view(shape).to(self.device)

    def _load_layer(self, i, slot):
        d = {}
        p = f"model.layers.{i}."
        for key in LAYER_KEYS:
            d[key] = self._materialize(p + LAYER_NAME[key])
        self._slots[slot] = d
        self._slot_layer[slot] = i

    def reset_stream(self):
        self._next = 0

    def __getitem__(self, key):
        i = self._next
        if i >= self.layers:
            raise IndexError(f"stream pass exhausted at layer {i}; call reset_stream()")
        slot = 0 if self._slot_layer[0] == i else 1
        if self._slot_layer[slot] != i:
            self._load_layer(i, slot)
        nxt = i + 1
        if nxt < self.layers:
            other = 1 - slot
            if self._slot_layer[other] != nxt:
                self._load_layer(nxt, other)
        self._next = nxt
        return self._slots[slot]