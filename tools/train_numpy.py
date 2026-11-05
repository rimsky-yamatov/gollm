#!/usr/bin/env python3
"""gollm の参照実装 (numpy)。Go 実装と同一のアーキテクチャ・重みフォーマット。

  python3 tools/train_numpy.py train  [steps]
  python3 tools/train_numpy.py chat
  python3 tools/train_numpy.py golden

Go のツールチェーンが無い環境でも学習・動作確認できるようにするためのもの。
学習済み重みは model/gollm.bin (Go の `gollm chat` がそのまま読める)。
"""
import glob, json, math, struct, sys, time, os
import numpy as np

D, L, H, FF, T = 128, 3, 4, 512, 128
HD = D // H
PAD, EOS, UNK = 0, 1, 2
BOT = "ノックス: "
MAGIC = b"GOLLM001"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BIN = os.path.join(ROOT, "model", "gollm.bin")
VOCAB = os.path.join(ROOT, "model", "vocab.json")


# ---------------- data ----------------
def load_convs():
    convs = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "corpus", "*.txt"))):
        txt = open(f, encoding="utf-8").read()
        for c in txt.split("\n\n"):
            c = c.strip("\n")
            if c:
                convs.append(c + "\n")
    return convs


def build_vocab(convs):
    chars = sorted(set("".join(convs)))
    return ["<pad>", "<eos>", "<unk>"] + chars


def encode(s, c2i):
    return [c2i.get(ch, UNK) for ch in s]


def build_stream(convs, c2i):
    ids, mask, starts = [], [], []
    for c in convs:
        starts.append(len(ids))
        for line in c.split("\n")[:-1]:
            line += "\n"
            m = 1.0 if line.startswith("ノックス:") else 0.0
            for ch in line:
                ids.append(c2i.get(ch, UNK))
                mask.append(m)
        ids.append(EOS)
        mask.append(0.0)
    return np.array(ids, dtype=np.int32), np.array(mask, dtype=np.float32), np.array(starts)


# ---------------- model ----------------
def init_params(V, rng):
    s = 0.02
    p = {"E": rng.normal(0, s, (V, D)), "P": rng.normal(0, s, (T, D)),
         "lnfg": np.ones(D), "lnfb": np.zeros(D)}
    for l in range(L):
        p[f"{l}.ln1g"] = np.ones(D); p[f"{l}.ln1b"] = np.zeros(D)
        p[f"{l}.Wq"] = rng.normal(0, s, (D, D)); p[f"{l}.Wk"] = rng.normal(0, s, (D, D))
        p[f"{l}.Wv"] = rng.normal(0, s, (D, D))
        p[f"{l}.Wo"] = rng.normal(0, s / math.sqrt(2 * L), (D, D))
        p[f"{l}.ln2g"] = np.ones(D); p[f"{l}.ln2b"] = np.zeros(D)
        p[f"{l}.W1"] = rng.normal(0, s, (D, FF)); p[f"{l}.b1"] = np.zeros(FF)
        p[f"{l}.W2"] = rng.normal(0, s / math.sqrt(2 * L), (FF, D)); p[f"{l}.b2"] = np.zeros(D)
    return {k: v.astype(np.float32) for k, v in p.items()}


def param_order():
    names = ["E", "P"]
    for l in range(L):
        names += [f"{l}.{n}" for n in ["ln1g", "ln1b", "Wq", "Wk", "Wv", "Wo", "ln2g", "ln2b", "W1", "b1", "W2", "b2"]]
    return names + ["lnfg", "lnfb"]


def ln_fwd(x, g, b):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + 1e-5)
    xh = xc * rstd
    return xh * g + b, (xh, rstd)


def ln_bwd(dy, g, cache):
    xh, rstd = cache
    dxh = dy * g
    dx = rstd * (dxh - dxh.mean(-1, keepdims=True) - xh * (dxh * xh).mean(-1, keepdims=True))
    return dx, (dy * xh).reshape(-1, dy.shape[-1]).sum(0), dy.reshape(-1, dy.shape[-1]).sum(0)


C0 = math.sqrt(2.0 / math.pi)


def gelu_fwd(u):
    t = np.tanh(C0 * (u + 0.044715 * u ** 3))
    return 0.5 * u * (1 + t), t


def gelu_bwd(du, u, t):
    return du * (0.5 * (1 + t) + 0.5 * u * (1 - t * t) * C0 * (1 + 3 * 0.044715 * u * u))


def split_heads(x):  # B,T,D -> B,H,T,HD
    B, Tn, _ = x.shape
    return x.reshape(B, Tn, H, HD).transpose(0, 2, 1, 3)


def merge_heads(x):  # B,H,T,HD -> B,T,D
    B, _, Tn, _ = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, Tn, D)


def forward(p, ids, keep=False):
    B, Tn = ids.shape
    x = p["E"][ids] + p["P"][:Tn]
    caches = []
    mask = np.triu(np.full((Tn, Tn), -1e9, dtype=np.float32), 1)
    for l in range(L):
        h1, c1 = ln_fwd(x, p[f"{l}.ln1g"], p[f"{l}.ln1b"])
        q = split_heads(h1 @ p[f"{l}.Wq"]); k = split_heads(h1 @ p[f"{l}.Wk"]); v = split_heads(h1 @ p[f"{l}.Wv"])
        s = q @ k.transpose(0, 1, 3, 2) / math.sqrt(HD) + mask
        s = s - s.max(-1, keepdims=True)
        e = np.exp(s)
        pr = e / e.sum(-1, keepdims=True)
        o = merge_heads(pr @ v)
        x = x + o @ p[f"{l}.Wo"]
        h2, c2 = ln_fwd(x, p[f"{l}.ln2g"], p[f"{l}.ln2b"])
        u = h2 @ p[f"{l}.W1"] + p[f"{l}.b1"]
        g, t = gelu_fwd(u)
        x = x + g @ p[f"{l}.W2"] + p[f"{l}.b2"]
        if keep:
            caches.append((h1, c1, q, k, v, pr, o, h2, c2, u, g, t))
    hf, cf = ln_fwd(x, p["lnfg"], p["lnfb"])
    logits = hf @ p["E"].T
    return logits, (ids, caches, hf, cf)


def loss_and_grads(p, ids, tgt, w):
    logits, (ids, caches, hf, cf) = forward(p, ids, keep=True)
    B, Tn, V = logits.shape
    z = logits - logits.max(-1, keepdims=True)
    ez = np.exp(z)
    sm = ez / ez.sum(-1, keepdims=True)
    lse = np.log(ez.sum(-1))
    nll = lse - np.take_along_axis(z, tgt[..., None], -1)[..., 0]
    wsum = max(w.sum(), 1.0)
    loss = float((nll * w).sum() / wsum)
    dl = sm
    np.put_along_axis(dl, tgt[..., None], np.take_along_axis(dl, tgt[..., None], -1) - 1.0, -1)
    dl *= (w / wsum)[..., None]
    gr = {k: np.zeros_like(v) for k, v in p.items()}
    gr["E"] += dl.reshape(-1, V).T @ hf.reshape(-1, D)
    dhf = dl @ p["E"]
    dx, gr["lnfg"], gr["lnfb"] = ln_bwd(dhf, p["lnfg"], cf)
    for l in reversed(range(L)):
        h1, c1, q, k, v, pr, o, h2, c2, u, g, t = caches[l]
        flat = lambda a: a.reshape(-1, a.shape[-1])
        gr[f"{l}.W2"] = flat(g).T @ flat(dx); gr[f"{l}.b2"] = flat(dx).sum(0)
        dg = dx @ p[f"{l}.W2"].T
        du = gelu_bwd(dg, u, t)
        gr[f"{l}.W1"] = flat(h2).T @ flat(du); gr[f"{l}.b1"] = flat(du).sum(0)
        dh2 = du @ p[f"{l}.W1"].T
        dxl, gr[f"{l}.ln2g"], gr[f"{l}.ln2b"] = ln_bwd(dh2, p[f"{l}.ln2g"], c2)
        dx = dx + dxl
        gr[f"{l}.Wo"] = flat(o).T @ flat(dx)
        do = split_heads(dx @ p[f"{l}.Wo"].T)
        dpr = do @ v.transpose(0, 1, 3, 2)
        dv = pr.transpose(0, 1, 3, 2) @ do
        ds = pr * (dpr - (dpr * pr).sum(-1, keepdims=True)) / math.sqrt(HD)
        dq = ds @ k
        dk = ds.transpose(0, 1, 3, 2) @ q
        dq, dk, dv = merge_heads(dq), merge_heads(dk), merge_heads(dv)
        gr[f"{l}.Wq"] = flat(h1).T @ flat(dq); gr[f"{l}.Wk"] = flat(h1).T @ flat(dk); gr[f"{l}.Wv"] = flat(h1).T @ flat(dv)
        dh1 = dq @ p[f"{l}.Wq"].T + dk @ p[f"{l}.Wk"].T + dv @ p[f"{l}.Wv"].T
        dxl, gr[f"{l}.ln1g"], gr[f"{l}.ln1b"] = ln_bwd(dh1, p[f"{l}.ln1g"], c1)
        dx = dx + dxl
    np.add.at(gr["E"], ids.reshape(-1), dx.reshape(-1, D))
    gr["P"][:Tn] += dx.sum(0)
    return loss, gr


# ---------------- io ----------------
def save_bin(p, V, path=BIN):
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<6i", V, D, L, H, FF, T))
        for n in param_order():
            f.write(p[n].astype("<f4").tobytes())


def load_bin(path=BIN):
    with open(path, "rb") as f:
        assert f.read(8) == MAGIC
        V, d, l, h, ff, t = struct.unpack("<6i", f.read(24))
        assert (d, l, h, ff, t) == (D, L, H, FF, T)
        shapes = {"E": (V, D), "P": (T, D), "lnfg": (D,), "lnfb": (D,)}
        for i in range(L):
            for n, s in [("ln1g", (D,)), ("ln1b", (D,)), ("Wq", (D, D)), ("Wk", (D, D)), ("Wv", (D, D)), ("Wo", (D, D)),
                         ("ln2g", (D,)), ("ln2b", (D,)), ("W1", (D, FF)), ("b1", (FF,)), ("W2", (FF, D)), ("b2", (D,))]:
                shapes[f"{i}.{n}"] = s
        p = {}
        for n in param_order():
            cnt = int(np.prod(shapes[n]))
            p[n] = np.frombuffer(f.read(4 * cnt), dtype="<f4").reshape(shapes[n]).copy()
    return p


# ---------------- generation ----------------
def generate(p, vocab, history, user, temp=0.0, topk=1, maxlen=120, rng=None):
    c2i = {c: i for i, c in enumerate(vocab)}
    prompt = history + "ユーザー: " + user + "\n" + BOT
    ids = encode(prompt, c2i)
    out = ""
    for _ in range(maxlen):
        ctx = ids[-T:]
        logits, _ = forward(p, np.array([ctx], dtype=np.int32))
        lg = logits[0, -1].astype(np.float64)
        lg[[PAD, UNK, EOS]] = -1e9
        if temp <= 0:
            nxt = int(lg.argmax())
        else:
            idx = np.argsort(lg)[-topk:]
            pr = np.exp((lg[idx] - lg[idx].max()) / temp)
            nxt = int(rng.choice(idx, p=pr / pr.sum()))
        ids.append(nxt)
        if vocab[nxt] == "\n":
            break
        out += vocab[nxt]
    return out


# ---------------- commands ----------------
def cmd_train(steps):
    convs = load_convs()
    vocab = build_vocab(convs)
    c2i = {c: i for i, c in enumerate(vocab)}
    json.dump(vocab, open(VOCAB, "w", encoding="utf-8"), ensure_ascii=False)
    ids, mask, starts = build_stream(convs, c2i)
    V = len(vocab)
    print(f"convs={len(convs)} tokens={len(ids)} V={V}", flush=True)
    rng = np.random.default_rng(1234)
    ck = os.path.join(ROOT, "tools", "ckpt.npz")
    if os.path.exists(ck):
        z = np.load(ck)
        p = {k: z["p_" + k] for k in param_order()}
        m = {k: z["m_" + k] for k in param_order()}
        v = {k: z["v_" + k] for k in param_order()}
        step0 = int(z["step"])
        print("resume from", step0)
    else:
        p = init_params(V, rng)
        m = {k: np.zeros_like(a) for k, a in p.items()}
        v = {k: np.zeros_like(a) for k, a in p.items()}
        step0 = 0
    B, lr_max, warm = 16, 3e-3, 100
    b1, b2, eps, wd = 0.9, 0.99, 1e-8, 0.01
    t0 = time.time()
    ema = None
    for step in range(step0, steps):
        lr = lr_max * min(1.0, (step + 1) / warm) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        bi = rng.choice(starts, B)
        X = np.stack([np.resize(ids[s:s + T + 1], T + 1) if s + T + 1 > len(ids) else ids[s:s + T + 1] for s in bi])
        Wm = np.stack([np.resize(mask[s:s + T + 1], T + 1) if s + T + 1 > len(ids) else mask[s:s + T + 1] for s in bi])
        loss, gr = loss_and_grads(p, X[:, :T], X[:, 1:].astype(np.int64), Wm[:, 1:])
        gn = math.sqrt(sum(float((g * g).sum()) for g in gr.values()))
        sc = min(1.0, 1.0 / (gn + 1e-6))
        for k in p:
            g = gr[k] * sc
            m[k] = b1 * m[k] + (1 - b1) * g
            v[k] = b2 * v[k] + (1 - b2) * g * g
            mh = m[k] / (1 - b1 ** (step + 1)); vh = v[k] / (1 - b2 ** (step + 1))
            upd = mh / (np.sqrt(vh) + eps)
            if p[k].ndim == 2 and k not in ("E", "P"):
                upd = upd + wd * p[k]
            p[k] -= lr * upd
        ema = loss if ema is None else 0.98 * ema + 0.02 * loss
        if step % 20 == 0:
            print(f"step {step} loss {loss:.4f} ema {ema:.4f} lr {lr:.5f} {time.time()-t0:.0f}s", flush=True)
        if (step + 1) % 200 == 0 or step + 1 == steps:
            save_bin(p, V)
            np.savez(ck, step=step + 1, **{"p_" + k: p[k] for k in p}, **{"m_" + k: m[k] for k in p}, **{"v_" + k: v[k] for k in p})
    save_bin(p, V)
    print("done", flush=True)


def build_history(turns, user, c2i, ctx=T, reserve=70):
    """Go の chat.BuildPrompt と同じ: 古いターンから捨てて返信分の余白を確保する。"""
    budget = ctx - reserve
    for start in range(len(turns) + 1):
        h = "".join("ユーザー: " + u + "\n" + BOT + b + "\n" for u, b in turns[start:])
        if start >= len(turns) or len(encode(h + "ユーザー: " + user + "\n" + BOT, c2i)) <= budget:
            return h
