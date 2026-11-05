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
