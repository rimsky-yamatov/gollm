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
