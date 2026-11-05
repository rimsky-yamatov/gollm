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
