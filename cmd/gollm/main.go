// Command gollm is a small Japanese chat language model written in Go.
//
//	gollm chat   [-model model/gollm.bin] [-vocab model/vocab.json]
//	gollm gen    -prompt "こんにちは"
//	gollm train  [-data data/corpus] [-steps 4000] [-resume]
//	gollm info
package main

import (
	"bufio"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"strings"
	"time"

	"github.com/rimsky-yamatov/gollm/internal/chat"
	"github.com/rimsky-yamatov/gollm/internal/model"
	"github.com/rimsky-yamatov/gollm/internal/sample"
	"github.com/rimsky-yamatov/gollm/internal/tokenizer"
	"github.com/rimsky-yamatov/gollm/internal/train"
)

const usage = `gollm - 日本語で会話できる小さな言語モデル (Go実装)

使い方:
  gollm chat  [-model PATH] [-vocab PATH] [-temp 0.5] [-topk 8] [-seed N]
  gollm gen   -prompt "こんにちは" [同上]
  gollm train [-data DIR] [-steps 4000] [-batch 16] [-lr 0.003] [-resume]
  gollm info  [-model PATH]
`
