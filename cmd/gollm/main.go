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

func main() {
	if len(os.Args) < 2 {
		fmt.Print(usage)
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "chat":
		err = runChat(os.Args[2:], false)
	case "gen":
		err = runChat(os.Args[2:], true)
	case "train":
		err = runTrain(os.Args[2:])
	case "info":
		err = runInfo(os.Args[2:])
	case "-h", "--help", "help":
		fmt.Print(usage)
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n\n%s", os.Args[1], usage)
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func loadAll(modelPath, vocabPath string) (*model.Weights, *tokenizer.Vocab, error) {
	w, err := model.Load(modelPath)
	if err != nil {
		return nil, nil, err
	}
	v, err := tokenizer.Load(vocabPath)
	if err != nil {
		return nil, nil, err
	}
	if v.Size() != w.Cfg.Vocab {
		return nil, nil, fmt.Errorf("vocab size %d does not match model (%d)", v.Size(), w.Cfg.Vocab)
	}
	return w, v, nil
}
