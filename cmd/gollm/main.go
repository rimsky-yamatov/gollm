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

func runChat(args []string, single bool) error {
	fs := flag.NewFlagSet("chat", flag.ExitOnError)
	mp := fs.String("model", "model/gollm.bin", "weights file")
	vp := fs.String("vocab", "model/vocab.json", "vocab file")
	temp := fs.Float64("temp", 0.5, "sampling temperature (0 = greedy)")
	topk := fs.Int("topk", 8, "top-k sampling")
	seed := fs.Int64("seed", 0, "random seed (0 = time)")
	prompt := fs.String("prompt", "", "single prompt (gen)")
	fs.Parse(args)

	w, v, err := loadAll(*mp, *vp)
	if err != nil {
		return err
	}
	if *seed == 0 {
		*seed = time.Now().UnixNano()
	}
	c := chat.New(w, v, sample.Options{Temp: *temp, TopK: *topk, MaxLen: 120, Rng: rand.New(rand.NewSource(*seed))})
	if single {
		if *prompt == "" {
			return fmt.Errorf("-prompt is required")
		}
		r, err := c.Reply(*prompt)
		if err != nil {
			return err
		}
		fmt.Println(r)
		return nil
	}
	fmt.Println("ノックスです。何でも話しかけてください。(/reset でリセット、/quit で終了)")
	sc := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("あなた> ")
		if !sc.Scan() {
			fmt.Println()
			return sc.Err()
		}
		line := strings.TrimSpace(sc.Text())
		switch line {
		case "":
			continue
		case "/quit", "/exit":
			return nil
		case "/reset":
			c.Reset()
			fmt.Println("(会話をリセットしました)")
			continue
		}
		r, err := c.Reply(line)
		if err != nil {
			return err
		}
		fmt.Println("ノックス>", r)
	}
}

func runTrain(args []string) error {
	o := train.DefaultOptions()
	fs := flag.NewFlagSet("train", flag.ExitOnError)
	fs.StringVar(&o.DataDir, "data", o.DataDir, "corpus directory (*.txt)")
	fs.StringVar(&o.OutModel, "out", o.OutModel, "output weights")
	fs.StringVar(&o.OutVocab, "vocab", o.OutVocab, "output vocab")
	fs.IntVar(&o.Steps, "steps", o.Steps, "training steps")
	fs.IntVar(&o.Batch, "batch", o.Batch, "batch size")
	fs.Float64Var(&o.LR, "lr", o.LR, "peak learning rate")
	fs.Int64Var(&o.Seed, "seed", o.Seed, "random seed")
	fs.BoolVar(&o.Resume, "resume", false, "continue from -out")
	fs.Parse(args)
	return train.Run(o)
}
