package train

import (
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"github.com/rimsky-yamatov/gollm/internal/model"
	"github.com/rimsky-yamatov/gollm/internal/tokenizer"
)

// Options configures Run.
type Options struct {
	DataDir   string
	OutModel  string
	OutVocab  string
	Steps     int
	Batch     int
	LR        float64
	Warmup    int
	Seed      int64
	Resume    bool
	SaveEvery int
	LogEvery  int
	Log       io.Writer
	Arch      model.Config // Vocab is filled in from the corpus
}

// DefaultOptions mirrors tools/train_numpy.py.
func DefaultOptions() Options {
	return Options{
		DataDir: "data/corpus", OutModel: "model/gollm.bin", OutVocab: "model/vocab.json",
		Steps: 4000, Batch: 16, LR: 3e-3, Warmup: 100, Seed: 1234,
		SaveEvery: 200, LogEvery: 20, Log: os.Stdout,
		Arch: model.Default(0),
	}
}

// LRAt returns the learning rate for a step: linear warm-up, then cosine
// decay to 5% of the peak.
func LRAt(o Options, step int) float64 {
	warm := math.Min(1, float64(step+1)/float64(o.Warmup))
	cos := 0.05 + 0.95*0.5*(1+math.Cos(math.Pi*float64(step)/float64(o.Steps)))
	return o.LR * warm * cos
}

// Run trains a model from scratch (or resumes from OutModel) on the corpus.
func Run(o Options) error {
	convs, err := LoadConversations(o.DataDir)
	if err != nil {
		return err
	}
	var voc *tokenizer.Vocab
	if o.Resume {
		if voc, err = tokenizer.Load(o.OutVocab); err != nil {
			return err
		}
	} else {
		voc = tokenizer.Build(convs)
	}
	stream := BuildStream(convs, voc)
	cfg := o.Arch
	cfg.Vocab = voc.Size()
	if err := cfg.Validate(); err != nil {
		return err
	}
	fmt.Fprintf(o.Log, "conversations=%d tokens=%d vocab=%d params=%d\n",
		len(convs), len(stream.IDs), voc.Size(), cfg.NumParams())

	rng := rand.New(rand.NewSource(o.Seed))
	var w *model.Weights
	if o.Resume {
		if w, err = model.Load(o.OutModel); err != nil {
			return err
		}
		cfg = w.Cfg
	} else {
		w = model.New(cfg, rng)
	}
	if err := os.MkdirAll(filepath.Dir(o.OutModel), 0o755); err != nil {
		return err
	}
	if err := voc.Save(o.OutVocab); err != nil {
		return err
	}

	T := cfg.Ctx
	tr := NewTrainer(w, o.Batch, T)
	x := make([]int32, o.Batch*T)
	y := make([]int32, o.Batch*T)
	wt := make([]float32, o.Batch*T)
	start := time.Now()
	var ema float64
	for step := 0; step < o.Steps; step++ {
		stream.Batch(rng, o.Batch, T, x, y, wt)
		loss := float64(tr.LossGrad(x, y, wt))
		lr := LRAt(o, step)
		gn := tr.Update(lr)
		if step == 0 {
			ema = loss
		} else {
			ema = 0.98*ema + 0.02*loss
		}
		if o.LogEvery > 0 && step%o.LogEvery == 0 {
			fmt.Fprintf(o.Log, "step %d loss %.4f ema %.4f gnorm %.3f lr %.5f %.0fs\n",
				step, loss, ema, gn, lr, time.Since(start).Seconds())
		}
		if (o.SaveEvery > 0 && (step+1)%o.SaveEvery == 0) || step+1 == o.Steps {
			if err := model.Save(o.OutModel, w); err != nil {
				return err
			}
		}
	}
	fmt.Fprintln(o.Log, "done")
	return nil
}
