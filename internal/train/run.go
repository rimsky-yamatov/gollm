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
