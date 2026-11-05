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
