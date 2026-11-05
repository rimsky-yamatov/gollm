package model

import (
	"errors"
	"fmt"
	"math"

	"github.com/rimsky-yamatov/gollm/internal/tensor"
)

// ErrContextFull is returned by Step when the context window is exhausted.
var ErrContextFull = errors.New("gollm: context window is full")

// Session performs incremental decoding with a per-layer key/value cache.
// It is not safe for concurrent use.
type Session struct {
	W      *Weights
	pos    int
	kc, vc [][]float32 // [layer][Ctx*Dim]

	x, h, q, k, v, o, tmp, hf []float32
	u, g                      []float32
	scores                    []float32
	logits                    []float32
}
