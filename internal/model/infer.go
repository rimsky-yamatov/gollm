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

// NewSession allocates a decoding session for w.
func NewSession(w *Weights) *Session {
	c := w.Cfg
	s := &Session{W: w}
	for i := 0; i < c.Layers; i++ {
		s.kc = append(s.kc, make([]float32, c.Ctx*c.Dim))
		s.vc = append(s.vc, make([]float32, c.Ctx*c.Dim))
	}
	mk := func(n int) []float32 { return make([]float32, n) }
	s.x, s.h, s.q, s.k, s.v = mk(c.Dim), mk(c.Dim), mk(c.Dim), mk(c.Dim), mk(c.Dim)
	s.o, s.tmp, s.hf = mk(c.Dim), mk(c.Dim), mk(c.Dim)
	s.u, s.g = mk(c.FF), mk(c.FF)
	s.scores = mk(c.Ctx)
	s.logits = mk(c.Vocab)
	return s
}

// Pos is the number of tokens consumed so far.
func (s *Session) Pos() int { return s.pos }

// Reset clears the context.
func (s *Session) Reset() { s.pos = 0 }
