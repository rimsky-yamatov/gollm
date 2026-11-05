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

// Feed consumes tokens and returns the logits after the last one.
func (s *Session) Feed(ids []int) ([]float32, error) {
	var lg []float32
	var err error
	for _, id := range ids {
		if lg, err = s.Step(id); err != nil {
			return nil, err
		}
	}
	return lg, nil
}

// Step consumes one token and returns the next-token logits. The returned
// slice is reused by the following call.
func (s *Session) Step(tok int) ([]float32, error) {
	c := s.W.Cfg
	if s.pos >= c.Ctx {
		return nil, ErrContextFull
	}
	if tok < 0 || tok >= c.Vocab {
		return nil, fmt.Errorf("gollm: token id %d out of range", tok)
	}
	d, hd, ff := c.Dim, c.HeadDim(), c.FF
	e := s.W.E[tok*d : (tok+1)*d]
	p := s.W.P[s.pos*d : (s.pos+1)*d]
	for i := 0; i < d; i++ {
		s.x[i] = e[i] + p[i]
	}
	scale := float32(1 / math.Sqrt(float64(hd)))
	for l := range s.W.L {
		ly := &s.W.L[l]
		tensor.LayerNormFwd(s.h, nil, nil, s.x, ly.Ln1G, ly.Ln1B, 1, d)
		tensor.MatMul(s.q, s.h, ly.Wq, 1, d, d)
		tensor.MatMul(s.k, s.h, ly.Wk, 1, d, d)
		tensor.MatMul(s.v, s.h, ly.Wv, 1, d, d)
		copy(s.kc[l][s.pos*d:(s.pos+1)*d], s.k)
		copy(s.vc[l][s.pos*d:(s.pos+1)*d], s.v)
		for h := 0; h < c.Heads; h++ {
			qh := s.q[h*hd : (h+1)*hd]
			sc := s.scores[:s.pos+1]
			for j := 0; j <= s.pos; j++ {
				kh := s.kc[l][j*d+h*hd : j*d+(h+1)*hd]
				var dot float32
				for i, qv := range qh {
					dot += qv * kh[i]
				}
				sc[j] = dot * scale
			}
			tensor.Softmax(sc)
			oh := s.o[h*hd : (h+1)*hd]
			for i := range oh {
				oh[i] = 0
			}
			for j := 0; j <= s.pos; j++ {
				vh := s.vc[l][j*d+h*hd : j*d+(h+1)*hd]
				pj := sc[j]
				for i := range oh {
					oh[i] += pj * vh[i]
				}
			}
		}
		tensor.MatMul(s.tmp, s.o, ly.Wo, 1, d, d)
		tensor.AddInPlace(s.x, s.tmp)

		tensor.LayerNormFwd(s.h, nil, nil, s.x, ly.Ln2G, ly.Ln2B, 1, d)
		tensor.MatMul(s.u, s.h, ly.W1, 1, d, ff)
		tensor.AddBias(s.u, ly.B1, 1, ff)
		tensor.GeluFwd(s.g, nil, s.u)
		tensor.MatMul(s.tmp, s.g, ly.W2, 1, ff, d)
		tensor.AddBias(s.tmp, ly.B2, 1, d)
		tensor.AddInPlace(s.x, s.tmp)
	}
	tensor.LayerNormFwd(s.hf, nil, nil, s.x, s.W.LnfG, s.W.LnfB, 1, d)
	tensor.MatMulBT(s.logits, s.hf, s.W.E, 1, d, c.Vocab)
	s.pos++
	return s.logits, nil
}
