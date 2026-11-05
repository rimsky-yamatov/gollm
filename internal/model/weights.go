package model

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"os"
)

// Layer holds the parameters of one Transformer block.
// Matrices are [in*out] row-major so that y = x·W.
type Layer struct {
	Ln1G, Ln1B []float32
	Wq, Wk, Wv []float32
	Wo         []float32
	Ln2G, Ln2B []float32
	W1, B1     []float32
	W2, B2     []float32
}

// Weights is the complete parameter set. The output projection is tied to E.
type Weights struct {
	Cfg        Config
	E          []float32 // token embedding [Vocab*Dim]
	P          []float32 // position embedding [Ctx*Dim]
	L          []Layer
	LnfG, LnfB []float32
}

// Param is a named view onto one parameter tensor.
type Param struct {
	Name  string
	Data  []float32
	Decay bool // apply weight decay (2-D weight matrices except embeddings)
}

// Zeros allocates a zero-filled Weights for cfg (used for gradients/moments).
func Zeros(cfg Config) *Weights {
	d, ff := cfg.Dim, cfg.FF
	w := &Weights{
		Cfg:  cfg,
		E:    make([]float32, cfg.Vocab*d),
		P:    make([]float32, cfg.Ctx*d),
		LnfG: make([]float32, d),
		LnfB: make([]float32, d),
	}
	for i := 0; i < cfg.Layers; i++ {
		w.L = append(w.L, Layer{
			Ln1G: make([]float32, d), Ln1B: make([]float32, d),
			Wq: make([]float32, d*d), Wk: make([]float32, d*d), Wv: make([]float32, d*d),
			Wo:   make([]float32, d*d),
			Ln2G: make([]float32, d), Ln2B: make([]float32, d),
			W1: make([]float32, d*ff), B1: make([]float32, ff),
			W2: make([]float32, ff*d), B2: make([]float32, d),
		})
	}
	return w
}

// New creates freshly initialised weights (GPT-2 style, std 0.02).
func New(cfg Config, rng *rand.Rand) *Weights {
	w := Zeros(cfg)
	norm := func(a []float32, std float64) {
		for i := range a {
			a[i] = float32(rng.NormFloat64() * std)
		}
	}
	ones := func(a []float32) {
		for i := range a {
			a[i] = 1
		}
	}
	const s = 0.02
	resid := s / sqrt(float64(2*cfg.Layers))
	norm(w.E, s)
	norm(w.P, s)
	ones(w.LnfG)
	for i := range w.L {
		l := &w.L[i]
		ones(l.Ln1G)
		ones(l.Ln2G)
		norm(l.Wq, s)
		norm(l.Wk, s)
		norm(l.Wv, s)
		norm(l.Wo, resid)
		norm(l.W1, s)
		norm(l.W2, resid)
	}
	return w
}
