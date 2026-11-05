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

// Params lists all tensors in the on-disk order.
func (w *Weights) Params() []Param {
	ps := []Param{{"E", w.E, false}, {"P", w.P, false}}
	for i := range w.L {
		l := &w.L[i]
		p := fmt.Sprintf("%d.", i)
		ps = append(ps,
			Param{p + "ln1g", l.Ln1G, false}, Param{p + "ln1b", l.Ln1B, false},
			Param{p + "Wq", l.Wq, true}, Param{p + "Wk", l.Wk, true},
			Param{p + "Wv", l.Wv, true}, Param{p + "Wo", l.Wo, true},
			Param{p + "ln2g", l.Ln2G, false}, Param{p + "ln2b", l.Ln2B, false},
			Param{p + "W1", l.W1, true}, Param{p + "b1", l.B1, false},
			Param{p + "W2", l.W2, true}, Param{p + "b2", l.B2, false},
		)
	}
	return append(ps, Param{"lnfg", w.LnfG, false}, Param{"lnfb", w.LnfB, false})
}

const magic = "GOLLM001"

// Save writes the weights in the gollm binary format.
func Save(path string, w *Weights) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	bw := bufio.NewWriter(f)
	if _, err := bw.WriteString(magic); err != nil {
		f.Close()
		return err
	}
	c := w.Cfg
	hdr := []int32{int32(c.Vocab), int32(c.Dim), int32(c.Layers), int32(c.Heads), int32(c.FF), int32(c.Ctx)}
	if err := binary.Write(bw, binary.LittleEndian, hdr); err != nil {
		f.Close()
		return err
	}
	for _, p := range w.Params() {
		if err := binary.Write(bw, binary.LittleEndian, p.Data); err != nil {
			f.Close()
			return err
		}
	}
	if err := bw.Flush(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
