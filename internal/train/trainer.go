package train

import (
	"math"

	"github.com/rimsky-yamatov/gollm/internal/model"
	"github.com/rimsky-yamatov/gollm/internal/tensor"
)

type layerAct struct {
	h1, xh1, rs1 []float32
	q, k, v      []float32
	pr           []float32 // [B*H*T*T]
	o            []float32
	h2, xh2, rs2 []float32
	u, gl, th    []float32
}

// Trainer holds weights, gradients, Adam state and all activation buffers for
// a fixed (batch, T) shape.
type Trainer struct {
	W, G, M, V *model.Weights
	B, T, N    int
	cfg        model.Config
	step       int

	x, tmp, dx, dtmp   []float32
	ac                 []layerAct
	hf, xhf, rsf       []float32
	logits             []float32
	dO, dq, dk, dv     []float32
	dgl, du, dh        []float32
	ids                []int32
}

// NewTrainer allocates buffers for batch sequences of length T.
func NewTrainer(w *model.Weights, batch, T int) *Trainer {
	c := w.Cfg
	n := batch * T
	d, ff := c.Dim, c.FF
	mk := func(k int) []float32 { return make([]float32, k) }
	t := &Trainer{
		W: w, G: model.Zeros(c), M: model.Zeros(c), V: model.Zeros(c),
		B: batch, T: T, N: n, cfg: c,
		x: mk(n * d), tmp: mk(n * d), dx: mk(n * d), dtmp: mk(n * d),
		hf: mk(n * d), xhf: mk(n * d), rsf: mk(n),
		logits: mk(n * c.Vocab),
		dO:     mk(n * d), dq: mk(n * d), dk: mk(n * d), dv: mk(n * d),
		dgl: mk(n * ff), du: mk(n * ff), dh: mk(n * d),
	}
	for i := 0; i < c.Layers; i++ {
		t.ac = append(t.ac, layerAct{
			h1: mk(n * d), xh1: mk(n * d), rs1: mk(n),
			q: mk(n * d), k: mk(n * d), v: mk(n * d),
			pr: mk(batch * c.Heads * T * T),
			o:  mk(n * d),
			h2: mk(n * d), xh2: mk(n * d), rs2: mk(n),
			u: mk(n * ff), gl: mk(n * ff), th: mk(n * ff),
		})
	}
	return t
}
