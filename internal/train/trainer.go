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
