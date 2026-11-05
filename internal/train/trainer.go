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
