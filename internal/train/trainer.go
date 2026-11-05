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

func (t *Trainer) attnFwd(a *layerAct) {
	c := t.cfg
	d, hd, T, H := c.Dim, c.HeadDim(), t.T, c.Heads
	scale := float32(1 / math.Sqrt(float64(hd)))
	tensor.Parallel(t.B*H, func(lo, hi int) {
		for bh := lo; bh < hi; bh++ {
			b, h := bh/H, bh%H
			for i := 0; i < T; i++ {
				qi := a.q[(b*T+i)*d+h*hd : (b*T+i)*d+(h+1)*hd]
				pr := a.pr[(bh*T+i)*T : (bh*T+i)*T+T]
				mx := float32(-1e30)
				for j := 0; j <= i; j++ {
					kj := a.k[(b*T+j)*d+h*hd : (b*T+j)*d+(h+1)*hd]
					var s float32
					for u := 0; u < hd; u++ {
						s += qi[u] * kj[u]
					}
					s *= scale
					pr[j] = s
					if s > mx {
						mx = s
					}
				}
				var sum float32
				for j := 0; j <= i; j++ {
					e := float32(math.Exp(float64(pr[j] - mx)))
					pr[j] = e
					sum += e
				}
				inv := 1 / sum
				for j := 0; j <= i; j++ {
					pr[j] *= inv
				}
				for j := i + 1; j < T; j++ {
					pr[j] = 0
				}
				oi := a.o[(b*T+i)*d+h*hd : (b*T+i)*d+(h+1)*hd]
				for u := range oi {
					oi[u] = 0
				}
				for j := 0; j <= i; j++ {
					vj := a.v[(b*T+j)*d+h*hd : (b*T+j)*d+(h+1)*hd]
					p := pr[j]
					for u := range oi {
						oi[u] += p * vj[u]
					}
				}
			}
		}
	})
}

func (t *Trainer) attnBwd(a *layerAct) {
	c := t.cfg
	d, hd, T, H := c.Dim, c.HeadDim(), t.T, c.Heads
	scale := float32(1 / math.Sqrt(float64(hd)))
	tensor.Parallel(t.B*H, func(lo, hi int) {
		dpr := make([]float32, T)
		for bh := lo; bh < hi; bh++ {
			b, h := bh/H, bh%H
			for i := 0; i < T; i++ {
				doi := t.dO[(b*T+i)*d+h*hd : (b*T+i)*d+(h+1)*hd]
				pr := a.pr[(bh*T+i)*T : (bh*T+i)*T+T]
				var dot float32
				for j := 0; j <= i; j++ {
					vj := a.v[(b*T+j)*d+h*hd : (b*T+j)*d+(h+1)*hd]
					dvj := t.dv[(b*T+j)*d+h*hd : (b*T+j)*d+(h+1)*hd]
					var s float32
					for u := 0; u < hd; u++ {
						s += doi[u] * vj[u]
						dvj[u] += pr[j] * doi[u]
					}
					dpr[j] = s
					dot += s * pr[j]
				}
				qi := a.q[(b*T+i)*d+h*hd : (b*T+i)*d+(h+1)*hd]
				dqi := t.dq[(b*T+i)*d+h*hd : (b*T+i)*d+(h+1)*hd]
				for j := 0; j <= i; j++ {
					ds := pr[j] * (dpr[j] - dot) * scale
					kj := a.k[(b*T+j)*d+h*hd : (b*T+j)*d+(h+1)*hd]
					dkj := t.dk[(b*T+j)*d+h*hd : (b*T+j)*d+(h+1)*hd]
					for u := 0; u < hd; u++ {
						dqi[u] += ds * kj[u]
						dkj[u] += ds * qi[u]
					}
				}
			}
		}
	})
}
