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

func (t *Trainer) forward(ids []int32) {
	c := t.cfg
	d, ff, N, V := c.Dim, c.FF, t.N, c.Vocab
	t.ids = ids
	for n := 0; n < N; n++ {
		e := t.W.E[int(ids[n])*d : (int(ids[n])+1)*d]
		p := t.W.P[(n%t.T)*d : (n%t.T+1)*d]
		x := t.x[n*d : (n+1)*d]
		for i := range x {
			x[i] = e[i] + p[i]
		}
	}
	for l := range t.W.L {
		ly := &t.W.L[l]
		a := &t.ac[l]
		tensor.LayerNormFwd(a.h1, a.xh1, a.rs1, t.x, ly.Ln1G, ly.Ln1B, N, d)
		tensor.MatMul(a.q, a.h1, ly.Wq, N, d, d)
		tensor.MatMul(a.k, a.h1, ly.Wk, N, d, d)
		tensor.MatMul(a.v, a.h1, ly.Wv, N, d, d)
		t.attnFwd(a)
		tensor.MatMul(t.tmp, a.o, ly.Wo, N, d, d)
		tensor.AddInPlace(t.x, t.tmp)

		tensor.LayerNormFwd(a.h2, a.xh2, a.rs2, t.x, ly.Ln2G, ly.Ln2B, N, d)
		tensor.MatMul(a.u, a.h2, ly.W1, N, d, ff)
		tensor.AddBias(a.u, ly.B1, N, ff)
		tensor.GeluFwd(a.gl, a.th, a.u)
		tensor.MatMul(t.tmp, a.gl, ly.W2, N, ff, d)
		tensor.AddBias(t.tmp, ly.B2, N, d)
		tensor.AddInPlace(t.x, t.tmp)
	}
	tensor.LayerNormFwd(t.hf, t.xhf, t.rsf, t.x, t.W.LnfG, t.W.LnfB, N, d)
	tensor.MatMulBT(t.logits, t.hf, t.W.E, N, d, V)
}

// Loss runs a forward pass only and returns the masked mean cross-entropy.
func (t *Trainer) Loss(ids, tgt []int32, w []float32) float32 {
	t.forward(ids)
	return float32(t.lossAndDLogits(tgt, w))
}

// lossAndDLogits converts t.logits in place into ∂L/∂logits and returns L.
func (t *Trainer) lossAndDLogits(tgt []int32, w []float32) float64 {
	V := t.cfg.Vocab
	var wsum float64
	for _, x := range w {
		wsum += float64(x)
	}
	if wsum < 1 {
		wsum = 1
	}
	var loss float64
	for n := 0; n < t.N; n++ {
		row := t.logits[n*V : (n+1)*V]
		mx := row[0]
		for _, v := range row {
			if v > mx {
				mx = v
			}
		}
		tl := float64(row[tgt[n]] - mx)
		var sum float64
		for i, v := range row {
			e := math.Exp(float64(v - mx))
			row[i] = float32(e)
			sum += e
		}
		wn := float64(w[n])
		loss += wn * (math.Log(sum) - tl)
		inv := 1 / sum
		sc := float32(wn / wsum)
		for i := range row {
			row[i] = float32(float64(row[i])*inv) * sc
		}
		row[tgt[n]] -= sc
	}
	return loss / wsum
}

// LossGrad computes the loss for a batch and accumulates ∂L/∂θ into t.G
// (t.G is zeroed first). ids and tgt have length B*T.
func (t *Trainer) LossGrad(ids, tgt []int32, w []float32) float32 {
	for _, p := range t.G.Params() {
		tensor.Zero(p.Data)
	}
	t.forward(ids)
	loss := t.lossAndDLogits(tgt, w)

	c := t.cfg
	d, ff, N, V := c.Dim, c.FF, t.N, c.Vocab
	G := t.G
	tensor.MatMulATAcc(G.E, t.logits, t.hf, N, V, d)
	tensor.MatMul(t.dtmp, t.logits, t.W.E, N, V, d)
	tensor.Zero(t.dx)
	tensor.LayerNormBwd(t.dx, G.LnfG, G.LnfB, t.dtmp, t.xhf, t.rsf, t.W.LnfG, N, d)

	for l := c.Layers - 1; l >= 0; l-- {
		ly, gl, a := &t.W.L[l], &G.L[l], &t.ac[l]
		// MLP
		tensor.MatMulATAcc(gl.W2, a.gl, t.dx, N, ff, d)
		tensor.ColSumAcc(gl.B2, t.dx, N, d)
		tensor.MatMulBT(t.dgl, t.dx, ly.W2, N, d, ff)
		tensor.GeluBwd(t.du, t.dgl, a.u, a.th)
		tensor.MatMulATAcc(gl.W1, a.h2, t.du, N, d, ff)
		tensor.ColSumAcc(gl.B1, t.du, N, ff)
		tensor.MatMulBT(t.dh, t.du, ly.W1, N, ff, d)
		tensor.LayerNormBwd(t.dx, gl.Ln2G, gl.Ln2B, t.dh, a.xh2, a.rs2, ly.Ln2G, N, d)
		// Attention
		tensor.MatMulATAcc(gl.Wo, a.o, t.dx, N, d, d)
		tensor.MatMulBT(t.dO, t.dx, ly.Wo, N, d, d)
		tensor.Zero(t.dq)
		tensor.Zero(t.dk)
		tensor.Zero(t.dv)
		t.attnBwd(a)
		tensor.MatMulATAcc(gl.Wq, a.h1, t.dq, N, d, d)
		tensor.MatMulATAcc(gl.Wk, a.h1, t.dk, N, d, d)
		tensor.MatMulATAcc(gl.Wv, a.h1, t.dv, N, d, d)
		tensor.MatMulBT(t.dh, t.dq, ly.Wq, N, d, d)
		tensor.MatMulBT(t.dtmp, t.dk, ly.Wk, N, d, d)
		tensor.AddInPlace(t.dh, t.dtmp)
		tensor.MatMulBT(t.dtmp, t.dv, ly.Wv, N, d, d)
		tensor.AddInPlace(t.dh, t.dtmp)
		tensor.LayerNormBwd(t.dx, gl.Ln1G, gl.Ln1B, t.dh, a.xh1, a.rs1, ly.Ln1G, N, d)
	}
	for n := 0; n < N; n++ {
		dxr := t.dx[n*d : (n+1)*d]
		ge := G.E[int(t.ids[n])*d : (int(t.ids[n])+1)*d]
		gp := G.P[(n%t.T)*d : (n%t.T+1)*d]
		for i, v := range dxr {
			ge[i] += v
			gp[i] += v
		}
	}
	return float32(loss)
}

// Adam hyper-parameters.
const (
	beta1  = 0.9
	beta2  = 0.99
	adamEps = 1e-8
	weightDecay = 0.01
	gradClip = 1.0
)

// Update applies one AdamW step with the gradients currently in t.G.
// It returns the global gradient norm (before clipping).
func (t *Trainer) Update(lr float64) float64 {
	params, grads := t.W.Params(), t.G.Params()
	ms, vs := t.M.Params(), t.V.Params()
	var sq float64
	for _, g := range grads {
		for _, x := range g.Data {
			sq += float64(x) * float64(x)
		}
	}
	gn := math.Sqrt(sq)
	clip := math.Min(1, gradClip/(gn+1e-6))
	t.step++
	bc1 := 1 - math.Pow(beta1, float64(t.step))
	bc2 := 1 - math.Pow(beta2, float64(t.step))
	for i := range params {
		p, g, m, v := params[i], grads[i].Data, ms[i].Data, vs[i].Data
		tensor.Parallel(len(p.Data), func(lo, hi int) {
			for j := lo; j < hi; j++ {
				gj := float64(g[j]) * clip
				mj := beta1*float64(m[j]) + (1-beta1)*gj
				vj := beta2*float64(v[j]) + (1-beta2)*gj*gj
				m[j], v[j] = float32(mj), float32(vj)
				upd := (mj / bc1) / (math.Sqrt(vj/bc2) + adamEps)
				if p.Decay {
					upd += weightDecay * float64(p.Data[j])
				}
				p.Data[j] -= float32(lr * upd)
			}
		})
	}
	return gn
}

// Steps returns the number of optimiser updates performed.
func (t *Trainer) Steps() int { return t.step }
