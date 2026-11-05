package train

import (
	"math"
	"math/rand"
	"testing"

	"github.com/rimsky-yamatov/gollm/internal/model"
)

// TestGradientCheck compares analytic gradients with central finite
// differences on a tiny model.
func TestGradientCheck(t *testing.T) {
	cfg := model.Config{Vocab: 11, Dim: 8, Layers: 2, Heads: 2, FF: 16, Ctx: 6}
	rng := rand.New(rand.NewSource(7))
	w := model.New(cfg, rng)
	// Make the weights large enough for gradients to be well conditioned.
	for _, p := range w.Params() {
		for i := range p.Data {
			p.Data[i] += float32(rng.NormFloat64() * 0.1)
		}
	}
	const B, T = 2, 6
	tr := NewTrainer(w, B, T)
	ids := make([]int32, B*T)
	tgt := make([]int32, B*T)
	wt := make([]float32, B*T)
	for i := range ids {
		ids[i] = int32(rng.Intn(cfg.Vocab))
		tgt[i] = int32(rng.Intn(cfg.Vocab))
		wt[i] = 1
	}
	tr.LossGrad(ids, tgt, wt)
	grads := tr.G.Params()
	params := w.Params()

	const eps = 1e-2
	checked := 0
	for pi, p := range params {
		for trial := 0; trial < 2; trial++ {
			j := rng.Intn(len(p.Data))
			ana := float64(grads[pi].Data[j])
			old := p.Data[j]
			p.Data[j] = old + eps
			lp := float64(tr.Loss(ids, tgt, wt))
			p.Data[j] = old - eps
			lm := float64(tr.Loss(ids, tgt, wt))
			p.Data[j] = old
			num := (lp - lm) / (2 * eps)
			diff := math.Abs(num - ana)
			if diff > 2e-3 && diff > 0.1*math.Max(math.Abs(num), math.Abs(ana)) {
				t.Errorf("%s[%d]: numeric %.6f analytic %.6f", p.Name, j, num, ana)
			}
			checked++
		}
	}
	if checked == 0 {
		t.Fatal("nothing checked")
	}
}

func TestTrainingReducesLoss(t *testing.T) {
	cfg := model.Config{Vocab: 8, Dim: 16, Layers: 1, Heads: 2, FF: 32, Ctx: 8}
	rng := rand.New(rand.NewSource(1))
	w := model.New(cfg, rng)
	const B, T = 4, 8
	tr := NewTrainer(w, B, T)
	ids := make([]int32, B*T)
	tgt := make([]int32, B*T)
	wt := make([]float32, B*T)
	for i := range ids { // repeating pattern 3,4,5,6,3,4,5,6...
		ids[i] = int32(3 + i%4)
		tgt[i] = int32(3 + (i+1)%4)
		wt[i] = 1
	}
	first := tr.LossGrad(ids, tgt, wt)
	var last float32
	for s := 0; s < 100; s++ {
		last = tr.LossGrad(ids, tgt, wt)
		tr.Update(2e-2)
	}
	if !(last < first*0.5) {
		t.Fatalf("loss did not drop enough: %.4f -> %.4f", first, last)
	}
}
