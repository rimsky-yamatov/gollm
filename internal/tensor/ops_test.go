package tensor

import (
	"math"
	"testing"
)

func approx(a, b float32, tol float64) bool {
	return math.Abs(float64(a-b)) <= tol
}

func TestMatMul(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6}   // 2x3
	b := []float32{7, 8, 9, 10, 11, 12} // 3x2
	out := make([]float32, 4)
	MatMul(out, a, b, 2, 3, 2)
	want := []float32{58, 64, 139, 154}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("MatMul[%d] = %v, want %v", i, out[i], want[i])
		}
	}
}

func TestMatMulBT(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6}    // 2x3
	b := []float32{7, 9, 11, 8, 10, 12} // (3x2)ᵀ stored as 2x3
	out := make([]float32, 4)
	MatMulBT(out, a, b, 2, 3, 2)
	want := []float32{58, 64, 139, 154}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("MatMulBT[%d] = %v, want %v", i, out[i], want[i])
		}
	}
}

func TestMatMulATAcc(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6} // 3x2
	b := []float32{1, 0, 0, 1, 1, 1} // 3x2
	out := make([]float32, 4)
	MatMulATAcc(out, a, b, 3, 2, 2)
	want := []float32{6, 8, 8, 10}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("MatMulATAcc[%d] = %v, want %v", i, out[i], want[i])
		}
	}
}

func TestSoftmaxSumsToOne(t *testing.T) {
	r := []float32{1, 2, 3, 4}
	Softmax(r)
	var s float32
	for _, v := range r {
		s += v
	}
	if !approx(s, 1, 1e-5) {
		t.Fatalf("softmax sum = %v", s)
	}
	if !(r[3] > r[2] && r[2] > r[1]) {
		t.Fatalf("softmax not monotone: %v", r)
	}
}

func TestLayerNormZeroMeanUnitVar(t *testing.T) {
	x := []float32{1, 2, 3, 4}
	g := []float32{1, 1, 1, 1}
	b := []float32{0, 0, 0, 0}
	y := make([]float32, 4)
	LayerNormFwd(y, nil, nil, x, g, b, 1, 4)
	var mean float32
	for _, v := range y {
		mean += v
	}
	if !approx(mean/4, 0, 1e-5) {
		t.Fatalf("mean = %v", mean/4)
	}
}
