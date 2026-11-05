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
