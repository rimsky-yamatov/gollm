package tensor

import (
	"math"
	"testing"
)

func approx(a, b float32, tol float64) bool {
	return math.Abs(float64(a-b)) <= tol
}
