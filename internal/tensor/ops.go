// Package tensor provides the small set of float32 kernels used by gollm.
// All matrices are dense, row-major []float32.
package tensor

import (
	"math"
	"runtime"
	"sync"
)
