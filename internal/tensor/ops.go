// Package tensor provides the small set of float32 kernels used by gollm.
// All matrices are dense, row-major []float32.
package tensor

import (
	"math"
	"runtime"
	"sync"
)

// Parallel splits [0,n) into contiguous chunks and runs fn on each chunk
// in its own goroutine. fn must only write to data owned by its chunk.
func Parallel(n int, fn func(lo, hi int)) {
	w := runtime.NumCPU()
	if w > n {
		w = n
	}
	if w <= 1 {
		fn(0, n)
		return
	}
	chunk := (n + w - 1) / w
	var wg sync.WaitGroup
	for lo := 0; lo < n; lo += chunk {
		hi := lo + chunk
		if hi > n {
			hi = n
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			fn(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}
