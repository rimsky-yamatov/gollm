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

// MatMul computes out[m*n] = a[m*k] · b[k*n] (out is overwritten).
func MatMul(out, a, b []float32, m, k, n int) {
	Parallel(m, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			o := out[i*n : (i+1)*n]
			for j := range o {
				o[j] = 0
			}
			ar := a[i*k : (i+1)*k]
			for p, av := range ar {
				if av == 0 {
					continue
				}
				br := b[p*n : (p+1)*n]
				for j, bv := range br {
					o[j] += av * bv
				}
			}
		}
	})
}

// MatMulBT computes out[m*n] = a[m*k] · bᵀ where b is stored as [n*k].
func MatMulBT(out, a, b []float32, m, k, n int) {
	Parallel(m, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			ar := a[i*k : (i+1)*k]
			for j := 0; j < n; j++ {
				br := b[j*k : (j+1)*k]
				var s float32
				for p, av := range ar {
					s += av * br[p]
				}
				out[i*n+j] = s
			}
		}
	})
}

// MatMulATAcc accumulates out[k*n] += aᵀ · b where a is [m*k] and b is [m*n].
func MatMulATAcc(out, a, b []float32, m, k, n int) {
	Parallel(k, func(lo, hi int) {
		for r := 0; r < m; r++ {
			br := b[r*n : (r+1)*n]
			for i := lo; i < hi; i++ {
				av := a[r*k+i]
				if av == 0 {
					continue
				}
				o := out[i*n : (i+1)*n]
				for j, bv := range br {
					o[j] += av * bv
				}
			}
		}
	})
}

// AddInPlace performs dst[i] += src[i].
func AddInPlace(dst, src []float32) {
	for i, v := range src {
		dst[i] += v
	}
}

// Zero sets every element of a to 0.
func Zero(a []float32) {
	for i := range a {
		a[i] = 0
	}
}

// AddBias adds the vector b (length d) to each of the rows of x.
func AddBias(x, b []float32, rows, d int) {
	for r := 0; r < rows; r++ {
		xr := x[r*d : (r+1)*d]
		for i, bv := range b {
			xr[i] += bv
		}
	}
}
