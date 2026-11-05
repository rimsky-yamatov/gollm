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

// ColSumAcc accumulates the column sums of x[rows*d] into out[d].
func ColSumAcc(out, x []float32, rows, d int) {
	for r := 0; r < rows; r++ {
		xr := x[r*d : (r+1)*d]
		for i, v := range xr {
			out[i] += v
		}
	}
}

// Softmax normalises row in place (numerically stable).
func Softmax(row []float32) {
	mx := row[0]
	for _, v := range row {
		if v > mx {
			mx = v
		}
	}
	var sum float64
	for i, v := range row {
		e := math.Exp(float64(v - mx))
		row[i] = float32(e)
		sum += e
	}
	inv := float32(1 / sum)
	for i := range row {
		row[i] *= inv
	}
}

const lnEps = 1e-5

// LayerNormFwd computes y = LN(x)*g + b for each row. xh and rstd are optional
// caches (pass nil to skip) needed by LayerNormBwd.
func LayerNormFwd(y, xh, rstd, x, g, b []float32, rows, d int) {
	Parallel(rows, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			xr := x[r*d : (r+1)*d]
			var mu float64
			for _, v := range xr {
				mu += float64(v)
			}
			mu /= float64(d)
			var vr float64
			for _, v := range xr {
				c := float64(v) - mu
				vr += c * c
			}
			vr /= float64(d)
			rs := float32(1.0 / math.Sqrt(vr+lnEps))
			if rstd != nil {
				rstd[r] = rs
			}
			m32 := float32(mu)
			for i, v := range xr {
				h := (v - m32) * rs
				if xh != nil {
					xh[r*d+i] = h
				}
				y[r*d+i] = h*g[i] + b[i]
			}
		}
	})
}

// LayerNormBwd accumulates gradients: dx += ∂L/∂x, dg += ∂L/∂g, db += ∂L/∂b.
func LayerNormBwd(dx, dg, db, dy, xh, rstd, g []float32, rows, d int) {
	dxh := make([]float32, d)
	for r := 0; r < rows; r++ {
		dyr := dy[r*d : (r+1)*d]
		xr := xh[r*d : (r+1)*d]
		var m1, m2 float32
		for i := 0; i < d; i++ {
			dxh[i] = dyr[i] * g[i]
			m1 += dxh[i]
			m2 += dxh[i] * xr[i]
			dg[i] += dyr[i] * xr[i]
			db[i] += dyr[i]
		}
		m1 /= float32(d)
		m2 /= float32(d)
		rs := rstd[r]
		out := dx[r*d : (r+1)*d]
		for i := 0; i < d; i++ {
			out[i] += rs * (dxh[i] - m1 - xr[i]*m2)
		}
	}
}
