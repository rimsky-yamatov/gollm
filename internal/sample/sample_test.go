package sample

import (
	"math/rand"
	"testing"
)

func TestGreedySkipsReserved(t *testing.T) {
	logits := []float32{100, 100, 100, 1, 5, 2}
	if got := Pick(logits, Greedy(10)); got != 4 {
		t.Fatalf("got %d, want 4", got)
	}
}
