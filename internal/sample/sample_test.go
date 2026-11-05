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

func TestTopKOne(t *testing.T) {
	logits := []float32{0, 0, 0, 1, 5, 2}
	o := Options{Temp: 1, TopK: 1, Rng: rand.New(rand.NewSource(1))}
	for i := 0; i < 20; i++ {
		if got := Pick(logits, o); got != 4 {
			t.Fatalf("got %d, want 4", got)
		}
	}
}
