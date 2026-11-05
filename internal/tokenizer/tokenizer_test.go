package tokenizer

import "testing"

func TestRoundTrip(t *testing.T) {
	v := Build([]string{"こんにちは\n", "ユーザー: 元気？"})
	s := "こんにちは、元気？\n"
	ids := v.Encode(s)
	if len(ids) != len([]rune(s)) {
		t.Fatalf("len = %d", len(ids))
	}
	got := v.Decode(ids)
	// "、" is not in the vocabulary, so it must be dropped as <unk>.
	if got != "こんにちは元気？\n" {
		t.Fatalf("decode = %q", got)
	}
}
