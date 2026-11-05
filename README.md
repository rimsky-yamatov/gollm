# gollm

日本語で会話できる小さな言語モデルを Go で実装したものです。
文字単位の Transformer(3層・128次元・約70万パラメータ)で、学習用の対話コーパスと学習済み重みを同梱しています。

## 特徴

- 依存ライブラリなし(Go標準ライブラリのみ)
- 学習(手書き backprop + AdamW)と推論(KVキャッシュ)の両方が Go 製
- 学習データ `data/corpus/` に約6万会話・約700万文字を同梱
- 学習済み重み `model/gollm.bin`(約2.7MB)ですぐ会話できる

## 使い方

```sh
go run ./cmd/gollm chat          # 対話
go run ./cmd/gollm gen -prompt "こんにちは"
go run ./cmd/gollm train -steps 4000   # Goで最初から学習
go test ./...
```

