.PHONY: build test chat train corpus golden

build:
	go build -o bin/gollm ./cmd/gollm

test:
	go test ./...

chat: build
	./bin/gollm chat

# Go で最初から学習し直す (CPUコア数に応じて数十分〜)
train: build
	./bin/gollm train -steps 4000

# コーパスの再生成 (python3 が必要)
corpus:
	python3 tools/gen_corpus.py 60000 60

# numpy 参照実装との照合データを作り直す
golden:
	python3 tools/train_numpy.py golden
