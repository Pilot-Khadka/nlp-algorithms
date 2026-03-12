# nlp-algorithms
This repo implements classic and modern nlp algorithms. This project is ongoing as more algorithms, metrics and tasks are added.

## components:
- [x] Tokenization (whitespace, BPE)
- [x] Stemming and Lemmatization
- [x] Vectorization (One-hot, N-gram, TF-IDF, Word2vec)
- [x] RNN-family architectures
- [x] Transformer models
  - [x] Sequence-to-Sequence
  - [ ] Encoder only
  - [ ] Decoder only
- [x] NLP tasks
  - [x] Classification
  - [x] Language Modeling
  - [ ] Machine Translation (ongoing)

## architectures

| Architecture                | Status        |
| --------------------------- | ------------- |
| RNN (vanilla, stacked)      | Implemented   |
| GRU / Bi-GRU                | Implemented   |
| LSTM / Bi-LSTM              | Implemented   |
| AWD-LSTM                    | Implemented   |
| Transformer Attention       | Implemented   |
| Transformer Encoder only model         | Planned   |
| Transformer Encoder-Decoder | Implemented   |
| Transformer Decoder-only model    | Planned       |
| Hybrid Architectures        | Planned  |

## datasets

- [x] Penn treebank (language modeling)
- [x] Imdb (classification)
- [x] Tatoeba (machine translation)
- [x] Huggingface datasets

## model comparison
Comparison between custom model implementation and PyTorch equivalents

### classification (imdb dataset)

| Model      | Implementation | Train Loss | Train Time (avg) | Valid Loss | Valid Time (avg) | Accuracy | Precision | Recall | F1     |
| -----------|----------------| ---------- | ---------------- | ---------- | ---------------- | -------- | --------- | ------ | ------ |
| **BiGRU**  |Pytorch         | 0.0143     | 34.39s           | 0.9941     | 11.77s           | 0.8162   | 0.8172    | 0.8162 | 0.8167 |
| **BiGRU**  |Custom          | 0.0125     | 382.14s          | 1.3821     | 111.09s          | 0.8086   | 0.8102    | 0.8086 | 0.8094 |
| **GRU**    |Pytorch         | 0.0154     | 16.70s           | 1.3159     | 5.40s            | 0.7993   | 0.8028    | 0.7993 | 0.8010 |
| **GRU**    |Custom          | 0.0175     | 157.92s          | 1.1263     | 49.58s           | 0.8109   | 0.8109    | 0.8109 | 0.8109 |
| **BiLSTM** |Pytorch         | 0.0239     | 40.11s           | 1.3207     | 14.41s           | 0.7825   | 0.7840    | 0.7825 | 0.7832 |
| **BiLSTM** |Custom          | 0.0203     | 365.49s          | 1.1990     | 88.54s           | 0.7948   | 0.7951    | 0.7948 | 0.7949 |
| **LSTM**   |Pytorch         | 0.0540     | 19.41s           | 1.1087     | 7.28s            | 0.7839   | 0.7840    | 0.7839 | 0.7840 |
| **LSTM**   |Custom          | 0.0266     | 176.13s          | 1.5475     | 48.30s           | 0.7376   | 0.7377    | 0.7376 | 0.7377 |
| **QRNN**     | Custom         | 0.0106     | 110.53s          | 1.7951     | 15.72s           | 0.7281   | 0.7290    | 0.7281 | 0.7286 |
| **AWD-LSTM** | Custom         | 0.0806     | 292.92s          | 0.8338     | 45.44s           | 0.8020   | 0.8028    | 0.8020 | 0.8024 |

**Notes**

- **PyTorch implementations are ~8-10x faster** due to optimized CUDA kernels and fused operations.
- RNN / BiRNN models for both custom and pytorch were excluded since they converged to ~50% accuracy (random).

## roadmap
- [ ] training seq-2-seq model on machine translation
- [ ] benchmarking against pytorch/official models
- [ ] add BERT as encoder only model
- [ ] add GPT-2 as decoder only model
