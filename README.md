# nlp-algorithms
This repo implements classic and modern nlp algorithms. This project is ongoing as more algorithms, metrics and tasks are added.

## components:
- [x] Tokenization (whitespace, BPE)
- [x] Stemming and Lemmatization
- [x] Vectorization (One-hot, N-gram, TF-IDF, Word2vec)
- [x] RNN-family architectures
- [x] Transformer models
  - [x] Sequence-to-Sequence
  - [] Encoder only
  - [] Decoder only
- [x] NLP tasks
  - [x] Classification
  - [x] Language Modeling
  - [] Machine Translation (ongoing)

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

| Model  | Library | Accuracy | Training Speed |
| ------ | ------- | -------- | -------------- |
| RNN    | Custom  | —        | —              |
| RNN    | PyTorch | —        | —              |
| LSTM   | Custom  | —        | —              |
| LSTM   | PyTorch | —        | —              |
| BiLSTM | Custom  | —        | —              |
| BiLSTM | PyTorch | —        | —              |

## goals
- [ ] training seq-2-seq model on machine translation
- [ ] benchmarking against pytorch/official models
- [ ] add BERT as encoder only model
- [ ] add GPT-2 as decoder only model
