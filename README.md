# nlp-algorithms
This repo implements classic and modern nlp algorithms. This project is ongoing as more algorithms, metrics and tasks are added.

## components:
- [x] Tokenization (whitespace, BPE)
- [x] Stemming and Lemmatization
- [x] Vectorization (One-hot, N-gram, TF-IDF, Word2vec)
- [x] RNN-family architectures
- [x] Transformer models
  - [x] Sequence-to-Sequence
  - [x] Encoder only (BERT)
  - [ ] Decoder only
- [x] NLP tasks
  - [x] Classification
  - [x] Language Modeling
  - [x] Machine Translation

## architectures

| Architecture                | Status        |
| --------------------------- | ------------- |
| RNN (vanilla, stacked)      | Implemented   |
| GRU / Bi-GRU                | Implemented   |
| LSTM / Bi-LSTM              | Implemented   |
| AWD-LSTM                    | Implemented   |
| Transformer Attention       | Implemented   |
| Transformer Encoder only model (BERT) | Implemented   |
| Transformer Encoder-Decoder | Implemented   |
| Transformer Decoder-only model    | Planned       |

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

### bert (imdb dataset)

Pretraining (masked language modeling + next sentence prediction) on the imdb reviews, then fine-tuning on imdb sentiment classification.

| Stage       | Config                                                                                           | Data                                   | MLM Loss | MLM Accuracy | MLM Perplexity | NSP Accuracy | Train Time |
| ----------- | ------------------------------------------------------------------------------------------------ | -------------------------------------- | -------- | ------------ | -------------- | ------------ | ---------- |
| Pretraining | 4 layers, d=256, FF=1024, 4 heads, 5.85M params, byte-level BPE 10k, SeqLen=256, BS=128, 150 epochs | imdb train + unsupervised (75k reviews) | 2.94     | 0.446        | 18.9           | 0.891        | 2h36m (2x T4) |

| Model                  | Implementation                                                                                     | Config (Key Hyperparams)                                          | Pretraining Data                          | Test Accuracy |
| ---------------------- | -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- | ----------------------------------------- | ------------- |
| **BERT-Large**         | [reported score from Yang et al. 2019 (XLNet), Table 4](https://arxiv.org/abs/1906.08237)          | 24 layers, d=1024, 340M params                                    | BooksCorpus + Wikipedia                   | 95.49         |
| **BERT-Base**          | [reported score from Sun et al. 2019, Table 5](https://arxiv.org/abs/1905.05583)                   | 12 layers, d=768, 110M params, SeqLen=512 (head+tail), lr 2e-5    | BooksCorpus + Wikipedia                   | 94.60         |
| **BERT-Base**          | [reported score from Sun et al. 2019, Table 5](https://arxiv.org/abs/1905.05583)                   | same, further pretrained on the imdb train set for 100k steps     | BooksCorpus + Wikipedia, then imdb        | **95.63**     |
| **BERT (small)**       | Custom                                                                                             | 4 layers, d=256, 5.85M params, SeqLen=256 (head only), lr 1e-4, 4 epochs | imdb only (75k reviews)            | **89.65** |

**Notes**

- The reported scores are given as test error rates in the papers (4.51, 5.40 and 4.37), shown here as accuracy.
- The custom model is ~19x smaller than BERT-Base, sees only the first 254 tokens of a review (imdb reviews average ~300 word pieces) and is pretrained on ~100M characters of movie reviews instead of 3.3B words.
- There is no imdb validation split. The custom score is the final model after 4 epochs (89.83 at epoch 2, the best epoch picked on the test set). Fine-tuning takes 2.6 min on 2x T4.
- With the same imdb data the small pretrained BERT beats the best RNN above (BiGRU, 81.62) by 8 points.

### language modeling (penn treebank)

| Model            | Implementation               | Config                                                                               | Epoch     | Test Perplexity | Train Time / Epoch |
| ---------------- | ---------------------------- | ------------------------------------------------------------------------------------ | --------- | --------------- | ------------------ |
| **AWD-LSTM**     | Official (no finetuning)     | BS=20, SeqLen=70, Dropouti=0.4, Dropouth=0.25                                        | up to 500 | **58.8**        | 65s                |
| **AWD-LSTM**     | Custom                       | BS=32, SeqLen=70, Dropouti=0.4, Dropouth=0.25, 3-layer, 1000 hidden                  | 60        | 86.42           | 156s               |
| **QRNN**         | Official (no finetuning)     | BS=20, SeqLen=70, Dropouti=0.4, Dropouth=0.3, 4-layer, 1550 hidden, emsize=400       | up to 550 | 58.3            | 28s                |
| **QRNN**         | Custom                       | BS=32, SeqLen=70, Dropouti=0.4, Dropouth=0.25, 3-layer, 1000 hidden, locked dropout | 60        | 113.21          | 118.69s            |
| **GRU**          | Custom (locked drop, embed drop, variational drop) | BS=16, SeqLen=70, Dropouti=0.4, EmbDrop=0.1, Dropouth=0.25, Dropout=0.4, 3-layer, 1000 hidden | 60 | **91.55** | 296s |
| **GRU**          | PyTorch (weight drop, locked drop, embed drop)     | BS=16, SeqLen=70, Dropouti=0.4, EmbDrop=0.1, Dropouth=0.25, Dropout=0.4, 3-layer, 1000 hidden | 60 | **91.61** | 101s |

### machine translation (multi30k en->de)

| Model                | Implementation               | Config (Key Hyperparams)                                              | Dataset        | Test BLEU |
| -------------------- | ---------------------------- | --------------------------------------------------------------------- | -------------- | --------- |
| **Transformer-Base** | [reported score from hyunwoongko/transformer](https://github.com/hyunwoongko/transformer) | 6L/6L, d=512, FF=2048, 8 heads, dropout=0.1, sinusoidal PE, BPE       | Multi30k En–De | **26.4**  |
| **Transformer-Base** | Custom                | 6L/6L, d=512, FF=2048, 8 heads, dropout=0.1, **RoPE**, byte-level BPE | Multi30k En–De | **23.91** (greedy decoding) |

* Reported score uses the its repository’s custom BLEU calculation.
* This repos custom implementation uses **sacreBLEU** for BLEU calculation.

### machine translation (nepali->english)

| Model                | Implementation               | Config (Key Hyperparams)                                              | Dataset        | Test BLEU |
| -------------------- | ---------------------------- | --------------------------------------------------------------------- | -------------- | --------- |
| **Transformer**      | [reported score from Duwal & Bal 2019](https://huggingface.co/datasets/sharad461/ne-en-parallel-208k) (fairseq) | 5L/5L, d=512, FF=2048, 8 heads, dropout=0.3, sinusoidal PE, joint SentencePiece BPE 2500, beam 5 | ne-en 208k, FLORES v1 devtest | **13.0** |
| **Transformer**      | Custom                | 5L/5L, d=512, FF=2048, 8 heads, dropout=0.3, **RoPE**, joint byte-level BPE 4000, shared embeddings, ~16k tokens/batch, 100 epochs | ne-en 208k, FLORES+ devtest | 11.16 (greedy decoding) |
| **Transformer**      | Custom                | same, last 5 checkpoints averaged, beam 5                             | ne-en 208k, FLORES+ devtest | **13.30** |

* Both are trained on the same 208k parallel sentences (no back-translated data).
* The reported score is on FLORES v1 devtest, this repo uses FLORES+ devtest (different sentences), so the scores are close but not the same test set.
* Nepali text is normalized (Devanagari digits to ASCII, danda spacing, zero-width characters) and duplicate pairs are removed before training.

## roadmap
- [x] training seq-2-seq model on machine translation
- [ ] benchmarking against pytorch/official models
- [x] add BERT as encoder only model
- [ ] add GPT-2 as decoder only model
