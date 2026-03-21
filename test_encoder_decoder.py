from os.path import exists

import torch
import sacrebleu


from nlp_algorithms.encoder_decoder.seq2seq import make_model
from nlp_algorithms.encoder_decoder.train import greedy_decode
from nlp_algorithms.encoder_decoder.data import (
    load_vocab,
    load_multi30k,
    Batch,
    create_dataloaders,
    collate_batch,
    HFDatasetWrapper,
)
from torch.utils.data import DataLoader


def load_trained_model():
    vocab_src, vocab_tgt = load_vocab()
    config = {
        "batch_size": 32,
        "distributed": False,
        "num_epochs": 8,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "num_layers": 6,
        "d_model": 512,
        "file_prefix": "multi30k_model_",
    }
    model_path = "multi30k_model_final.pt"
    if not exists(model_path):
        raise RuntimeError(f"Model at {model_path} not found")

    model = make_model(
        len(vocab_src),
        len(vocab_tgt),
        N=config["num_layers"],
        d_model=config["d_model"],
    )
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
    )
    return model, vocab_src, vocab_tgt


def check_outputs(
    valid_dataloader,
    model,
    vocab_src,
    vocab_tgt,
    n_examples=15,
    pad_idx=2,
    eos_string="</s>",
):
    results = []
    for idx in range(n_examples):
        print("\nExample %d ========\n" % idx)
        b = next(iter(valid_dataloader))
        rb = Batch(b[0], b[1], pad_idx)

        src_tokens = [vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx]
        tgt_tokens = [vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx]

        print("Source Text (Input)        : " + " ".join(src_tokens).replace("\n", ""))
        print("Target Text (Ground Truth) : " + " ".join(tgt_tokens).replace("\n", ""))
        model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]
        model_txt = (
            " ".join(
                [vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
            ).split(eos_string, 1)[0]
            + eos_string
        )
        print("Model Output               : " + model_txt.replace("\n", ""))
        results.append((rb, src_tokens, tgt_tokens, model_out, model_txt))
    return results


def calculate_bleu(
    model,
    vocab_src,
    vocab_tgt,
    pad_idx=2,
    max_padding=72,
    eos_string="</s>",
):
    device = torch.device("cpu")

    def collate_fn(batch):
        return collate_batch(
            batch,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src["<blank>"],
        )

    _, _, test_data = load_multi30k()
    test_dataloader = DataLoader(
        HFDatasetWrapper(test_data),
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
    )

    hypotheses = []
    references = []

    model.eval()
    with torch.no_grad():
        for b in test_dataloader:
            rb = Batch(b[0], b[1], pad_idx)
            for i in range(rb.src.shape[0]):
                src = rb.src[i].unsqueeze(0)
                src_mask = rb.src_mask[i].unsqueeze(0)

                model_out = greedy_decode(
                    model, src, src_mask, max_padding, start_symbol=0
                )[0]
                hypothesis = (
                    " ".join(
                        [vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
                    )
                    .split(eos_string, 1)[0]
                    .strip()
                )

                reference = (
                    " ".join(
                        [vocab_tgt.get_itos()[x] for x in rb.tgt[i] if x != pad_idx]
                    )
                    .split(eos_string, 1)[0]
                    .strip()
                )

                hypotheses.append(hypothesis)
                references.append(reference)

    result = sacrebleu.corpus_bleu(hypotheses, [references])
    return result


def run_model_example(n_examples=5):
    print("Loading vocabulary ...")
    model, vocab_src, vocab_tgt = load_trained_model()

    print("Preparing validation dataloader ...")
    _, valid_dataloader = create_dataloaders(
        torch.device("cpu"),
        vocab_src,
        vocab_tgt,
        batch_size=1,
        is_distributed=False,
    )

    print("Checking Model Outputs:")
    example_data = check_outputs(
        valid_dataloader, model, vocab_src, vocab_tgt, n_examples=n_examples
    )

    print("\nComputing BLEU score on test set ...")
    bleu = calculate_bleu(model, vocab_src, vocab_tgt)
    print(f"Test BLEU Score: {bleu.score:.2f}")

    return model, example_data, bleu


if __name__ == "__main__":
    run_model_example()
