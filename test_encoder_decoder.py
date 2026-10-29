import glob
from os.path import exists

import torch
import sacrebleu


from nlp_algorithms.encoder_decoder.seq2seq import make_model
from nlp_algorithms.encoder_decoder.train import greedy_decode, beam_search
from nlp_algorithms.encoder_decoder.data import (
    load_vocab,
    load_ne_en,
    Batch,
    create_dataloaders,
    collate_batch,
)


def average_checkpoints(paths, out_path):
    # average of the last epochs, usually better than a single checkpoint
    states = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
    avg = {k: sum(s[k].float() for s in states) / len(states) for k in states[0]}
    torch.save(avg, out_path)
    return avg


def load_trained_model(prefix="checkpoint/ne_en/model_", keep_last=5):
    vocab_src, vocab_tgt = load_vocab(
        load_fn=load_ne_en, vocab_path="checkpoint/ne_en/vocab.pt"
    )
    # same model as train_encoder_decoder.py
    config = {
        "num_layers": 5,
        "d_model": 512,
        "shared_embeddings": True,
    }

    model_path = f"{prefix}avg.pt"
    if exists(model_path):
        state = torch.load(model_path, map_location="cpu", weights_only=True)
    else:
        # train_model keeps the last keep_last epoch checkpoints
        paths = sorted(glob.glob(f"{prefix}[0-9][0-9].pt"))[-keep_last:]
        if not paths:
            raise RuntimeError(f"No checkpoints found at {prefix}*.pt")
        print("Averaging:", paths)
        state = average_checkpoints(paths, model_path)

    model = make_model(
        len(vocab_src),
        len(vocab_tgt),
        N=config["num_layers"],
        d_model=config["d_model"],
        shared_embeddings=config["shared_embeddings"],
    )
    model.load_state_dict(state)
    return model, vocab_src, vocab_tgt


def check_outputs(
    valid_dataloader,
    model,
    vocab_src,
    vocab_tgt,
    n_examples=15,
    pad_idx=2,
    max_padding=256,
    eos_string="</s>",
):
    results = []
    for idx in range(n_examples):
        b = next(iter(valid_dataloader))
        rb = Batch(b[0], b[1], pad_idx)

        src_ids = [x for x in rb.src[0].tolist() if x != pad_idx]
        tgt_ids = [x for x in rb.tgt[0].tolist() if x != pad_idx]
        src_tokens = [vocab_src.get_itos()[x] for x in src_ids]
        tgt_tokens = [vocab_tgt.get_itos()[x] for x in tgt_ids]

        print(
            "Source Text (Input)        : "
            + vocab_src.decode(src_ids).replace("\n", "")
        )
        print(
            "Target Text (Ground Truth) : "
            + vocab_tgt.decode(tgt_ids).replace("\n", "")
        )
        model_out = greedy_decode(model, rb.src, rb.src_mask, max_padding, 0)[0]
        model_ids = [x for x in model_out.tolist() if x != pad_idx]
        model_txt = vocab_tgt.decode(model_ids).split(eos_string, 1)[0]
        print("Model Output               : " + model_txt.replace("\n", ""))
        results.append((rb, src_tokens, tgt_tokens, model_out, model_txt))
    return results


def calculate_bleu(
    model,
    vocab_src,
    vocab_tgt,
    max_padding=256,
    beam_size=5,
    len_penalty=1.2,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pad_idx = vocab_src["<blank>"]

    # flores devtest, the source is normalized by load_ne_en, the references are not
    _, _, test_data = load_ne_en()
    references = list(test_data["tgt"])
    hypotheses = []

    model.to(device).eval()
    with torch.no_grad():
        for row in test_data:
            src, _ = collate_batch(
                [row],
                vocab_src,
                vocab_tgt,
                device,
                max_padding=max_padding,
                pad_id=pad_idx,
            )
            src_mask = (src != pad_idx).unsqueeze(-2)
            if beam_size > 1:
                model_out = beam_search(
                    model,
                    src,
                    src_mask,
                    max_padding,
                    start_symbol=0,
                    beam_size=beam_size,
                    len_penalty=len_penalty,
                )[0]
            else:
                model_out = greedy_decode(
                    model, src, src_mask, max_padding, start_symbol=0
                )[0]
            # decode drops <s>, </s> and <blank>
            hypotheses.append(vocab_tgt.decode(model_out.tolist()).strip())

    result = sacrebleu.corpus_bleu(hypotheses, [references])
    return result


def run_model_example(n_examples=5):
    model, vocab_src, vocab_tgt = load_trained_model()

    _, valid_dataloader = create_dataloaders(
        device=torch.device("cpu"),
        vocab_src=vocab_src,
        vocab_tgt=vocab_tgt,
        batch_size=1,
        is_distributed=False,
        load_fn=load_ne_en,
    )

    model.eval()
    example_data = check_outputs(
        valid_dataloader, model, vocab_src, vocab_tgt, n_examples=n_examples
    )

    bleu = calculate_bleu(model, vocab_src, vocab_tgt)
    print(f"Test BLEU Score: {bleu.score:.2f}")
    print(bleu)


if __name__ == "__main__":
    run_model_example()
