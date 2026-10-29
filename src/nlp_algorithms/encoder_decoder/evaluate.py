import glob

import torch
import sacrebleu

from nlp_algorithms.encoder_decoder.seq2seq import make_model
from nlp_algorithms.encoder_decoder.train import greedy_decode, beam_search
from nlp_algorithms.encoder_decoder.data import collate_batch


def average_checkpoints(paths):
    # average of the last epochs, usually better than a single checkpoint
    states = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
    return {k: sum(s[k].float() for s in states) / len(states) for k in states[0]}


def last_checkpoints(prefix, keep_last):
    # train_model keeps the last keep_last epoch checkpoints
    return sorted(glob.glob(f"{prefix}[0-9][0-9].pt"))[-keep_last:]


def load_model(config, vocab_src, vocab_tgt, paths, device):
    # several paths -> average the weights
    model = make_model(
        len(vocab_src),
        len(vocab_tgt),
        N=config["num_layers"],
        d_model=config["d_model"],
        shared_embeddings=config.get("shared_embeddings", False),
    )
    model.load_state_dict(average_checkpoints(paths))
    return model.to(device).eval()


def translate(
    model,
    rows,
    vocab_src,
    vocab_tgt,
    max_len,
    beam_size=1,
    len_penalty=1.0,
    device=None,
):
    if device is None:
        device = next(model.parameters()).device
    pad_idx = vocab_src["<blank>"]
    hypotheses = []
    with torch.no_grad():
        for row in rows:
            src, _ = collate_batch(
                [row], vocab_src, vocab_tgt, device, max_padding=max_len, pad_id=pad_idx
            )
            src_mask = (src != pad_idx).unsqueeze(-2)
            if beam_size > 1:
                out = beam_search(
                    model,
                    src,
                    src_mask,
                    max_len,
                    start_symbol=0,
                    beam_size=beam_size,
                    len_penalty=len_penalty,
                )[0]
            else:
                out = greedy_decode(model, src, src_mask, max_len, start_symbol=0)[0]
            # decode drops <s>, </s> and <blank>
            hypotheses.append(vocab_tgt.decode(out.tolist()).strip())
    return hypotheses


def evaluate_bleu(
    config, vocab_src, vocab_tgt, test_rows, runs, len_penalty=1.2, device=None
):
    # runs: {name: (checkpoint paths, beam size)}
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    references = list(test_rows["tgt"])
    scores, outputs = {}, {}
    for name, (paths, beam_size) in runs.items():
        model = load_model(config, vocab_src, vocab_tgt, paths, device)
        hyps = translate(
            model,
            test_rows,
            vocab_src,
            vocab_tgt,
            config["max_padding"],
            beam_size=beam_size,
            len_penalty=len_penalty,
            device=device,
        )
        bleu = sacrebleu.corpus_bleu(hyps, [references])
        print(f"{name}: {bleu}", flush=True)
        scores[name], outputs[name] = bleu.score, hyps
    return scores, outputs
