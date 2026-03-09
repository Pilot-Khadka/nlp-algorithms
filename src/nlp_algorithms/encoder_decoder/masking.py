from ..attention import (
    convert_to_additive,
    make_causal_mask,
    make_padding_mask,
)


def create_masks(src, tgt, pad_idx):
    """
    description:
        encoder self-attention: padding mask only (src)
        decoder self-attention: causal+padding mask (tgt)
        cross-attention: padding mask for encoder source tokens (src)

    outputs:
        src padding mask, tgt combined mask

    """
    B, T = tgt.shape
    src_pad = make_padding_mask(seq=src, pad_idx=pad_idx).to(tgt.device)
    tgt_pad = make_padding_mask(seq=tgt, pad_idx=pad_idx).to(tgt.device)
    causal_mask = make_causal_mask(T).to(tgt.device)  # True = keep

    src_padding_mask = convert_to_additive(src_pad)
    tgt_combined_mask = convert_to_additive(tgt_pad & causal_mask)
    return src_padding_mask, tgt_combined_mask
