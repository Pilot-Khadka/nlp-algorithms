import math


def perplexity(avg_loss=None, **kwargs):
    return {"perplexity": math.exp(avg_loss)}
