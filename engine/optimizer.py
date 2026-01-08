import torch


def get_optimizer(model, cfg):
    opt_type = getattr(cfg.train, "optimizer", "adamw").lower()
    lr = cfg.train.learning_rate
    wd = getattr(cfg.train, "weight_decay", 0.01)

    if opt_type == "asgd":
        optimizer = torch.optim.ASGD(
            model.parameters(), lr=lr, t0=0, lambd=0.0, weight_decay=wd
        )
    elif opt_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    return optimizer
