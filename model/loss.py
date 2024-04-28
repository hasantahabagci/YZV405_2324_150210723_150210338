import torch
import torch.nn as nn


def cross_entropy_loss(logits, targets):
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    loss = -target_log_probs.mean()
    return loss