import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', ignore_index=None):
        """
        Focal Loss implementation with ignore index.

        :param alpha: Balance factor for positive examples (default: 1)
        :param gamma: Focusing parameter to down-weight easy examples (default: 2)
        :param reduction: Reduction method to apply to the loss (default: 'mean')
        :param ignore_index: Index of the token to be ignored in the loss computation
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        Forward pass for the Focal Loss.
        
        :param inputs: Predicted logits [batch_size, num_classes, seq_len]
        :param targets: Ground truth labels [batch_size, seq_len]
        """
        # Ensure inputs are in the shape [batch_size * seq_len, num_classes]
        inputs = inputs.transpose(1, 2).contiguous()
        inputs = inputs.view(-1, inputs.size(2))
        
        # Flatten targets to [batch_size * seq_len]
        targets = targets.view(-1)

        # Get a mask for valid targets (ignoring ignore_index)
        valid_mask = (targets != self.ignore_index)

        # Filter out padding tokens
        inputs = inputs[valid_mask]
        targets = targets[valid_mask]

        # Convert inputs to probabilities (softmax for multi-class problems)
        probs = F.softmax(inputs, dim=-1)
        
        # Create one-hot encoded targets
        targets_one_hot = torch.zeros_like(probs).scatter(1, targets.unsqueeze(1), 1)
        
        # Get the probabilities for the true class
        pt = (probs * targets_one_hot).sum(dim=1)

        # Calculate the focal weight
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        # Calculate cross-entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Apply the focal weight to the cross-entropy loss
        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean() if focal_loss.numel() > 0 else torch.tensor(0.0)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss