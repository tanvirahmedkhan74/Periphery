import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedDistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        """
        Initialize the combined loss function.

        Args:
        - alpha (float): Weight for BCE loss.
        - beta (float): Weight for MSE loss.
        """
        super(CombinedDistillationLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, student_logits, teacher_logits, ground_truth):
        """
        Compute the combined loss.

        Args:
        - student_logits (torch.Tensor): Upsampled logits from the student model. Shape: [B, 1, 256, 256]
        - teacher_logits (torch.Tensor): Upsampled logits from the teacher model. Shape: [B, 1, 256, 256]
        - ground_truth (torch.Tensor): Ground truth saliency maps. Shape: [B, 256, 256]

        Returns:
        - torch.Tensor: Combined loss value.
        """
        # Binary Cross-Entropy Loss with Ground Truth
        bce_loss = F.binary_cross_entropy_with_logits(student_logits, ground_truth.unsqueeze(1))  # Shape: [B,1,256,256]

        # Mean Squared Error Loss between Sigmoid Outputs of Teacher and Student
        student_probs = torch.sigmoid(student_logits)
        teacher_probs = torch.sigmoid(teacher_logits)
        mse_loss = F.mse_loss(student_probs, teacher_probs)

        # Total Loss
        total_loss = self.alpha * bce_loss + self.beta * mse_loss
        return total_loss