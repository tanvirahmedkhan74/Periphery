import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedBCELossWithMaskedTeacher(nn.Module):
    def __init__(self):
        """
        Masked Binary Cross-Entropy Loss for Salient Object Detection.
        This loss aligns student predictions with masked teacher predictions.
        """
        super(MaskedBCELossWithMaskedTeacher, self).__init__()

    def forward(self, student_logits, teacher_logits, ground_truth):
        """
        Compute the BCE loss with masked teacher logits.
        - student_logits: Raw logits from the student model (B, C, H, W).
        - teacher_logits: Raw logits from the teacher model (B, C, H, W).
        - ground_truth: Binary ground truth mask (B, H, W).
        """
        # Resize logits to match the ground truth size
        student_logits_resized = F.interpolate(
            student_logits, size=ground_truth.shape[1:], mode='bilinear', align_corners=False
        )
        teacher_logits_resized = F.interpolate(
            teacher_logits, size=ground_truth.shape[1:], mode='bilinear', align_corners=False
        )

        # Apply the ground truth mask to the teacher logits
        ground_truth = ground_truth.unsqueeze(1)  # Add channel dimension for broadcasting (B, 1, H, W)
        masked_teacher_logits = teacher_logits_resized * ground_truth  # Masked teacher logits

        # Apply sigmoid to logits
        student_probs = torch.sigmoid(student_logits_resized)
        teacher_probs = torch.sigmoid(masked_teacher_logits)

        # Compute BCE loss directly between student probabilities and masked teacher probabilities
        bce_loss = F.binary_cross_entropy(student_probs, teacher_probs, reduction='mean')

        return bce_loss