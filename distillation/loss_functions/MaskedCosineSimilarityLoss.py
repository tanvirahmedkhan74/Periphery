import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCosineSimilarityLoss(nn.Module):
    def __init__(self):
        """
        Masked Cosine Similarity Loss for Salient Object Detection.
        This loss aligns the student and teacher predictions in the salient regions
        defined by the binary ground truth mask.
        """
        super(MaskedCosineSimilarityLoss, self).__init__()

    def forward(self, student_logits, teacher_logits, ground_truth):
        """
        Compute the masked cosine similarity loss.
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

        # Apply the ground truth mask
        ground_truth = ground_truth.unsqueeze(1)  # Add channel dimension for broadcasting (B, 1, H, W)
        student_masked = student_logits_resized * ground_truth
        teacher_masked = teacher_logits_resized * ground_truth

        # Compute cosine similarity
        # Flatten the masked tensors along spatial dimensions for batch-wise computation
        student_flat = student_masked.view(student_masked.size(0), -1)
        teacher_flat = teacher_masked.view(teacher_masked.size(0), -1)

        # Normalize the flattened tensors to unit vectors
        student_norm = F.normalize(student_flat, p=2, dim=1)
        teacher_norm = F.normalize(teacher_flat, p=2, dim=1)

        # Compute cosine similarity loss (1 - cosine similarity mean)
        cosine_loss = 1 - torch.mean(torch.sum(student_norm * teacher_norm, dim=1))

        return cosine_loss