import os
import copy
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.nn as nn

from torchvision import transforms
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim

from evaluation.metrics.iou import compute_iou
from evaluation.metrics.pixel_acc import compute_pixel_accuracy

from distillation.loss_functions.CombinedDistillationLoss import CombinedDistillationLoss

def periphery_distillation_training(teacher_model, student_model, trainloader, config, valloader=None):
    """
    Trains the student model using knowledge distillation from a teacher model
    with a combined loss for saliency object detection.

    Args:
    - teacher_model (nn.Module): Pre-trained teacher model.
    - student_model (nn.Module): Student model to be trained.
    - trainloader (DataLoader): Training data loader.
    - config (dict): Configuration dictionary containing hyperparameters.
    - valloader (DataLoader, optional): Validation data loader.
    """
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model.to(device).eval()
    student_model.to(device).train()

    # Initialize loss function and optimizer
    criterion = CombinedDistillationLoss(alpha=config.get("alpha", 0.7),
                                        beta=config.get("beta", 0.3))
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=config["learning_rate"])

    # Learning Rate Scheduler with Warm-Up and Cosine Annealing
    warmup_steps = config.get("warmup_steps", 500)
    base_lr = config.get("base_learning_rate", 1e-5)
    max_lr = config["learning_rate"]

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max_lr - base_lr)
        return 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min((step+1)/warmup_steps, 1.0))
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"] * len(trainloader))

    # Early stopping variables
    best_loss = float('inf')
    patience_counter = 0
    best_model_weights = copy.deepcopy(student_model.state_dict())

    # Training loop
    global_step = 0
    all_batch_losses = []  # Store batch losses for graphing

    for epoch in range(config["epochs"]):
        student_model.train()
        running_loss = 0.0
        running_iou = 0.0
        running_pixel_acc = 0.0
        epoch_losses = []  # Track losses for this epoch

        progress_bar = tqdm(trainloader, desc=f"Epoch {epoch + 1}/{config['epochs']}")

        for batch_idx, batch in enumerate(progress_bar):
            # Assuming batch is a tuple/list where:
            # batch[0]: inputs
            # batch[5]: ground truth saliency maps
            inputs, gt_labels = batch[0].to(device), batch[5].to(device).float()
            scribbles = batch[2].to(device)

            # Forward pass through the teacher model
            with torch.no_grad():
                teacher_logits= teacher_model(inputs)  # Teacher's forward method returns logits and possibly other outputs

            # Forward pass through the student model
            student_logits = student_model(inputs)  # Student's forward method returns logits

            # Upsample logits to match ground truth size
            teacher_logits_upsampled = F.interpolate(teacher_logits, size=gt_labels.shape[1:], mode='bilinear', align_corners=False)
            student_logits_upsampled = F.interpolate(student_logits, size=gt_labels.shape[1:], mode='bilinear', align_corners=False)

            # Compute distillation loss
            loss = criterion(student_logits_upsampled, teacher_logits_upsampled, gt_labels)

            # Compute IoU and Pixel Accuracy
            student_probs = torch.sigmoid(student_logits_upsampled)
            student_binary = (student_probs > 0.5).float()

            iou = compute_iou(student_binary, gt_labels.unsqueeze(1))
            pixel_acc = compute_pixel_accuracy(student_binary, gt_labels.unsqueeze(1))

            # Update running metrics
            running_loss += loss.item()
            running_iou += iou.mean().item()
            running_pixel_acc += pixel_acc.mean().item()
            epoch_losses.append(loss.item())
            current_lr = optimizer.param_groups[0]['lr']

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Adjust learning rate
            if global_step < warmup_steps:
                warmup_scheduler.step()
            else:
                main_scheduler.step()

            global_step += 1

            # Update tqdm description
            progress_bar.set_postfix({
                "Total Loss": f"{running_loss / (batch_idx + 1):.4f}",
                "IoU": f"{running_iou / (batch_idx + 1):.4f}",
                "Pixel Acc": f"{running_pixel_acc / (batch_idx + 1):.4f}",
                "lr": f"{current_lr:.8f}"
            })

            # Save visualization for the first batch of each epoch
            if batch_idx == 0:
                save_visualization(
                    inputs,
                    student_logits_upsampled,
                    teacher_logits_upsampled,
                    output_dir="./outputs/visualizations",
                    epoch=epoch,
                    ground_truth=gt_labels
                )  # Assuming save_visualization is defined

        # Store all batch losses for graphing
        all_batch_losses.extend(epoch_losses)

        # Save loss graph every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_loss_graph(all_batch_losses, epoch + 1, output_dir="./outputs/checkpoints")  # Assuming save_loss_graph is defined

        # Epoch logging
        epoch_loss = running_loss / len(trainloader)
        epoch_iou = running_iou / len(trainloader)
        epoch_pixel_acc = running_pixel_acc / len(trainloader)
        print(f"Epoch {epoch + 1}/{config['epochs']} - Loss: {epoch_loss:.4f} - IoU: {epoch_iou:.4f} - Pixel Acc: {epoch_pixel_acc:.4f}")

        # Save best model checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_weights = copy.deepcopy(student_model.state_dict())
            os.makedirs('./outputs/checkpoints', exist_ok=True)
            torch.save(best_model_weights, './outputs/checkpoints/best_student_model.pth')
            print(f"Best model saved at epoch {epoch + 1} with loss: {best_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.get("patience", 10):
            print("Early stopping triggered.")
            break

    # Load best weights and finalize
    student_model.load_state_dict(best_model_weights)
    print("Training complete. Best model restored.")

def save_visualization(inputs, student_output, teacher_output, output_dir, epoch, ground_truth=None, num_images=5):
    """
    Save visualizations of the input, student output, teacher output, and ground truth.

    Args:
        inputs: Tensor of input images.
        student_output: Tensor of student model outputs.
        teacher_output: Tensor of teacher model outputs.
        output_dir: Directory to save visualizations.
        epoch: Current epoch number.
        ground_truth: Optional tensor of ground truth masks.
        num_images: Number of images to save from each batch (default: 5).
    """
    # Create subdirectories for each epoch and for each type of output
    epoch_dir = os.path.join(output_dir, f"epoch_{epoch + 1}")
    inputs_dir = os.path.join(epoch_dir, "inputs")
    student_dir = os.path.join(epoch_dir, "student")
    teacher_dir = os.path.join(epoch_dir, "teacher")
    binarized_dir = os.path.join(epoch_dir, "student_binarized")

    os.makedirs(inputs_dir, exist_ok=True)
    os.makedirs(student_dir, exist_ok=True)
    os.makedirs(teacher_dir, exist_ok=True)
    os.makedirs(binarized_dir, exist_ok=True)

    # Create ground truth directory if ground truth is provided
    if ground_truth is not None:
        gt_dir = os.path.join(epoch_dir, "ground_truth")
        os.makedirs(gt_dir, exist_ok=True)

    # Move tensors to CPU and convert to images
    inputs = inputs[:num_images].squeeze().cpu()  # Limit to first `num_images`
    student_output = student_output[:num_images].squeeze().cpu()
    teacher_output = teacher_output[:num_images].squeeze().cpu()
    student_binarized = (student_output > 0.5).float()  # Apply threshold for binarized output
    if ground_truth is not None:
        ground_truth = ground_truth[:num_images].squeeze().cpu()

    # Resize outputs to match the dimensions of inputs
    student_output = F.interpolate(student_output.unsqueeze(0), size=inputs.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)
    teacher_output = F.interpolate(teacher_output.unsqueeze(0), size=inputs.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)
    student_binarized = F.interpolate(student_binarized.unsqueeze(0), size=inputs.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)
    if ground_truth is not None:
        ground_truth = F.interpolate(ground_truth.unsqueeze(0), size=inputs.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)

    # Normalize and convert to PIL images
    to_pil = transforms.ToPILImage()
    inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min() + 1e-5)
    student_output = (student_output - student_output.min()) / (student_output.max() - student_output.min() + 1e-5)
    teacher_output = (teacher_output - teacher_output.min()) / (teacher_output.max() - teacher_output.min() + 1e-5)
    student_binarized = (student_binarized - student_binarized.min()) / (student_binarized.max() - student_binarized.min() + 1e-5)
    if ground_truth is not None:
        ground_truth = (ground_truth - ground_truth.min()) / (ground_truth.max() - ground_truth.min() + 1e-5)

    # Save images in the appropriate directories
    for i in range(num_images):  # Loop through the first `num_images`
        input_image = to_pil(inputs[i])
        student_image = to_pil(student_output[i])
        teacher_image = to_pil(teacher_output[i] * ground_truth[i])
        binarized_image = to_pil(student_binarized[i])

        input_image.save(os.path.join(inputs_dir, f"input_{i + 1}.png"))
        student_image.save(os.path.join(student_dir, f"student_{i + 1}.png"))
        teacher_image.save(os.path.join(teacher_dir, f"teacher_{i + 1}.png"))
        binarized_image.save(os.path.join(binarized_dir, f"student_binarized_{i + 1}.png"))

        # Save ground truth if provided
        if ground_truth is not None:
            gt_image = to_pil(ground_truth[i])
            gt_image.save(os.path.join(gt_dir, f"ground_truth_{i + 1}.png"))

    print(f"Saved Last {num_images} images for epoch {epoch + 1} in {epoch_dir}.")

def save_loss_graph(losses, epoch, output_dir="./outputs/checkpoints"):
    """
    Saves a loss graph showing the progression of losses up to a given epoch.

    Args:
        losses (list): List of loss values for each batch.
        epoch (int): Current epoch number.
        output_dir (str): Directory to save the graph.
    """

    os.makedirs(output_dir, exist_ok=True)
    graph_output_path = os.path.join(output_dir, f"loss_graph_epoch_{epoch}.png")

    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Batch Loss", color="blue", alpha=0.8)
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title(f"Loss Progress up to Epoch {epoch}")
    plt.legend()
    plt.grid()
    plt.savefig(graph_output_path)
    plt.close()
    print(f"Loss graph saved at {graph_output_path}")


def validate_model(student_model, teacher_model, valloader, criterion, device):
    """
    Perform validation using the student and teacher models with a progress bar.
    """
    student_model.eval()
    teacher_model.eval()

    val_loss = 0.0
    running_iou = 0.0
    running_pixel_acc = 0.0
    num_batches = len(valloader)

    progress_bar = tqdm(valloader, desc="Validation", leave=False)

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            inputs, gt_labels = batch[0].to(device), batch[5].to(device).float()
            teacher_logits = teacher_model(inputs)
            student_logits = student_model(inputs)

            # Compute loss
            loss = criterion(student_logits, teacher_logits, gt_labels)
            val_loss += loss.item()

            # Compute IoU and Pixel Accuracy
            student_probs_binary = torch.sigmoid(student_logits) > 0.5  # Binarize logits
            iou = compute_iou(student_probs_binary, gt_labels.unsqueeze(1).float())
            pixel_acc = compute_pixel_accuracy(student_probs_binary, gt_labels.unsqueeze(1).float())

            running_iou += iou.mean().item()
            running_pixel_acc += pixel_acc.mean().item()

            # Update the progress bar with metrics
            progress_bar.set_postfix({
                "Loss": f"{val_loss / (batch_idx + 1):.4f}",
                "IoU": f"{running_iou / (batch_idx + 1):.4f}",
                "Pixel Acc": f"{running_pixel_acc / (batch_idx + 1):.4f}"
            })

    val_loss /= num_batches
    avg_iou = running_iou / num_batches
    avg_pixel_acc = running_pixel_acc / num_batches

    print(f"Validation - Loss: {val_loss:.4f}, IoU: {avg_iou:.4f}, Pixel Accuracy: {avg_pixel_acc:.4f}")
    return val_loss, avg_iou, avg_pixel_acc
