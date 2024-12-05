import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from model import PeekabooModel
from models.student_base_model import StudentModel
from datasets.datasets import build_dataset
from misc import load_config


def save_predictions(input_image, student_preds, teacher_preds, output_dir, idx):
    """Save the input image, student and teacher predictions."""

    # Convert tensors to PIL images
    input_pil = transforms.ToPILImage()(input_image)
    student_pred_pil = transforms.ToPILImage()(student_preds)
    teacher_pred_pil = transforms.ToPILImage()(teacher_preds)

    # Save images
    input_pil.save(os.path.join(output_dir, "inputs", f"input_image_{idx + 1}.png"))
    student_pred_pil.save(os.path.join(output_dir, "student_predictions", f"student_prediction_{idx + 1}.png"))
    teacher_pred_pil.save(os.path.join(output_dir, "teacher_predictions", f"teacher_prediction_{idx + 1}.png"))

    print(f"Saved input, student, and teacher predictions for image {idx + 1}")


def test_and_save_predictions(teacher_model, student_model, dataset, output_dir, num_images=5):
    # Set up directories for saving predictions and inputs
    os.makedirs(os.path.join(output_dir, "teacher_predictions"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "student_predictions"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "inputs"), exist_ok=True)

    # Load a portion of the dataset for testing
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    sigmoid = nn.Sigmoid()

    # Set models to evaluation mode and move to the appropriate device
    teacher_model.eval()
    student_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model.to(device)
    student_model.to(device)

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if idx >= num_images:
                break

            inputs, _, _, _, _, _, _ = data
            inputs = inputs.to(device)

            # Pre-process input image for saving (normalize to [0, 1] for visualization)
            input_image = inputs.squeeze().cpu()
            input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min() + 1e-5)

            # Teacher's prediction
            teacher_preds = teacher_model(inputs)
            teacher_preds = sigmoid(teacher_preds).squeeze().cpu()
            teacher_preds = (teacher_preds - teacher_preds.min()) / (teacher_preds.max() - teacher_preds.min() + 1e-5)

            # Student's prediction with binary thresholding
            student_preds = student_model(inputs)
            student_preds = sigmoid(student_preds).squeeze().cpu()
            student_preds = (student_preds - student_preds.min()) / (student_preds.max() - student_preds.min() + 1e-5)

            # Apply a binary threshold for the student predictions (for black-white mask)
            threshold = 0.5
            binary_mask = (student_preds < threshold).float()  # 0 for highlighted, 1 for background
            binary_mask = 1 - binary_mask  # Invert colors for black (highlighted) and white (background)

            # Save all images for this sample
            save_predictions(input_image, binary_mask, teacher_preds, output_dir, idx)

    print(f"Predictions saved in '{output_dir}'")


def main():
    # Load configuration
    config_path = "configs/ukan_mini_DUTS-TR.yaml"
    config, _ = load_config(config_path)

    # Specify output directory for predictions
    output_dir = "./outputs/test_predictions"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize teacher and student models with respective weights
    teacher_model = PeekabooModel(
        vit_model=config.model["pre_training"],
        vit_arch=config.model["arch"],
        vit_patch_size=config.model["patch_size"],
        enc_type_feats=config.peekaboo["feats"],
    )
    teacher_model.decoder_load_weights("./data/weights/peekaboo_decoder_weights_niter500.pt")

    student_model = StudentModel()
    student_model.decoder_load_weights('./outputs/peekaboo-DUTS-TR-vit_small8/student_model_final.pth')

    # Build the dataset for evaluation
    dataset = build_dataset(
        root_dir='./datasets_local/',
        dataset_name=config.training["dataset"],
        dataset_set=config.training["dataset_set"],
        config=config,
        for_eval=True,
    )

    # Run testing and save predictions
    test_and_save_predictions(teacher_model, student_model, dataset, output_dir, num_images=10)


if __name__ == "__main__":
    main()