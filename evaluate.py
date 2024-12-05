import argparse
import torch
from periphery import Periphery
from misc import load_config
from datasets.datasets import build_dataset
from evaluation.saliency import evaluate_saliency
from evaluation.uod import evaluation_unsupervised_object_discovery

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation of Peekaboo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--eval-type", type=str, choices=["saliency", "uod"], help="Evaluation type."
    )
    parser.add_argument(
        "--dataset-eval",
        type=str,
        choices=["ECSSD", "DUT-OMRON", "DUTS-TEST", "VOC07", "VOC12", "COCO20k"],
        help="Name of evaluation dataset.",
    )
    parser.add_argument(
        "--dataset-set-eval", type=str, default=None, help="Set of the dataset."
    )
    parser.add_argument(
        "--apply-bilateral", action="store_true", help="Use bilateral solver."
    )
    parser.add_argument(
        "--evaluation-mode",
        type=str,
        default="multi",
        choices=["single", "multi"],
        help="Type of evaluation.",
    )
    parser.add_argument(
        "--model-weights",
        type=str,
        default="data/weights/decoder_weights.pt",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ukan_mini_DUTS-TR.yaml",
    )
    parser.add_argument(
        "--student-model", action="store_true", help="Evaluate the student model instead of the teacher model."
    )
    args = parser.parse_args()
    print(args.__dict__)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configuration
    config, _ = load_config(args.config)

    encoder_config = {
        "num_classes": config.UKAN_Config["num_classes"],
        "input_channels": config.UKAN_Config["input_channels"],
        "deep_supervision": config.UKAN_Config["deep_supervision"],
        "img_size": config.UKAN_Config["img_size"],
        "patch_size": config.UKAN_Config["patch_size"],
        "in_chans": config.UKAN_Config["in_chans"],
        "embed_dims": config.UKAN_Config["embed_dims"],
        "no_kan": config.UKAN_Config["no_kan"],
        "drop_rate": 0.0,
        "drop_path_rate": 0.0,
        "norm_layer": eval(config.UKAN_Config["norm_layer"]),  # Convert string to actual class
        "depths": config.UKAN_Config["depths"],
    }

    model = Periphery(encoder_config=encoder_config, freeze=False, pretrained_weights_path=None)  # Load student model
    # Move the model to the device
    model = model.to(device)
    model.eval()
    checkpoint = torch.load(config.distillation["checkpoint_path"], map_location=torch.device('cpu') if not torch.cuda.is_available() else None)

    # Load the weights from the specified checkpoint path
    model.load_state_dict(checkpoint)
    print(f'In Eval Student Model Weight Loaded From {config.distillation["checkpoint_path"]} Successfully')

    # Build the validation dataset
    val_dataset = build_dataset(
        root_dir=args.dataset_dir,
        dataset_name=args.dataset_eval,
        dataset_set=args.dataset_set_eval,
        for_eval=True,
        evaluation_type=args.eval_type,
    )
    print(f"\nBuilding dataset {val_dataset.name} (#{len(val_dataset)} images)")

    # Start evaluation
    print(f"\nStarted evaluation on {val_dataset.name}")
    if args.eval_type == "saliency":
        if args.student_model:
            # Use student evaluation function for saliency
            evaluate_saliency(
                dataset=val_dataset,
                model=model,
                evaluation_mode=args.evaluation_mode,
                apply_bilateral=args.apply_bilateral,
            )
        else:
            # Use teacher evaluation function for saliency
            evaluate_saliency(
                dataset=val_dataset,
                model=model,
                evaluation_mode=args.evaluation_mode,
                apply_bilateral=args.apply_bilateral,
            )
    elif args.eval_type == "uod":
        if args.apply_bilateral:
            raise ValueError("Bilateral solver is not implemented for unsupervised object discovery.")
        # Use UOD evaluation for either model (assuming same function applies)
        evaluation_unsupervised_object_discovery(
            dataset=val_dataset,
            model=model,
            evaluation_mode=args.evaluation_mode,
        )
    else:
        raise ValueError("Other evaluation method not implemented.")