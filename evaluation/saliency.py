# Copyright 2022 - Valeo Comfort and Driving Assistance - valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torchvision
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torchvision.transforms as T

from tqdm import tqdm
from scipy import ndimage

from evaluation.metrics.average_meter import AverageMeter
from evaluation.metrics.f_measure import FMeasure
from evaluation.metrics.iou import compute_iou
from evaluation.metrics.mae import compute_mae
from evaluation.metrics.pixel_acc import compute_pixel_accuracy
from evaluation.metrics.s_measure import SMeasure

from misc import batch_apply_bilateral_solver


@torch.no_grad()
def write_metric_tf(writer, metrics, n_iter=-1, name=""):
    writer.add_scalar(
        f"Validation/{name}iou_pred",
        metrics["ious"].avg,
        n_iter,
    )
    writer.add_scalar(
        f"Validation/{name}acc_pred",
        metrics["pixel_accs"].avg,
        n_iter,
    )
    writer.add_scalar(
        f"Validation/{name}f_max",
        metrics["f_maxs"].avg,
        n_iter,
    )


@torch.no_grad()
def eval_batch(batch_gt_masks, batch_pred_masks, metrics_res={}, reset=False):
    """
    Evaluation code adapted from SelfMask: https://github.com/NoelShin/selfmask
    """

    f_values = {}
    # Keep track of f_values for each threshold
    for i in range(255):  # should equal n_bins in metrics/f_measure.py
        f_values[i] = AverageMeter()

    if metrics_res == {}:
        metrics_res["f_scores"] = AverageMeter()
        metrics_res["f_maxs"] = AverageMeter()
        metrics_res["f_maxs_fixed"] = AverageMeter()
        metrics_res["f_means"] = AverageMeter()
        metrics_res["maes"] = AverageMeter()
        metrics_res["ious"] = AverageMeter()
        metrics_res["pixel_accs"] = AverageMeter()
        metrics_res["s_measures"] = AverageMeter()

    if reset:
        metrics_res["f_scores"].reset()
        metrics_res["f_maxs"].reset()
        metrics_res["f_maxs_fixed"].reset()
        metrics_res["f_means"].reset()
        metrics_res["maes"].reset()
        metrics_res["ious"].reset()
        metrics_res["pixel_accs"].reset()
        metrics_res["s_measures"].reset()

    # iterate over batch dimension
    for _, (pred_mask, gt_mask) in enumerate(zip(batch_pred_masks, batch_gt_masks)):
        assert pred_mask.shape == gt_mask.shape, f"{pred_mask.shape} != {gt_mask.shape}"
        assert len(pred_mask.shape) == len(gt_mask.shape) == 2
        # Compute
        # Binarize at 0.5 for IoU and pixel accuracy

        # avg = torch.mean(pred_mask)
        binary_pred = (pred_mask > 0.5).float().squeeze()
        iou = compute_iou(binary_pred, gt_mask)
        f_measures = FMeasure()(pred_mask, gt_mask)  # soft mask for F measure
        mae = compute_mae(binary_pred, gt_mask)
        pixel_acc = compute_pixel_accuracy(binary_pred, gt_mask)

        # Update
        metrics_res["ious"].update(val=iou.numpy(), n=1)
        metrics_res["f_scores"].update(val=f_measures["f_measure"].numpy(), n=1)
        metrics_res["f_maxs"].update(val=f_measures["f_max"].numpy(), n=1)
        metrics_res["f_means"].update(val=f_measures["f_mean"].numpy(), n=1)
        metrics_res["s_measures"].update(
            val=SMeasure()(pred_mask=pred_mask, gt_mask=gt_mask.to(torch.float32)), n=1
        )
        metrics_res["maes"].update(val=mae.numpy(), n=1)
        metrics_res["pixel_accs"].update(val=pixel_acc.numpy(), n=1)

        # Keep track of f_values for each threshold
        all_f = f_measures["all_f"].numpy()
        for k, v in f_values.items():
            v.update(val=all_f[k], n=1)
        # Then compute the max for the f_max_fixed
        metrics_res["f_maxs_fixed"].update(
            val=np.max([v.avg for v in f_values.values()]), n=1
        )

    results = {}
    # F-measure, F-max, F-mean, MAE, S-measure, IoU, pixel acc.
    results["f_measure"] = metrics_res["f_scores"].avg
    results["f_max"] = metrics_res["f_maxs"].avg
    results["f_maxs_fixed"] = metrics_res["f_maxs_fixed"].avg
    results["f_mean"] = metrics_res["f_means"].avg
    results["s_measure"] = metrics_res["s_measures"].avg
    results["mae"] = metrics_res["maes"].avg
    results["iou"] = float(iou.numpy())
    results["pixel_acc"] = metrics_res["pixel_accs"].avg

    return results, metrics_res

import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from scipy import ndimage
from torchvision import transforms


# def evaluate_saliency(
#     dataset,
#     model,
#     writer=None,
#     batch_size=1,
#     n_iter=-1,
#     apply_bilateral=False,
#     im_fullsize=True,
#     method="pred",  # can also be "bkg",
#     apply_weights: bool = True,
#     evaluation_mode: str = "single",  # choices are ["single", "multi"]
#     save_visualizations: bool = True,  # New parameter to toggle saving
#     save_every: int = 50,  # Frequency of saving
#     output_dir: str = "outputs/eval_visualizations",  # Base directory for visualizations
#     num_images_to_save: int = 5,  # Number of images to save per save call
# ):
#
#     if im_fullsize:
#         # Change transformation
#         dataset.fullimg_mode()
#         batch_size = 1
#
#     valloader = torch.utils.data.DataLoader(
#         dataset, batch_size=batch_size, shuffle=False, num_workers=2
#     )
#
#     sigmoid = nn.Sigmoid()
#
#     metrics_res = {}
#     metrics_res_bs = {}
#     valbar = tqdm(enumerate(valloader, 0), leave=None)
#     model.eval()
#
#     # Define output directories
#     # This is handled by save_visualization, so no need to define here
#     # But ensure the base output directory exists
#     os.makedirs(output_dir, exist_ok=True)
#
#     for i, data in valbar:
#         inputs, _, _, _, _, gt_labels, _ = data
#         inputs = inputs.to("cuda")
#         gt_labels = gt_labels.to("cuda").float()
#
#         transform = torchvision.transforms.Resize((256, 256))
#         inputs_resized = transform(inputs)
#
#         # Forward step
#         with torch.no_grad():
#             preds = model(inputs_resized)
#
#         h, w = gt_labels.shape[-2:]
#         preds_up = F.interpolate(preds, size=(h, w), mode="bilinear", align_corners=False)
#         soft_preds = sigmoid(preds_up.detach()).squeeze(0)
#         binary_pred = (soft_preds > 0.5).float()
#
#         reset = True if i == 0 else False
#
#         if evaluation_mode == "single":
#             labeled, nr_objects = ndimage.label(binary_pred.squeeze().cpu().numpy())
#             if nr_objects == 0:
#                 preds_up_one_cc = binary_pred.squeeze()
#                 print("nr_objects == 0")
#             else:
#                 nb_pixel = [np.sum(labeled == j) for j in range(nr_objects + 1)]
#                 pixel_order = np.argsort(nb_pixel)
#
#                 cc = [torch.Tensor(labeled == j) for j in pixel_order]
#                 cc = torch.stack(cc).cuda()
#
#                 # Find CC set as background, here not necessarily the biggest
#                 cc_background = (
#                     (
#                         (
#                             (~(binary_pred[None, :, :, :].bool())).float()
#                             + cc[:, None, :, :].cuda()
#                         )
#                         > 1
#                     )
#                     .sum(-1)
#                     .sum(-1)
#                     .argmax()
#                 )
#                 pixel_order = np.delete(pixel_order, int(cc_background.cpu().numpy()))
#
#                 preds_up_one_cc = torch.Tensor(labeled == pixel_order[-1]).cuda()
#
#             _, metrics_res = eval_batch(
#                 gt_labels,
#                 preds_up_one_cc.unsqueeze(0),
#                 metrics_res=metrics_res,
#                 reset=reset,
#             )
#         elif evaluation_mode == "multi":
#             _, metrics_res = eval_batch(
#                 gt_labels,
#                 soft_preds.unsqueeze(0) if len(soft_preds.shape) == 2 else soft_preds,
#                 metrics_res=metrics_res,
#                 reset=reset,
#             )  # soft preds needed for F beta measure
#
#         # Apply bilateral solver
#         preds_bs = None
#         if apply_bilateral:
#             get_all_cc = True if evaluation_mode == "multi" else False
#             preds_bs, _ = batch_apply_bilateral_solver(
#                 data, preds_up.detach(), get_all_cc=get_all_cc
#             )
#
#             _, metrics_res_bs = eval_batch(
#                 gt_labels,
#                 preds_bs[None, :, :].float(),
#                 metrics_res=metrics_res_bs,
#                 reset=reset,
#             )
#
#         bar_str = (
#             f"{dataset.name} | {evaluation_mode} mode | "
#             f"F-max {metrics_res['f_maxs'].avg:.3f} "
#             f"IoU {metrics_res['ious'].avg:.3f}, "
#             f"PA {metrics_res['pixel_accs'].avg:.3f}"
#         )
#
#         if apply_bilateral:
#             bar_str += (
#                 f" | with bilateral solver: "
#                 f"F-max {metrics_res_bs['f_maxs'].avg:.3f}, "
#                 f"IoU {metrics_res_bs['ious'].avg:.3f}, "
#                 f"PA. {metrics_res_bs['pixel_accs'].avg:.3f}"
#             )
#
#         valbar.set_description(bar_str)
#
#         # Save every `save_every` iteration
#         if save_visualizations and ((i + 1) % save_every == 0):
#             # Define the current epoch or iteration
#             epoch_num = (i + 1) // save_every
#
#             # Prepare arguments for save_visualization
#             # Assuming teacher_output is not available; pass None
#             teacher_output = None  # Replace with actual teacher output if available
#
#             # Select a subset of images to save
#             # If batch_size > num_images_to_save, randomly select or take the first `num_images_to_save`
#             num_images = min(num_images_to_save, inputs.size(0))
#
#             # Call save_visualization
#             # save_visualization(
#             #     inputs=inputs[:num_images],
#             #     student_output=preds_up[:num_images],
#             #     teacher_output=binary_pred,  # Modify if teacher_output is available
#             #     output_dir=output_dir,
#             #     epoch=epoch_num,
#             #     ground_truth=gt_labels[:num_images],
#             #     num_images=num_images
#             # )
#
#     # Writing in tensorboard
#     if writer is not None:
#         write_metric_tf(
#             writer,
#             metrics_res,
#             n_iter=n_iter,
#             name=f"{dataset.name}_{evaluation_mode}_",
#         )
#
#         if apply_bilateral:
#             write_metric_tf(
#                 writer,
#                 metrics_res_bs,
#                 n_iter=n_iter,
#                 name=f"{dataset.name}_{evaluation_mode}-BS_",
#             )
#
#     # Go back to original transformation
#     if im_fullsize:
#         dataset.training_mode()

def evaluate_saliency(
    dataset,
    model,
    writer=None,
    batch_size=1,
    n_iter=-1,
    apply_bilateral=False,
    im_fullsize=True,
    method="pred",  # can also be "bkg",
    apply_weights: bool = True,
    evaluation_mode: str = "single",  # choices are ["single", "multi"]
):

    if im_fullsize:
        # Change transformation
        dataset.fullimg_mode()
        batch_size = 1

    valloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    sigmoid = nn.Sigmoid()

    metrics_res = {}
    metrics_res_bs = {}
    valbar = tqdm(enumerate(valloader, 0), leave=None)
    model.eval()
    for i, data in valbar:
        inputs, _, _, _, _, gt_labels, _ = data
        inputs = inputs.to("cuda")
        gt_labels = gt_labels.to("cuda").float()

        transform = torchvision.transforms.Resize((256, 256))
        inputs = transform(inputs)

        # Forward step
        with torch.no_grad():
            # preds = model(inputs, for_eval=True)
            preds = model(inputs)

        h, w = gt_labels.shape[-2:]
        # preds_up = F.interpolate(
        #     preds,
        #     scale_factor=model.vit_patch_size,
        #     mode="bilinear",
        #     align_corners=False,
        # )[..., :h, :w]
        preds_up = F.interpolate(preds, size=(h, w), mode="bilinear", align_corners=False)
        soft_preds = sigmoid(preds_up.detach()).squeeze(0)
        preds_up = (sigmoid(preds_up.detach()) > 0.55).squeeze(0).float()

        reset = True if i == 0 else False
        if evaluation_mode == "single":
            labeled, nr_objects = ndimage.label(preds_up.squeeze().cpu().numpy())
            if nr_objects == 0:
                preds_up_one_cc = preds_up.squeeze()
                print("nr_objects == 0")
            else:
                nb_pixel = [np.sum(labeled == i) for i in range(nr_objects + 1)]
                pixel_order = np.argsort(nb_pixel)

                cc = [torch.Tensor(labeled == i) for i in pixel_order]
                cc = torch.stack(cc).cuda()

                # Find CC set as background, here not necessarily the biggest
                cc_background = (
                    (
                        (
                            (~(preds_up[None, :, :, :].bool())).float()
                            + cc[:, None, :, :].cuda()
                        )
                        > 1
                    )
                    .sum(-1)
                    .sum(-1)
                    .argmax()
                )
                pixel_order = np.delete(pixel_order, int(cc_background.cpu().numpy()))

                preds_up_one_cc = torch.Tensor(labeled == pixel_order[-1]).cuda()

            _, metrics_res = eval_batch(
                gt_labels,
                preds_up_one_cc.unsqueeze(0),
                metrics_res=metrics_res,
                reset=reset,
            )

        elif evaluation_mode == "multi":
            # Eval without bilateral solver
            _, metrics_res = eval_batch(
                gt_labels,
                soft_preds.unsqueeze(0) if len(soft_preds.shape) == 2 else soft_preds,
                metrics_res=metrics_res,
                reset=reset,
            )  # soft preds needed for F beta measure

        # Apply bilateral solver
        preds_bs = None
        if apply_bilateral:
            get_all_cc = True if evaluation_mode == "multi" else False
            preds_bs, _ = batch_apply_bilateral_solver(
                data, preds_up.detach(), get_all_cc=get_all_cc
            )

            _, metrics_res_bs = eval_batch(
                gt_labels,
                preds_bs[None, :, :].float(),
                metrics_res=metrics_res_bs,
                reset=reset,
            )

        bar_str = (
            f"{dataset.name} | {evaluation_mode} mode | "
            f"F-max {metrics_res['f_maxs'].avg:.3f} "
            f"IoU {metrics_res['ious'].avg:.3f}, "
            f"PA {metrics_res['pixel_accs'].avg:.3f}"
        )

        if apply_bilateral:
            bar_str += (
                f" | with bilateral solver: "
                f"F-max {metrics_res_bs['f_maxs'].avg:.3f}, "
                f"IoU {metrics_res_bs['ious'].avg:.3f}, "
                f"PA. {metrics_res_bs['pixel_accs'].avg:.3f}"
            )

        valbar.set_description(bar_str)

    # Writing in tensorboard
    if writer is not None:
        write_metric_tf(
            writer,
            metrics_res,
            n_iter=n_iter,
            name=f"{dataset.name}_{evaluation_mode}_",
        )

        if apply_bilateral:
            write_metric_tf(
                writer,
                metrics_res_bs,
                n_iter=n_iter,
                name=f"{dataset.name}_{evaluation_mode}-BS_",
            )

    # Go back to original transformation
    if im_fullsize:
        dataset.training_mode()

@torch.no_grad()
def eval_batch_student(batch_gt_masks, batch_pred_masks, metrics_res={}, reset=False):
    """
    Evaluates a batch of predictions for the student model without debugging prints.
    """
    f_values = {i: AverageMeter() for i in range(255)}  # F-measure thresholds

    # Initialize metrics_res if empty
    if not metrics_res:
        metrics_res = {
            "f_scores": AverageMeter(),
            "f_maxs": AverageMeter(),
            "f_maxs_fixed": AverageMeter(),
            "f_means": AverageMeter(),
            "maes": AverageMeter(),
            "ious": AverageMeter(),
            "pixel_accs": AverageMeter(),
            "s_measures": AverageMeter(),
        }

    # Reset metrics if specified
    if reset:
        for meter in metrics_res.values():
            meter.reset()

    # Iterate over batch of predictions and ground truth masks
    for pred_mask, gt_mask in zip(batch_pred_masks, batch_gt_masks):
        pred_mask = pred_mask.squeeze()
        gt_mask = gt_mask.squeeze()

        # Binarize the prediction mask at 0.5 threshold
        binary_pred = (pred_mask > 0.5).float()

        # Calculate metrics
        iou = compute_iou(binary_pred, gt_mask)
        f_measures = FMeasure()(pred_mask, gt_mask)
        mae = compute_mae(binary_pred, gt_mask)
        pixel_acc = compute_pixel_accuracy(binary_pred, gt_mask)

        # Update metrics
        metrics_res["ious"].update(val=iou.numpy(), n=1)
        metrics_res["f_scores"].update(val=f_measures["f_measure"].numpy(), n=1)
        metrics_res["f_maxs"].update(val=f_measures["f_max"].numpy(), n=1)
        metrics_res["f_means"].update(val=f_measures["f_mean"].numpy(), n=1)
        metrics_res["s_measures"].update(SMeasure()(pred_mask, gt_mask), n=1)
        metrics_res["maes"].update(val=mae.numpy(), n=1)
        metrics_res["pixel_accs"].update(val=pixel_acc.numpy(), n=1)

        # Track F-measure at different thresholds
        all_f = f_measures["all_f"].numpy()
        for k, v in f_values.items():
            v.update(val=all_f[k], n=1)

        # Update f_maxs_fixed with max of f_values at different thresholds
        metrics_res["f_maxs_fixed"].update(val=np.max([v.avg for v in f_values.values()]), n=1)

    # Final average metrics
    results = {k: v.avg for k, v in metrics_res.items()}
    return results, metrics_res


@torch.no_grad()
def student_evaluation_saliency(
        dataset,
        student_model,
        batch_size=1,
        apply_bilateral=False,
        im_fullsize=True,
        evaluation_mode="multi",
        output_dir="outputs/evaluation"
):
    """
    Evaluates the StudentModel for saliency detection on a specified dataset.

    Parameters:
    - dataset: Dataset to evaluate on.
    - student_model: The student model to be evaluated.
    - batch_size: Number of images per batch.
    - apply_bilateral: Whether to apply a bilateral solver for smoothing.
    - im_fullsize: Whether to evaluate at full image size.
    - evaluation_mode: Mode of evaluation ("single" or "multi").
    - output_dir: Directory to save the output images.
    """
    student_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model.to(device)

    # Create output subfolders
    os.makedirs(os.path.join(output_dir, "predicted"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "ground_truth"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "original"), exist_ok=True)

    if im_fullsize:
        dataset.fullimg_mode()
        batch_size = 1

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    sigmoid = nn.Sigmoid()
    metrics_res = {}

    valbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating")
    for i, data in valbar:
        inputs = data[0]  # Adjusted to directly access inputs
        gt_labels = data[5].to(device).float()

        transform = torchvision.transforms.Resize((128, 128))
        inputs = transform(inputs)

        inputs = inputs.to(device)  # Move inputs to the appropriate device
        # Generate predictions
        with torch.no_grad():
            preds = student_model(inputs)

        # Resize predictions to match ground truth dimensions
        h, w = gt_labels.shape[-2:]
        preds_up = F.interpolate(preds, size=(h, w), mode="bilinear", align_corners=False)
        soft_preds = sigmoid(preds_up).squeeze(0)  # Soft prediction for F-measure
        binary_preds = (soft_preds > 0.5).float()  # Binary prediction for IoU and Pixel Accuracy


        # Save every 100th image
        if i % 100 == 0:
            save_evaluation_visualization(
                inputs.cpu().squeeze(0),
                gt_labels.cpu().squeeze(0),
                preds_up.cpu().squeeze(0),  # Soft predictions
                binary_preds.cpu().squeeze(0),  # Binarized predictions
                output_dir,
                i
            )
        reset = i == 0

        _, metrics_res = eval_batch_student(
            gt_labels, soft_preds.unsqueeze(0), metrics_res=metrics_res, reset=reset
        )
        # if evaluation_mode == "single":
        #     labeled, num_objects = ndimage.label(binary_preds.cpu().numpy())
        #     if num_objects == 0:
        #         preds_up_one_cc = binary_preds
        #     else:
        #         sizes = [np.sum(labeled == j) for j in range(1, num_objects + 1)]
        #         largest_cc = (labeled == (np.argmax(sizes) + 1))
        #         preds_up_one_cc = torch.tensor(largest_cc, dtype=torch.float32, device=device)
        #
        #     _, metrics_res = eval_batch_student(
        #         gt_labels, preds_up_one_cc.unsqueeze(0), metrics_res=metrics_res, reset=reset
        #     )
        #
        # elif evaluation_mode == "multi":
        #     _, metrics_res = eval_batch_student(
        #         gt_labels, soft_preds.unsqueeze(0), metrics_res=metrics_res, reset=reset
        #     )

        # Bilateral solver option
        if apply_bilateral:
            preds_bs, _ = batch_apply_bilateral_solver(data, binary_preds.detach(),
                                                       get_all_cc=(evaluation_mode == "multi"))
            _, metrics_res = eval_batch_student(gt_labels, preds_bs[None, :, :].float(), metrics_res=metrics_res,
                                                reset=reset)

        # Update progress bar
        valbar.set_postfix(
            f_max=metrics_res.get("f_maxs", AverageMeter()).avg,
            IoU=metrics_res.get("ious", AverageMeter()).avg,
            pixel_acc=metrics_res.get("pixel_accs", AverageMeter()).avg,
        )

    return metrics_res

@torch.no_grad()
def eval_batch_periphery(batch_gt_masks, batch_pred_masks, metrics_res={}, reset=False):
    """
    Evaluates a batch of predictions for the student model.
    This version ensures alignment with the simplified evaluation process.

    Parameters:
    - batch_gt_masks: Ground truth masks (binary), shape [B, 1, H, W].
    - batch_pred_masks: Predicted soft masks (probabilities), shape [B, 1, H, W].
    - metrics_res: A dictionary of metrics accumulators.
    - reset: Whether to reset the metrics at the beginning of the evaluation.
    """
    # Initialize metrics if not provided
    if not metrics_res:
        metrics_res = {
            "ious": AverageMeter(),
            "f_scores": AverageMeter(),
            "f_maxs": AverageMeter(),
            "f_means": AverageMeter(),
            "maes": AverageMeter(),
            "pixel_accs": AverageMeter(),
            "s_measures": AverageMeter(),
        }

    # Reset metrics if specified
    if reset:
        for meter in metrics_res.values():
            meter.reset()

    # Iterate over batch
    for pred_mask, gt_mask in zip(batch_pred_masks, batch_gt_masks):
        # Ensure dimensions are consistent
        pred_mask = pred_mask.squeeze(1)  # Shape [H, W]
        gt_mask = gt_mask.squeeze(1)  # Shape [H, W]

        # Binarize predictions at 0.5 threshold
        binary_pred = (pred_mask > 0.5).float()

        # Calculate metrics
        iou = compute_iou(binary_pred, gt_mask)
        f_measures = FMeasure()(pred_mask, gt_mask)  # Soft prediction for F-measure
        mae = compute_mae(binary_pred, gt_mask)
        pixel_acc = compute_pixel_accuracy(binary_pred, gt_mask)
        s_measure = SMeasure()(pred_mask, gt_mask)  # Soft prediction for S-measure

        # Update metrics
        metrics_res["ious"].update(val=iou.numpy(), n=1)
        metrics_res["f_scores"].update(val=f_measures["f_measure"].numpy(), n=1)
        metrics_res["f_maxs"].update(val=f_measures["f_max"].numpy(), n=1)
        metrics_res["f_means"].update(val=f_measures["f_mean"].numpy(), n=1)
        metrics_res["s_measures"].update(val=s_measure, n=1)
        metrics_res["maes"].update(val=mae.numpy(), n=1)
        metrics_res["pixel_accs"].update(val=pixel_acc.numpy(), n=1)

    # Return results
    results = {k: v.avg for k, v in metrics_res.items()}
    return results, metrics_res


@torch.no_grad()
def periphery_evaluation_saliency(
        dataset,
        student_model,
        batch_size=1,
        output_dir="outputs/evaluation"
):
    """
    Simplified evaluation of the Periphery student model for saliency detection.

    Parameters:
    - dataset: Dataset to evaluate on.
    - student_model: The student model to be evaluated.
    - batch_size: Number of images per batch.
    - output_dir: Directory to save the output images.
    """

    # Set model to evaluation mode and move to device
    student_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model.to(device)

    # Create output directories
    os.makedirs(os.path.join(output_dir, "predicted"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "ground_truth"), exist_ok=True)

    # Initialize metrics
    metrics_res = {
        "ious": AverageMeter(),
        "f_scores": AverageMeter(),
        "f_maxs": AverageMeter(),
        "f_means": AverageMeter(),
        "maes": AverageMeter(),
        "pixel_accs": AverageMeter(),
        "s_measures": AverageMeter(),
    }

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    sigmoid = nn.Sigmoid()
    transform = T.Resize((128, 128))  # Resize to match input size for the student model

    valbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating")
    for i, data in valbar:
        inputs, gt_labels = data[0], data[5]
        inputs = inputs.to(device)
        gt_labels = gt_labels.to(device).float()

        # Generate predictions
        preds = student_model(transform(inputs))
        preds_up = F.interpolate(preds, size=gt_labels.shape[-2:], mode="bilinear", align_corners=False)

        # Apply sigmoid for probabilities and binarize
        soft_preds = sigmoid(preds_up)  # [B, C, H, W]
        binary_preds = (soft_preds > 0.5).float()

        # Calculate metrics
        _, metrics_res = eval_batch_periphery(
            batch_gt_masks=gt_labels.unsqueeze(1),  # Ensure [B, C, H, W]
            batch_pred_masks=soft_preds,  # Probabilities
            metrics_res=metrics_res,
            reset=(i == 0)
        )

        # Save visualizations periodically
        if i % 100 == 0:
            save_evaluation_visualization(
                inputs.cpu().squeeze(0),
                gt_labels.cpu().squeeze(0),
                preds_up.cpu().squeeze(0),  # Raw logits
                binary_preds.cpu().squeeze(0),  # Binarized predictions
                output_dir,
                i
            )

        # Update progress bar with metrics
        valbar.set_postfix(
            f_max=metrics_res.get("f_maxs", AverageMeter()).avg,
            IoU=metrics_res.get("ious", AverageMeter()).avg,
            pixel_acc=metrics_res.get("pixel_accs", AverageMeter()).avg,
        )

    return metrics_res

def save_visualization(inputs, student_output, teacher_output, output_dir, epoch, ground_truth=None, num_images=5):
    """
    Save visualizations of the input, student output, teacher output, and ground truth.

    Args:
        inputs: Tensor of input images.
        student_output: Tensor of student model outputs.
        teacher_output: Tensor of teacher model outputs (binary or continuous).
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
    student_output = student_output[:num_images].cpu()
    teacher_output = teacher_output[:num_images].cpu()

    # Ensure tensors have the shape (N, C, H, W)
    if student_output.dim() == 3:  # If the output is (N, H, W), add channel dimension
        student_output = student_output.unsqueeze(1)

    if teacher_output.dim() == 3:  # If the output is (N, H, W), add channel dimension
        teacher_output = teacher_output.unsqueeze(1)

    # Resize outputs to match the dimensions of inputs
    student_output = F.interpolate(student_output, size=inputs.shape[-2:], mode='bilinear', align_corners=False)
    teacher_output = F.interpolate(teacher_output, size=inputs.shape[-2:], mode='bilinear', align_corners=False)

    # Squeeze out the channel dimension if necessary
    student_output = student_output.squeeze(1)  # Remove channel dimension
    teacher_output = teacher_output.squeeze(1)  # Remove channel dimension

    # Normalize and convert to PIL images
    to_pil = transforms.ToPILImage()
    inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min() + 1e-5)
    student_output = (student_output - student_output.min()) / (student_output.max() - student_output.min() + 1e-5)
    teacher_output = (teacher_output - teacher_output.min()) / (teacher_output.max() - teacher_output.min() + 1e-5)

    if ground_truth is not None:
        ground_truth = (ground_truth - ground_truth.min()) / (ground_truth.max() - ground_truth.min() + 1e-5)

    # Save images in the appropriate directories
    for i in range(num_images):  # Loop through the first `num_images`
        input_image = to_pil(inputs[i])
        student_image = to_pil(student_output[i])
        teacher_image = to_pil(teacher_output[i])

        input_image.save(os.path.join(inputs_dir, f"input_{i + 1}.png"))
        student_image.save(os.path.join(student_dir, f"student_{i + 1}.png"))
        teacher_image.save(os.path.join(teacher_dir, f"teacher_{i + 1}.png"))

        # Save ground truth if provided
        if ground_truth is not None:
            gt_image = to_pil(ground_truth[i])
            gt_image.save(os.path.join(gt_dir, f"ground_truth_{i + 1}.png"))

    print(f"Saved Last {num_images} images for epoch {epoch + 1} in {epoch_dir}.")