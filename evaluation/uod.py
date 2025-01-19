# Copyright 2021 - Valeo Comfort and Driving Assistance - Oriane Siméoni @ valeo.ai
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

"""
Code adapted from previous method LOST: https://github.com/valeoai/LOST
Code adapted from previous method Peekaboo: https://github.com/hasibzunair/peekaboo
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F

from tqdm import tqdm
from misc import bbox_iou, get_bbox_from_segmentation_labels


def evaluation_unsupervised_object_discovery(
        dataset,
        model,
        evaluation_mode: str = "single",  # choices are ["single", "multi"]
        output_dir: str = "outputs",
        no_hards: bool = False,
):
    assert evaluation_mode == "single"

    sigmoid = nn.Sigmoid()

    # Define input size for resizing
    input_size = (224, 224)

    # ----------------------------------------------------
    # Loop over images
    preds_dict = {}
    cnt = 0
    corloc = np.zeros(len(dataset.dataloader))

    start_time = time.time()
    pbar = tqdm(dataset.dataloader)
    for im_id, inp in enumerate(pbar):

        # ------------ IMAGE PROCESSING -------------------------------------------
        img = inp[0]

        init_image_size = img.shape

        # Get the name of the image
        im_name = dataset.get_image_name(inp[1])
        # Pass in case of no gt boxes in the image
        if im_name is None:
            continue

        # Resize the image to the model input size
        resize_transform = transforms.Resize(input_size)
        img_resized = resize_transform(img)

        # # Move to gpu
        img_resized = img_resized.cuda(non_blocking=True)

        # ------------ GROUND-TRUTH -------------------------------------------
        gt_bbxs, gt_cls = dataset.extract_gt(inp[1], im_name)

        if gt_bbxs is not None:
            # Discard images with no gt annotations
            # Happens only in the case of VOC07 and VOC12
            if gt_bbxs.shape[0] == 0 and no_hards:
                continue

        outputs = model(img_resized[None, :, :, :])  # Pass resized image into student model
        preds = (sigmoid(outputs[0].detach()) > 0.5).float().squeeze().cpu().numpy()

        # Interpolate preds to match init_image_size for correct bbox calculation
        if preds.shape != init_image_size[1:]:
            preds_tensor = torch.from_numpy(preds).unsqueeze(0).unsqueeze(0).float().cuda()
            preds_interpolated = F.interpolate(preds_tensor, size=init_image_size[1:], mode="bilinear",
                                              align_corners=False).squeeze(0).squeeze(0).cpu().numpy()
            preds = preds_interpolated

        # get bbox
        pred = get_bbox_from_segmentation_labels(
            segmenter_predictions=preds,
            scales=[1,1], # we already upsampled our segmentation map
            initial_image_size=init_image_size[1:],
        )



        # ------------ Visualizations -------------------------------------------
        # Save the prediction
        preds_dict[im_name] = pred

        # Compare prediction to GT boxes
        ious = bbox_iou(torch.from_numpy(pred), torch.from_numpy(gt_bbxs))

        if torch.any(ious >= 0.5):
            corloc[im_id] = 1

        cnt += 1
        if cnt % 50 == 0:
            pbar.set_description(f"Periphery {int(np.sum(corloc))}/{cnt}")

    # Evaluate
    print(f"corloc: {100 * np.sum(corloc) / cnt:.2f} ({int(np.sum(corloc))}/{cnt})")
    result_file = os.path.join(output_dir, "uod_results_student.txt")
    with open(result_file, "w") as f:
        f.write("corloc,%.1f,,\n" % (100 * np.sum(corloc) / cnt))
    print("File saved at %s" % result_file)