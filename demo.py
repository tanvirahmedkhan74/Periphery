# Code for Peekaboo
# Author: Hasib Zunair
# Modified from https://github.com/valeoai/FOUND, see license below.

# Copyright 2022 - Valeo Comfort and Driving Assistance - Oriane Siméoni @ valeo.ai
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

"""Visualize model predictions"""

import os
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision

from PIL import Image
from model import PeekabooModel
from misc import load_config
from torchvision import transforms as T
from torchinfo import summary

from periphery import Periphery

NORMALIZE = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation of Periphery",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--img-path",
        type=str,
        default="data/examples/dinosaur.jpeg",
        help="Image path.",
    )
    parser.add_argument(
        "--model-weights",
        type=str,
        default="data/weights/peekaboo_decoder_weights_niter500.pt",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ukan_mini_DUTS-TR.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
    )
    args = parser.parse_args()

    # Saving dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Configuration
    config, _ = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    checkpoint = torch.load(config.distillation["checkpoint_path"],
                            map_location=torch.device('cpu') if not torch.cuda.is_available() else None)

    # Load the weights from the specified checkpoint path
    model.load_state_dict(checkpoint)
    print(f'In Demo Model Weight Loaded From {config.distillation["checkpoint_path"]} Successfully')

    # Print params
    summary(model, input_size=(1, 3, 224, 224))
    print(f"\n")

    # Load the image
    with open(args.img_path, "rb") as f:
        img = Image.open(f)
        img = img.convert("RGB")

        t = T.Compose([T.ToTensor(), NORMALIZE])
        img_t = t(img)[None, :, :, :]
        inputs = img_t.to(device)

    transform = torchvision.transforms.Resize((256, 256))
    inputs = transform(inputs)

    # Forward step
    with torch.no_grad():
        preds = model(inputs)
        print(f"Shape of output is {preds.shape}")

    sigmoid = nn.Sigmoid()
    h, w = img_t.shape[-2:]
    preds_up = F.interpolate(preds, size=(h, w), mode="bilinear", align_corners=False)
    print(f"Shape of output after interpolation is {preds_up.shape}")
    preds_up = (sigmoid(preds_up.detach()) > 0.5).squeeze(0).float()

    plt.figure()
    plt.imshow(img)
    plt.imshow(
        preds_up.cpu().squeeze().numpy(), "gray", interpolation="none", alpha=0.5
    )
    plt.axis("off")
    img_name = args.img_path
    img_name = img_name.split("/")[-1].split(".")[0]
    plt.savefig(
        os.path.join(args.output_dir, f"{img_name}-periphery.png"),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()
    print(f"Saved model prediction.")
