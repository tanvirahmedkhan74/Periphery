import torch
import torch.nn as nn
from sympy import pprint
from ukan.Seg_UKAN.archs import UKAN


class Periphery(nn.Module):
    def __init__(self, encoder_config, pretrained_weights_path=None, freeze=False):
        """
        Initialize the Periphery Model with a UKAN encoder and a saliency detection decoder.

        Args:
        - encoder_config (dict): Configuration dictionary for initializing the UKAN encoder.
        - pretrained_weights_path (str): Path to the pre-trained weights for the UKAN encoder.
        - freeze (bool): If True, freeze the encoder parameters.
        """
        super(Periphery, self).__init__()

        # Initialize the UKAN encoder
        self.encoder = UKAN(**encoder_config)

        # Automatically determine device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # Dynamically determine the encoder's output dimension
        self.output_dim = self._determine_output_dim()

        # Modified decoder for saliency detection
        self.decoder = nn.Sequential(
            nn.Conv2d(self.output_dim, self.output_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),  # Non-linear activation
            nn.BatchNorm2d(self.output_dim // 2),  # Normalize intermediate features
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # Upsample to double the resolution
            nn.Conv2d(self.output_dim // 2, 1, kernel_size=3, padding=1),  # Final saliency map (1-channel)
            nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False)  # Ensure output is 32x32
        )

        # Load pre-trained weights for the encoder if provided
        if pretrained_weights_path is not None:
            checkpoint = torch.load(pretrained_weights_path,
                                    map_location=torch.device('cpu') if not torch.cuda.is_available() else None)
            self.load_state_dict(checkpoint)
            print(f'Periphery Weights Loaded Successfully from {pretrained_weights_path}')

        # Optionally freeze the encoder
        if freeze:
            self.freeze_encoder()

        self.to(self.device)

    def _determine_output_dim(self):
        """
        Dynamically determine the encoder's output dimension by passing a dummy input through the encoder.
        """
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 256, 256).to(self.device)  # Adjust shape if input size differs
            encoder_outputs = self.encoder(dummy_input)  # Should return multiple outputs
            return encoder_outputs.shape[1]  # Output channels from the first encoder output

    def freeze_encoder(self):
        """
        Freeze the encoder layers, preventing their weights from being updated during training.
        """
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Encoder layers have been frozen.")

    def forward(self, x):
        """
        Forward pass for the student model.

        Args:
        - x (torch.Tensor): Input tensor (image or batch of images).

        Returns:
        - torch.Tensor: Decoder output logits.
        """
        # print(x.shape)
        x = x.to(self.device)  # Ensure input is on the same device as the model

        # Pass input through encoder (feature extraction)
        encoder_features = self.encoder(x)
        # print('Encoder feats shape: ', encoder_features.shape)

        # Decode features to generate output
        logits = self.decoder(encoder_features)

        return logits

    @torch.no_grad()
    def load_decoder_weights(self, weights_path):
        """
        Load pre-trained weights for the decoder.

        Args:
        - weights_path (str): Path to the weights file.
        """
        print(f"Loading decoder weights from {weights_path}.")
        state_dict = torch.load(weights_path, map_location=self.device)
        self.decoder.load_state_dict(state_dict["decoder"])
        self.decoder.eval()
        self.decoder.to(self.device)  # Ensure the decoder is on the correct device

    @torch.no_grad()
    def save_decoder_weights(self, save_path):
        """
        Save decoder weights to a file.

        Args:
        - save_path (str): Path to save the weights file.
        """
        torch.save({"decoder": self.decoder.state_dict()}, save_path)
        print(f"Decoder weights saved at {save_path}.")