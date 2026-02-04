# Copyright 2026 Hendrik Sauer
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

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class LightweightUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, base_filters=16):
        """
        Args:
            n_channels (int): Number of input channels (e.g., 3 for RGB, 1 for Grayscale).
            n_classes (int): Number of output classes (1 for binary segmentation).
            base_filters (int): Starting number of filters (controls model width).
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # --- Encoder (Downsampling) ---
        self.inc = DoubleConv(n_channels, base_filters)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_filters, base_filters * 2)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_filters * 2, base_filters * 4)
        )
        
        # Bottom of the U
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_filters * 4, base_filters * 8)
        )

        # --- Decoder (Upsampling) ---
        # Up 1
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up1 = DoubleConv(base_filters * 8 + base_filters * 4, base_filters * 4)

        # Up  2
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up2 = DoubleConv(base_filters * 4 + base_filters * 2, base_filters * 2)

        # Up 3
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up3 = DoubleConv(base_filters * 2 + base_filters, base_filters)

        # Final Classifier
        self.outc = nn.Conv2d(base_filters, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)  # Bottleneck

        # Decoder with Skip Connections
        
        # Skip connection from x3
        x = self.up1(x4)
        # Handle slight shape mismatches if input isn't perfect multiple of 16
        if x.size() != x3.size():
            x = F.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x3, x], dim=1)
        x = self.conv_up1(x)

        # Skip connection from x2
        x = self.up2(x)
        if x.size() != x2.size():
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x2, x], dim=1)
        x = self.conv_up2(x)

        # Skip connection from x1
        x = self.up3(x)
        if x.size() != x1.size():
            x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x1, x], dim=1)
        x = self.conv_up3(x)

        logits = self.outc(x)
        return logits

# --- Usage Example ---
if __name__ == "__main__":
    # Create model: 3 input channels (RGB), 1 output channel (Binary Mask)
    # Using 16 base filters keeps it very light (~400k params vs standard ~30M)
    model = LightweightUNet(n_channels=3, n_classes=1, base_filters=16)
    
    # Dummy input: Batch size 1, 3 Channels, 256x256 Image
    x = torch.randn(1, 3, 256, 256)
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}") # Should be [1, 1, 256, 256]
    
    # Calculate params
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
