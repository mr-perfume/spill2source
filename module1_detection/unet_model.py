"""
unet_model.py
Standard U-Net for binary SAR oil-slick segmentation.

Input:  (B, in_channels, H, W)   -- in_channels=2 for VV+VH
Output: (B, out_channels, H, W)  -- raw logits (no sigmoid applied here;
                                     train_unet.py uses BCEWithLogitsLoss,
                                     so keep it that way for numerical stability)
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x2"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Maxpool then DoubleConv"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """Upsample (transposed conv), concat skip connection, then DoubleConv"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        # handle any off-by-one size mismatch from odd input dims
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = nn.functional.pad(
            x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2]
        )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 2, out_channels: int = 1, base_ch: int = 32):
        """
        base_ch=32 keeps this trainable comfortably on a single GPU / modest CPU
        at 256x256 resolution. Bump to 64 if you have the VRAM and want more
        capacity; the merged_dataset.npz has ~15.9k samples which is enough
        to support a slightly larger model if val IoU plateaus early.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inc = DoubleConv(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.down4 = Down(base_ch * 8, base_ch * 16)

        self.up1 = Up(base_ch * 16, base_ch * 8)
        self.up2 = Up(base_ch * 8, base_ch * 4)
        self.up3 = Up(base_ch * 4, base_ch * 2)
        self.up4 = Up(base_ch * 2, base_ch)

        self.outc = nn.Conv2d(base_ch, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)      # base_ch,     256
        x2 = self.down1(x1)   # base_ch*2,   128
        x3 = self.down2(x2)   # base_ch*4,   64
        x4 = self.down3(x3)   # base_ch*8,   32
        x5 = self.down4(x4)   # base_ch*16,  16

        x = self.up1(x5, x4)  # base_ch*8,   32
        x = self.up2(x, x3)   # base_ch*4,   64
        x = self.up3(x, x2)   # base_ch*2,   128
        x = self.up4(x, x1)   # base_ch,     256

        return self.outc(x)   # out_channels, 256   (raw logits)


if __name__ == "__main__":
    # quick shape sanity check: python unet_model.py
    model = UNet(in_channels=2, out_channels=1)
    dummy = torch.randn(4, 2, 256, 256)
    out = model(dummy)
    print(f"input:  {tuple(dummy.shape)}")
    print(f"output: {tuple(out.shape)}")
    assert out.shape == (4, 1, 256, 256), "unexpected output shape"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}")
