"""A compact spectrogram U-Net that accepts and returns mono waveforms."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    """Learn local spectrogram features without depending on batch statistics."""

    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.LeakyReLU(0.1),
        )


class OceanDenoiser(nn.Module):
    """Estimate an attenuation mask, then reconstruct audio with the input phase.

    Input and output: [batch, 1, samples]. The current dataset supplies 8 kHz
    audio. Use the same sample rate when training and using a trained model.
    """

    def __init__(self, base_channels=16, n_fft=1024, hop_length=256):
        super().__init__()
        if base_channels < 4 or base_channels % 4:
            raise ValueError("base_channels must be a positive multiple of 4")
        if n_fft < 4 or n_fft % 2:
            raise ValueError("n_fft must be an even number of at least 4")
        if not 1 <= hop_length <= n_fft // 2:
            raise ValueError("hop_length must be between 1 and n_fft // 2")

        self.base_channels = base_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))

        # Encoder: look at increasingly broad time/frequency patterns.
        self.encoder1 = ConvBlock(1, base_channels)
        self.encoder2 = ConvBlock(base_channels, base_channels * 2)
        self.encoder3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool = nn.AvgPool2d(2)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)

        # Decoder: restore detail, joining matching encoder features as skips.
        self.decoder3 = ConvBlock(base_channels * 12, base_channels * 4)
        self.decoder2 = ConvBlock(base_channels * 6, base_channels * 2)
        self.decoder1 = ConvBlock(base_channels * 3, base_channels)
        self.mask_head = nn.Conv2d(base_channels, 1, kernel_size=1)

        # Start near a gentle, almost constant mask: sigmoid(2) is about 0.88.
        # Small nonzero weights let gradients reach the encoder immediately.
        nn.init.normal_(self.mask_head.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.mask_head.bias, 2.0)

    @staticmethod
    def _upsample_and_join(features, skip):
        # Exact sizes handle odd dimensions such as 513 frequencies, 157 frames.
        features = F.interpolate(
            features, size=skip.shape[-2:], mode="bilinear", align_corners=False,
        )
        return torch.cat((features, skip), dim=1)

    def _predict_mask(self, features):
        frequency_count, frame_count = features.shape[-2:]
        # Three pooling stages need at least 8 bins/frames, even for short input.
        features = F.pad(
            features,
            (0, max(0, 8 - frame_count), 0, max(0, 8 - frequency_count)),
        )

        skip1 = self.encoder1(features)
        skip2 = self.encoder2(self.pool(skip1))
        skip3 = self.encoder3(self.pool(skip2))
        features = self.bottleneck(self.pool(skip3))

        features = self.decoder3(self._upsample_and_join(features, skip3))
        features = self.decoder2(self._upsample_and_join(features, skip2))
        features = self.decoder1(self._upsample_and_join(features, skip1))
        mask = torch.sigmoid(self.mask_head(features))
        return mask[:, :, :frequency_count, :frame_count]

    def forward(self, noisy_audio):
        if noisy_audio.ndim != 3 or noisy_audio.shape[1] != 1:
            raise ValueError("Expected audio shaped [batch, 1, samples]")
        if noisy_audio.shape[0] == 0 or noisy_audio.shape[-1] == 0:
            raise ValueError("Audio must contain at least one example and sample")
        if not noisy_audio.is_floating_point():
            raise ValueError("Audio must contain floating-point amplitudes")

        waveforms = noisy_audio.squeeze(1)

        # 1. Keep the original complex STFT, including its phase.
        spectrum = torch.stft(
            waveforms,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )

        # 2. Normalise features only; the original spectrum keeps its amplitude.
        rms = waveforms.square().mean(dim=-1, keepdim=True).clamp_min(1e-8).sqrt()
        features = torch.log1p(spectrum.abs() / rms.unsqueeze(-1)).unsqueeze(1)

        # 3. Learn a multiplier between 0 and 1 for each frequency/time bin.
        mask = self._predict_mask(features).squeeze(1)
        estimated_spectrum = spectrum * mask

        # 4. Match the transform settings and preserve the exact waveform length.
        estimate = torch.istft(
            estimated_spectrum,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            length=waveforms.shape[-1],
        )
        return estimate.unsqueeze(1)


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = OceanDenoiser(base_channels=16).eval()

    example = torch.randn(2, 1, 40000) * 0.1
    with torch.no_grad():
        estimate = model(example)

    print("Feature channels:", [16, 32, 64, 128])
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Input shape:", example.shape)
    print("Output shape:", estimate.shape)
    print("Untrained model check complete; these weights have not learned denoising yet.")
