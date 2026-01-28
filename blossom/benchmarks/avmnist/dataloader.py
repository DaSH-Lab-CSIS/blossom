# Standard library imports
import random
from typing import Callable, Dict, Optional, Tuple, Type, Union

# Torch imports
import torch
import torchaudio
from torchvision import transforms
from PIL import Image
from datasets import Dataset as HFDataset

# Local imports
from blossom.dataloader import MultimodalDataset


# ============================================================================
# Image Transforms
# ============================================================================

image_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)


# ============================================================================
# Dataset Class
# ============================================================================

class AVMNISTDataset(MultimodalDataset):
    """Audio-Visual MNIST dataset pairing MNIST images with spoken digits."""

    def __init__(
        self,
        dataset: HFDataset,
        train_split: bool,
        audio_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(dataset, train_split)

        self.image_transform = image_transform
        self.audio_transform = audio_transform

        # Audio parameters
        self.sr = 48000
        self.duration = 1500
        self.channel = 2
        self.shift_pct = 0.4

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Image.Image]]:
        item = self.dataset[idx]

        # Process image
        image = item["image"]
        if self.image_transform:
            image = self.image_transform(image)

        # Process audio
        audio = item["audio"]
        audio_array = audio["array"]
        sample_rate = audio["sampling_rate"]
        waveform = torch.tensor(audio_array).float().unsqueeze(0)

        # Audio pipeline
        aud = (waveform, sample_rate)
        reaud = AudioUtil.resample(aud, self.sr)
        rechan = AudioUtil.rechannel(reaud, self.channel)
        dur_aud = AudioUtil.pad_trunc(rechan, self.duration)

        if self.is_train_split:
            dur_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)

        # Create spectrogram
        if self.audio_transform is None:
            sgram = AudioUtil.spectrogram(dur_aud, n_mels=64, n_fft=1024)
            if self.is_train_split:
                sgram = AudioUtil.spec_augment(
                    sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2
                )
            processed_audio = sgram
        else:
            processed_audio = self.audio_transform(dur_aud[0])

        label = item["label"]

        return {
            "image": image,
            "audio": processed_audio,
            "label": torch.tensor(label, dtype=torch.long),
        }


# ============================================================================
# Audio Utilities
# ============================================================================

class AudioUtil:
    """Audio processing utilities for waveform manipulation."""

    @staticmethod
    def open(audio_file: str) -> Tuple[torch.Tensor, int]:
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)

    @staticmethod
    def rechannel(
        aud: Tuple[torch.Tensor, int], 
        new_channel: int
    ) -> Tuple[torch.Tensor, int]:
        sig, sr = aud

        if sig.shape[0] == new_channel:
            return aud

        if new_channel == 1:
            resig = sig[:1, :]
        else:
            resig = torch.cat([sig, sig]) if sig.shape[0] == 1 else sig

        return (resig, sr)

    @staticmethod
    def resample(
        aud: Tuple[torch.Tensor, int], 
        newsr: int
    ) -> Tuple[torch.Tensor, int]:
        sig, sr = aud

        if sr == newsr:
            return aud

        num_channels = sig.shape[0]
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1, :])
        if num_channels > 1:
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:, :])
            resig = torch.cat([resig, retwo])

        return (resig, newsr)

    @staticmethod
    def pad_trunc(
        aud: Tuple[torch.Tensor, int], 
        max_ms: int
    ) -> Tuple[torch.Tensor, int]:
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr // 1000 * max_ms

        if sig_len > max_len:
            sig = sig[:, :max_len]
        elif sig_len < max_len:
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            max_noise = sig.max()
            min_noise = sig.min()
            pad_begin = (max_noise - min_noise) * torch.rand(
                (num_rows, pad_begin_len)
            ) + min_noise
            pad_end = (max_noise - min_noise) * torch.rand(
                (num_rows, pad_end_len)
            ) + min_noise

            sig = torch.cat((pad_begin, sig, pad_end), 1)

        return (sig, sr)

    @staticmethod
    def time_shift(
        aud: Tuple[torch.Tensor, int], 
        shift_limit: float
    ) -> Tuple[torch.Tensor, int]:
        sig, sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)

    @staticmethod
    def spectrogram(
        aud: Tuple[torch.Tensor, int],
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_len: Optional[int] = None,
    ) -> torch.Tensor:
        sig, sr = aud
        top_db = 80

        spec = torchaudio.transforms.MelSpectrogram(
            sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels
        )(sig)
        spec = torchaudio.transforms.AmplitudeToDB(top_db=top_db)(spec)
        return spec

    @staticmethod
    def spec_augment(
        spec: torch.Tensor,
        max_mask_pct: float = 0.1,
        n_freq_masks: int = 1,
        n_time_masks: int = 1,
    ) -> torch.Tensor:
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = torchaudio.transforms.FrequencyMasking(int(freq_mask_param))(
                aug_spec, mask_value
            )

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = torchaudio.transforms.TimeMasking(int(time_mask_param))(
                aug_spec, mask_value
            )

        return aug_spec


# ============================================================================
# Factory Function
# ============================================================================

def get_dataset_class() -> Type:
    """Return the AVMNISTDataset class."""
    return AVMNISTDataset