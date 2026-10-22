from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset
from sklearn.model_selection import train_test_split

from .create_mixtures import give_noise


class WhaleDataset(IterableDataset):
    def __init__(self, audio_files, background_files, sample_rate=8000, training=False):
        super().__init__()

        self.audio_files = list(audio_files)
        self.background_files = list(background_files)
        self.sample_rate = sample_rate
        self.training = training

        if not self.audio_files:
            raise ValueError("No audio files supplied")

    def __iter__(self):
        for filepath in self.audio_files:
            for mixture, clean, _ in give_noise(
                filepath,
                self.background_files,
                target_sample_rate=self.sample_rate,
                training=self.training,
            ):
                yield (
                    torch.from_numpy(mixture).unsqueeze(0),
                    torch.from_numpy(clean).unsqueeze(0),
                )


def create_loaders(batch_size=4):
    project_root = Path(__file__).resolve().parents[2]
    audio_files = sorted(
        filepath
        for filepath in (project_root / "data/clean_calls").iterdir()
        if filepath.is_file()
        and filepath.suffix.lower() in {".wav", ".flac"}
    )

    train_files, test_files = train_test_split(
        audio_files,
        test_size=0.2,
        random_state=42,
    )

    background_files = sorted(
        filepath
        for filepath in (project_root / "data/backgrounds").iterdir()
        if filepath.is_file()
        and filepath.suffix.lower() in {".wav", ".flac"}
    )

    train_backgrounds, test_backgrounds = train_test_split(
        background_files,
        test_size=0.2,
        random_state=42,
    )

    train_dataset = WhaleDataset(
        train_files,
        train_backgrounds,
        training=True,
        )
    test_dataset = WhaleDataset(
        test_files,
        test_backgrounds,
        )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=0,
    )

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = create_loaders()

    inputs, targets = next(iter(train_loader))
    print("Noisy inputs:", inputs.shape)
    print("Clean targets:", targets.shape)
