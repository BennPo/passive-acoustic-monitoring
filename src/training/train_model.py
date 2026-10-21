from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from .dataset import create_loaders
from .model import OceanDenoiser


def train_model(model, train_loader, validation_loader, num_epoch=200):
    """Train a supplied model that maps noisy waveforms to matching waveforms."""
    if num_epoch < 1:
        raise ValueError("num_epoch must be positive")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Construct the optimiser after moving the model to its device.
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    criterion = nn.L1Loss()
    print("Device:", device)

    for epoch in range(num_epoch):
        # Training phase: calculate gradients and update the model.
        model.train()
        total_loss = 0.0
        train_examples = 0

        for batch_number, (inputs, targets) in enumerate(train_loader, start=1):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            if outputs.shape != targets.shape:
                raise ValueError("Model outputs and targets must have the same shape")
            loss = criterion(outputs, targets)
            if not torch.isfinite(loss).item():
                raise ValueError("Training loss is not finite; check the audio values")
            loss.backward()
            optimizer.step()

            # Weight each batch by its size, including a short final batch.
            batch_size = inputs.shape[0]
            total_loss += loss.item() * batch_size
            train_examples += batch_size
            if batch_number == 1 or batch_number % 50 == 0:
                print(
                    f"  Training batch {batch_number}: "
                    f"mean loss = {total_loss / train_examples:.6f}", flush=True,
                )

        if train_examples == 0:
            raise ValueError("The training loader produced no examples")
        avg_train_loss = total_loss / train_examples

        # Validation phase: measure loss without updating the model.
        model.eval()
        validation_loss = 0.0
        validation_examples = 0

        with torch.no_grad():
            for batch_number, (inputs, targets) in enumerate(validation_loader, start=1):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                if outputs.shape != targets.shape:
                    raise ValueError("Model outputs and targets must have the same shape")
                loss = criterion(outputs, targets)
                if not torch.isfinite(loss).item():
                    raise ValueError("Validation loss is not finite; check the audio values")

                batch_size = inputs.shape[0]
                validation_loss += loss.item() * batch_size
                validation_examples += batch_size
                if batch_number == 1 or batch_number % 50 == 0:
                    print(
                        f"  Validation batch {batch_number}: "
                        f"mean loss = {validation_loss / validation_examples:.6f}", flush=True,
                    )

        if validation_examples == 0:
            raise ValueError("The validation loader produced no examples")
        avg_validation_loss = validation_loss / validation_examples

        print(
            f"Epoch {epoch + 1}/{num_epoch}: "
            f"training loss = {avg_train_loss:.6f}, "
            f"validation loss = {avg_validation_loss:.6f}",
            flush=True,
        )
        scheduler.step()

    return model


if __name__ == "__main__":
    num_epoch = 5
    torch.set_num_threads(min(4, torch.get_num_threads()))
    train_loader, validation_loader = create_loaders(batch_size=2)

    # Show every selected recording before starting the run.
    for split_name, loader in (
        ("Training", train_loader), ("Validation", validation_loader),
    ):
        print(f"{split_name} recordings: {len(loader.dataset.audio_files)}")
        for filepath in loader.dataset.audio_files:
            print(f"  {filepath.name}")
        print(f"{split_name} backgrounds: {len(loader.dataset.background_files)}")

    model = OceanDenoiser(base_channels=16)
    print(f"Starting {num_epoch} epoch with batch size {train_loader.batch_size}.", flush=True)
    trained_model = train_model(model, train_loader, validation_loader, num_epoch=num_epoch)

    # Keep the learned weights and settings for a later listening/inference step.
    project_root = Path(__file__).resolve().parents[2]
    checkpoint_path = project_root / "outputs/ocean_denoiser.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": trained_model.state_dict(),
            "model_config": {
                "base_channels": trained_model.base_channels,
                "n_fft": trained_model.n_fft,
                "hop_length": trained_model.hop_length,
            },
            "sample_rate": train_loader.dataset.sample_rate,
            "epochs_completed": num_epoch,
        },
        checkpoint_path,
    )
    print(f"Saved trained weights to {checkpoint_path}")
