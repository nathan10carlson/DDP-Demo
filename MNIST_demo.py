import torch

import torch.nn as nn
import matplotlib.pyplot as plt
import argparse

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from architectures import CNN2D
from utils import get_distributed_info

parser = argparse.ArgumentParser()

parser.add_argument(
    "--learning-rate",
    type=float,
    default=0.001
)

parser.add_argument(
    "--epochs",
    type=int,
    default=10
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=64
)

parser.add_argument(
    "--local-rank",
    "--local_rank",
    type=int,
    default=0
)

args = parser.parse_args()

distributed, rank, local_rank, world_size = get_distributed_info()

if distributed:
    dist.init_process_group(
    backend="nccl",
    init_method="env://"
    ) # this connects them

print(f"Distributed: {distributed}")
print(f"Rank: {rank}")
print(f"Local rank: {local_rank}")
print(f"World size: {world_size}")

learning_rate = args.learning_rate
num_epochs = args.epochs
batch_size = args.batch_size

# ============================================================
# DEVICE
# ============================================================
if distributed:
    device = torch.device(f"cuda:{local_rank}")
else:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

print(f"Using device: {device}")


# ============================================================
# MODEL
# ============================================================



model = CNN2D().to(device)

# Quick sanity check
dummy_data = torch.rand(1, 1, 28, 28).to(device)
dummy_output = model(dummy_data)
# this also sets the lazy linear layer
print(f"Dummy input shape:  {dummy_data.shape}")
print(f"Dummy output shape: {dummy_output.shape}")

if distributed:
    model = DDP(
        model,
        device_ids=[local_rank]
    )




# ============================================================
# DATASET
# ============================================================

transform = transforms.ToTensor()

mnist_trainset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

mnist_testset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

print(f"Training samples: {len(mnist_trainset)}")
print(f"Testing samples:  {len(mnist_testset)}")


# ============================================================
# DATALOADERS
# ============================================================

if distributed:
    train_sampler = DistributedSampler(
        mnist_trainset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
else:
    train_sampler = None

train_loader = DataLoader(
    mnist_trainset,
    batch_size=batch_size,
    shuffle=(train_sampler is None), # sampler handles shuffling in distrib mode
    sampler=train_sampler # 
) 

test_loader = DataLoader(
    mnist_testset,
    batch_size=batch_size,
    shuffle=False
)

# Inspect one batch
images, labels = next(iter(train_loader))

print(f"Batch image shape: {images.shape}")
print(f"Batch label shape: {labels.shape}")


# ============================================================
# TRAINING SETUP
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)


# ============================================================
# TRAINING
# ============================================================

model.train()

for epoch in range(num_epochs):

    if distributed:
        train_sampler.set_epoch(epoch)

    epoch_loss = 0.0

    for images, labels in train_loader:

        # Move the current batch to the same device as the model
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        logits = model(images)

        # Compare predictions against true labels
        loss = criterion(logits, labels)

        # Remove gradients left over from the previous batch
        optimizer.zero_grad()

        # Compute gradients
        loss.backward()

        # Update model parameters
        optimizer.step()

        # loss.item() gives us the loss as a normal Python number
        epoch_loss += loss.item()

    average_loss = epoch_loss / len(train_loader)
    if rank == 0:
        print(
            f"Epoch {epoch + 1}/{num_epochs} "
            f"- Loss: {average_loss:.4f}"
        )

if distributed:
    dist.barrier()
if rank == 0:
    print("Model trained.")


# ============================================================
# EVALUATION
# ============================================================

if rank == 0:
    print("\nEvaluating model...")

    eval_model = model.module if distributed else model
    eval_model.eval()

    total = 0
    correct_guesses = 0

    all_incorrect_images = []
    all_incorrect_true = []
    all_incorrect_pred = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = eval_model(images)

            # Pick the class with the largest logit
            predictions = torch.argmax(logits, dim=1)

            # Determine which predictions were correct
            correct = predictions == labels

            correct_guesses += correct.sum().item()
            total += labels.shape[0]

            # Save incorrect predictions for visualization
            incorrect_mask = ~correct

            all_incorrect_images.append(
                images[incorrect_mask].cpu()
            )

            all_incorrect_true.append(
                labels[incorrect_mask].cpu()
            )

            all_incorrect_pred.append(
                predictions[incorrect_mask].cpu()
            )


    accuracy = correct_guesses / total

    print(f"Total samples:   {total}")
    print(f"Correct guesses: {correct_guesses}")
    print(f"Total accuracy:  {accuracy:.4f}")


    # ============================================================
    # COLLECT INCORRECT PREDICTIONS
    # ============================================================

    all_incorrect_images = torch.cat(all_incorrect_images)
    all_incorrect_true = torch.cat(all_incorrect_true)
    all_incorrect_pred = torch.cat(all_incorrect_pred)

    print(f"Incorrect predictions: {len(all_incorrect_images)}")


    # ============================================================
    # VISUALIZE 9 INCORRECT PREDICTIONS
    # ============================================================

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))

    for i, ax in enumerate(axes.flat):

        image = all_incorrect_images[i].squeeze(0)

        true_label = all_incorrect_true[i].item()
        predicted_label = all_incorrect_pred[i].item()

        ax.imshow(image, cmap="gray")
        ax.set_title(
            f"True: {true_label} | Pred: {predicted_label}"
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("incorrect.png")
    plt.close()
    print("Saved incorrect predictions to incorrect.png")

if distributed:
    dist.barrier()
    dist.destroy_process_group()


