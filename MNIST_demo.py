import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from architectures import CNN2D


# ============================================================
# DEVICE
# ============================================================

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

print(f"Dummy input shape:  {dummy_data.shape}")
print(f"Dummy output shape: {dummy_output.shape}")


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

batch_size = 64

train_loader = DataLoader(
    mnist_trainset,
    batch_size=batch_size,
    shuffle=True
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

num_epochs = 10
learning_rate = 0.001

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

    print(
        f"Epoch {epoch + 1}/{num_epochs} "
        f"- Loss: {average_loss:.4f}"
    )


print("Model trained.")


# ============================================================
# EVALUATION
# ============================================================

print("\nEvaluating model...")

model.eval()

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
        logits = model(images)

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