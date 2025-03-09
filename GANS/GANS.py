import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True  # Optimize cuDNN performance

# Define Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 28 * 28)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.tanh(self.fc4(x))
        return x.view(x.size(0), 1, 28, 28)  # Reshape to image format

# Define Discriminator (Remove Sigmoid)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)  # No sigmoid activation
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)  # No Sigmoid

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
generator = Generator().to(device, memory_format=torch.channels_last)
discriminator = Discriminator().to(device, memory_format=torch.channels_last)

# Use BCEWithLogitsLoss (Numerically Stable)
criterion = nn.BCEWithLogitsLoss()

# Optimizers with fused operations for better GPU performance
lr = 0.0002
beta1 = 0.5
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999), fused=True)
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999), fused=True)

# Enable Mixed Precision Training
scaler = torch.amp.GradScaler(device=device)

# Load dataset with optimizations
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=8, pin_memory=True)

# Training Loop
num_epochs = 10000
for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(train_loader):
        print(f"Epoch {epoch}, Batch {i}")
        real_imgs = imgs.to(device, memory_format=torch.channels_last)
        batch_size = real_imgs.size(0)

        # Create labels
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        #### Train Discriminator ####
        optimizer_d.zero_grad()

        with torch.amp.autocast(device_type="cuda"):  # Enable Mixed Precision
            outputs_real = discriminator(real_imgs)
            d_loss_real = criterion(outputs_real, real_labels)

            z = torch.randn(batch_size, 100, device=device)
            fake_imgs = generator(z)
            outputs_fake = discriminator(fake_imgs.detach())
            d_loss_fake = criterion(outputs_fake, fake_labels)

            d_loss = d_loss_real + d_loss_fake

        # Scale loss and backpropagate
        scaler.scale(d_loss).backward()
        scaler.step(optimizer_d)
        scaler.update()

        #### Train Generator ####
        optimizer_g.zero_grad()

        with torch.amp.autocast(device_type="cuda"):  # Enable Mixed Precision
            outputs = discriminator(fake_imgs)
            g_loss = criterion(outputs, real_labels)

        # Scale loss and backpropagate
        scaler.scale(g_loss).backward()
        scaler.step(optimizer_g)
        scaler.update()

    # Print progress and save images at intervals
    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

        if epoch % 1000 == 0:
            with torch.no_grad():
                z = torch.randn(64, 100, device=device)
                generated_images = generator(z).cpu().numpy()
                plt.figure(figsize=(8, 8))
                for i in range(generated_images.shape[0]):
                    plt.subplot(8, 8, i+1)
                    plt.imshow(generated_images[i, 0], cmap='gray')
                    plt.axis('off')
                plt.tight_layout()
                plt.savefig(f'gan_generated_image_epoch_{epoch}.png')
                plt.close()

