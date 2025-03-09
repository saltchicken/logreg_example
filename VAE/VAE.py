import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Sample structured dataset
data = pd.DataFrame({
    'age': [25, 32, 40, 29, 50, 45, 36, 23, 60, 38],
    'height_cm': [175, 180, 165, 170, 185, 178, 160, 172, 190, 176],
    'income': [50000, 60000, 55000, 52000, 70000, 65000, 48000, 51000, 75000, 62000]
})

# Normalize the data
data_mean = data.mean()
data_std = data.std()
data_normalized = (data - data_mean) / data_std

# Convert to tensor with the correct dtype (ensure float32)
data_tensor = torch.tensor(data_normalized.values, dtype=torch.float32)

# Define a simple VAE architecture
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc21 = nn.Linear(128, latent_dim)  # Mean of latent space
        self.fc22 = nn.Linear(128, latent_dim)  # Log-variance of latent space
        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc4 = nn.Linear(128, input_dim)    # No sigmoid here, linear output

    def encode(self, x):
        x = x.float()  # Ensure input is float32
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = z.float()  # Ensure latent vector is float32
        h3 = torch.relu(self.fc3(z))
        return self.fc4(h3)  # Linear activation, no squashing

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Define the VAE loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KL

# Set up parameters
input_dim = data_normalized.shape[1]
latent_dim = 2  # Latent space dimensionality

# Initialize model and optimizer
model = VAE(input_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Prepare data loader
batch_size = 2
train_loader = DataLoader(TensorDataset(data_tensor), batch_size=batch_size, shuffle=True)

# Train the model
epochs = 1000
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_idx, (data_batch,) in enumerate(train_loader):
        data_batch = data_batch.view(-1, input_dim)  # Flatten batch
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data_batch)
        loss = loss_function(recon_batch, data_batch, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch {epoch}: Train loss {train_loss / len(train_loader)}')

# Generate new synthetic data by sampling from the latent space
model.eval()
with torch.no_grad():
    # Sample from latent space (using standard normal distribution)
    z = torch.randn(5, latent_dim)  # Generate 5 synthetic samples
    generated_data = model.decode(z)

# Post-process the generated data (denormalize)
generated_data = generated_data * data_std.values + data_mean.values

# Convert generated data to a DataFrame for better visualization
generated_data_df = pd.DataFrame(generated_data.numpy(), columns=data.columns)

# Print the generated synthetic data
print("Generated Synthetic Data:")
print(generated_data_df)

