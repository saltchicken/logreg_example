import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from safetensors.torch import save_file, load_file

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten feature maps
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

def train():
# Define transformations for the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

# Load the MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the CNN model

# Instantiate model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Testing the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    save_file(model.state_dict(), "cnn_model.safetensors")


def preprocess_image(image_path):
    # Define the transformations (same as the training data transformations)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale if the image is in color
        transforms.Resize((28, 28)),                  # Resize image to 28x28
        transforms.ToTensor(),                        # Convert image to tensor
        transforms.Normalize((0.5,), (0.5,))          # Normalize the image to [-1, 1] range
    ])
    
    # Load the image
    image = Image.open(image_path)
    
    # Apply the transformations
    image = transform(image)
    
    # Add a batch dimension (1 image in the batch)
    image = image.unsqueeze(0)  # Shape becomes (1, 1, 28, 28)
    
    return image

def predict(image_path):
# Load the model (make sure it's on the same device as during training)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)  # Assuming the model is defined as in the previous example

# Load the trained model's weights (if it's saved as a .pth file)
    # model.load_state_dict(torch.load('cnn_model.pth'))  # Provide the correct path to the saved model
    state_dict = load_file("cnn_model.safetensors")
    model.load_state_dict(state_dict)

    model.eval()  # Set the model to evaluation mode

# Preprocess the input image
    image = preprocess_image(image_path).to(device)

# Make the prediction
    with torch.no_grad():
        output = model(image)
        
        # Get the predicted class (digit)
        _, predicted_class = torch.max(output, 1)

        print(f"Predicted digit: {predicted_class.item()}")
if __name__ == "__main__":
    # train()
    predict("7.jpeg")
