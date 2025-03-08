import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from safetensors.torch import save_file, load_file
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Fully connected layers
        self.fc1 = nn.Linear(28*28, 128)  # Input layer (28x28 image size)
        self.fc2 = nn.Linear(128, 64)     # Hidden layer
        self.fc3 = nn.Linear(64, 10)      # Output layer (10 classes)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the image
        x = torch.relu(self.fc1(x))  # Apply ReLU activation
        x = torch.relu(self.fc2(x))  # Apply ReLU activation
        x = self.fc3(x)             # Output layer (raw scores)
        return x

def train(output_filename):
# Download and load the Fashion MNIST dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)


# Classes in the Fashion MNIST dataset
    classes = ['T-shirt', 'Pullover', 'Trouser', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Define a simple Feedforward Neural Network (FNN)

# Initialize the model
    model = Net()
    model.to(device)
# Loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # For multi-class classification
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Train the model
    num_epochs = 5
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 batches
                print(f"[Epoch {epoch+1}, Batch {i+1}] Loss: {running_loss / 100:.3f}")
                running_loss = 0.0

    print("Finished Training")

# Evaluate the model on the test data
    correct = 0
    total = 0
    with torch.no_grad():  # No need to compute gradients during evaluation
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get the index of the max output
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on the test set: {100 * correct / total:.2f}%")

# Save the trained model
    # torch.save(model.state_dict(), f"{output_filename}.pth")
    save_file(model.state_dict(), f"{output_filename}.safetensors")
    print("Model saved!")

def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net()
    model.to(device)
    # model.load_state_dict(torch.load("fashion_mnist_model.pth"))
    state_dict = load_file("fashion_mnist_model.safetensors")
    model.load_state_dict(state_dict)
    model.eval()

    image = Image.open("test2.jpg")  # Load an image (make sure it's a 28x28 image)
    image = image.convert("L")
    image.show()
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize to match the model's input size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize (if used in training)
    ])

# Apply transformations to the image
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension: (1, 1, 28, 28)
    image_tensor = image_tensor.to(device)  # Move the tensor to the same device as the model

# Make a prediction
    with torch.no_grad():  # No need to track gradients for inference
        output = model(image_tensor)  # Forward pass through the model
        _, predicted_class = torch.max(output, 1)  # Get the index of the max logit (predicted class)

# Print the predicted class
    print(f"Predicted class: {predicted_class.item()}")

# def resize(image_path, output_path, size=(28, 28)):
#     image = Image.open(image_path)  # Open the image using PIL
#     resized_image = image.resize(size)  # Resize the image to the desired size
#     resized_image.save(output_path)  # Save the resized image
#
# def convert(image_path, output_path, size=(28, 28)):
#     image = Image.open(image_path)
#     image.convert("RGB").save(output_path, "JPEG")
#
def show_images(num_images=6):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)
    dataiter = iter(trainloader)
    images, labels = next(dataiter)


    plt.figure(figsize=(10, 5))
    for i in range(num_images):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i].squeeze(), cmap="gray")  # Remove extra dimension and set grayscale
        plt.title(f"Label: {labels[i].item()}")
        plt.axis("off")
    plt.show()

if __name__ == "__main__":
    # train("fashion_mnist_model")
    predict()
    # show_images()

