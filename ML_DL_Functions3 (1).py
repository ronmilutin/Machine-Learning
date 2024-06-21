import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
def ID1():
    '''
        Personal ID of the first student.
    '''
    # Insert your ID here
    return 316389584

def ID2():
    '''
        Personal ID of the second student. Fill this only if you were allowed to submit in pairs, Otherwise leave it zeros.
    '''
    # Insert your ID here
    return 000000000

class CNN(nn.Module):
    def __init__(self): # Do NOT change the signature of this function
       # Call the constructor of the parent class (nn.Module)
        super(CNN, self).__init__()
        
        # Set the value of n (number of channels)
        self.n = 8  # You can adjust this value
        
        # Set the kernel size
        kernel_size = 3
        
        # Calculate padding to maintain the same height and width
        padding = (kernel_size - 1) // 2
        
        # Define the first convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.n, kernel_size=kernel_size, padding=padding)
        
        # Define the second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=self.n, out_channels=2*self.n, kernel_size=kernel_size, padding=padding)
        
        # Define the third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=2*self.n, out_channels=4*self.n, kernel_size=kernel_size, padding=padding)
        
        # Define the fourth convolutional layer
        self.conv4 = nn.Conv2d(in_channels=4*self.n, out_channels=8*self.n, kernel_size=kernel_size, padding=padding)
        
        # Define downsampling layers (2x2 max pooling)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Define fully-connected layers
        self.fc1 = nn.Linear(8*self.n * 28 * 14, 100)  # Adjust the input size based on your downsampling strategy
        self.fc2 = nn.Linear(100, 2)

    # Define the forward method to specify the forward pass of the model
    def forward(self, inp):
        # First convolutional layer
        out = self.conv1(inp)
        out = F.relu(out)  # Apply ReLU activation function
        out = self.pool(out)  # Apply 2x2 max pooling
        
        # Second convolutional layer
        out = self.conv2(out)
        out = F.relu(out)
        out = self.pool(out)
        
        # Third convolutional layer
        out = self.conv3(out)
        out = F.relu(out)
        out = self.pool(out)
        
        # Fourth convolutional layer
        out = self.conv4(out)
        out = F.relu(out)
        out = self.pool(out)
        
        # Flatten the output for fully-connected layers
        out = out.reshape(-1,8 * self.n * 28 * 14 )
        
        # Fully-connected layers
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        
        return out

class CNNChannel(nn.Module):
    def __init__(self):# Do NOT change the signature of this function
        super(CNNChannel, self).__init__()
        
        # Set the value of n (number of channels)
        self.n = 8  # You can adjust this value
        
        # Set the kernel size
        kernel_size = 3
        
        # Calculate padding to maintain the same height and width
        padding = (kernel_size - 1) // 2
        
        # Define the first convolutional layer
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=self.n, kernel_size=kernel_size, padding=padding)
        
        # Define the second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=self.n, out_channels=2*self.n, kernel_size=kernel_size, padding=padding)
        
        # Define the third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=2*self.n, out_channels=4*self.n, kernel_size=kernel_size, padding=padding)
        
        # Define the fourth convolutional layer
        self.conv4 = nn.Conv2d(in_channels=4*self.n, out_channels=8*self.n, kernel_size=kernel_size, padding=padding)
        
        # Define downsampling layers (2x2 max pooling)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Define fully-connected layers
        self.fc1 = nn.Linear(8*self.n * 14 * 14, 100)  # Adjust the input size based on your downsampling strategy
        self.fc2 = nn.Linear(100, 2)

    def forward(self, inp):  # Do NOT change the signature of this function
        # Manipulate the image: concatenate left and right shoes along the channel dimension
        manipulated_inp = torch.cat([inp[:, :, :224, :], inp[:, :, 224:, :]], dim=1)
        
        # First convolutional layer
        out = self.conv1(manipulated_inp)
        out = F.relu(out)  # Apply ReLU activation function
        out = self.pool(out)  # Apply 2x2 max pooling
        
        # Second convolutional layer
        out = self.conv2(out)
        out = F.relu(out)
        out = self.pool(out)
        
        # Third convolutional layer
        out = self.conv3(out)
        out = F.relu(out)
        out = self.pool(out)
        
        # Fourth convolutional layer
        out = self.conv4(out)
        out = F.relu(out)
        out = self.pool(out)
        
        # Flatten the output for fully-connected layers
        out = out.reshape(-1,8 * self.n * 14 * 14 )
        
        # Fully-connected layers
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        
        return out