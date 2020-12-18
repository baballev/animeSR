import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import os
import time
import copy
from PIL import Image
## GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# MNIST convolutional autoencoder

# convert data to torch.FloatTensor
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='dataset', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='dataset', train=False, download=True, transform=transform)
num_workers = 0
batch_size = 20
trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # encoder layers
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1) # IN: [1, 28, 28], OUT: [16, 28,28] (thx to padding)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1) # IN: [16, 14, 14], OUT: [8, 14, 14]
        self.pool = nn.MaxPool2d(2, 2)

        # decoder layers
        self.conv4 = nn.Conv2d(4, 16, 3, padding=1) # IN: [8, 14, 14], OUT: [16, 14, 14]
        self.conv5 = nn.Conv2d(16, 1, 3, padding=1) # IN: [16, 28, 28], OUT: [1, 28, 28]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation: [8, 7, 7]
        # decoder
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.relu(self.conv4(x))
        # upsample again, output should have a sigmoid applied
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.sigmoid(self.conv5(x))
        return x

##
def trainModel(model, loss_function, optimizer, epochs_nb):
    since = time.time()
    best_model = copy.deepcopy(model.state_dict())
    best_loss = 6500000.0
    train_size = len(train_data)
    valid_size = len(test_data)
    print("Training start")
    for epoch in range(epochs_nb):
        # Verbose 1
        print("Epoch [" + str(epoch+1) + " / " + str(epochs_nb) + "]")
        print("-" * 10)

        # Training
        running_corrects = 0
        running_loss = 0.0
        verbose_loss = 0.0
        for i, data in enumerate(trainloader):
            inp = data[0].to(device)
            optimizer.zero_grad()
            output = model(inp)
            loss = loss_function(output, inp) # AE => label = input
            loss.backward()
            optimizer.step()
            if i%100 == 0:
                print("Batch " + str(i) + " / " + str(int(train_size/batch_size)))
            running_loss += loss.item()
            verbose_loss += loss.item()
            if i% 100 == 0 and i !=0:
                print("Loss over last 100 batches: " + str(verbose_loss/(100*batch_size)))
                verbose_loss = 0.0

        # Verbose 2
        epoch_loss = running_loss / train_size
        print(" ")
        print(" ")
        print(" ")
        print("****************")
        print('Training Loss: {:.4f}'.format(epoch_loss))

        # Validation
        running_corrects = 0
        running_loss = 0.0
        verbose_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(testloader):
                inp = data[0].to(device)
                output = model(inp)
                loss = loss_function(output, inp)
                running_loss += loss.item()

            # Verbose 3
            epoch_loss = running_loss / valid_size
            print('Validation Loss: {:.4f}'.format(epoch_loss))

        # Copy the model if it gets better
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model = copy.deepcopy(model.state_dict())
    # Verbose 4
    time_elapsed = time.time() - since
    print("Training finished in {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))
    print("Best validation loss: " + str(best_loss))


    model.load_state_dict(best_model) # In place anyway
    return model # Returning just in case

model = ConvAutoencoder()
model.to(device)
print(model)
loss_function = nn.BCELoss() # Binary Cross Entropy:
# With N = batch_size, BCE(x,y) = {l1, ..., lN}^T, ln = yn*log(xn) + (1-yn)*log(1 - xn)

optimizer = optim.Adam(model.parameters(), lr=0.01, amsgrad=True)

trainModel(model, loss_function, optimizer, 10)

def makeCheckpoint(model, save_path):
    torch.save(model.state_dict(), save_path)
    print("Weights saved to: " + save_path)
    return

save_path = 'E:/Programmation/Python/HArchiver/weights/MNIST_convAE_smaller_lowerLR_10ep.pth'
makeCheckpoint(model, save_path)


## Reconstructing the image
load_path = 'E:/Programmation/Python/HArchiver/weights/MNIST_convAE_smaller_higherLR_10ep.pth'

model = ConvAutoencoder()
model.load_state_dict(torch.load(load_path))

# obtain one batch of testing images
with torch.no_grad():
    dataiter = iter(testloader)
    images, _ = dataiter.next() # images in [0, 1]
    built = model(images)
    built = built.numpy()
    images = images.numpy()
    # get one image from the batch
    img = np.squeeze(images[0])
    blt = np.squeeze(built[0])
    fig = plt.figure(figsize = (5,5))
    fig.add_subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    fig.add_subplot(1, 2, 2)
    plt.imshow(blt, cmap='gray')
    plt.show()


















