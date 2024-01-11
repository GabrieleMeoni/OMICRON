import os
import torch
import torch.nn as nn 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0" # GPU index

print(os.environ["CUDA_VISIBLE_DEVICES"])
print(torch.cuda.is_available())

print("Available GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)

%load_ext autoreload
%autoreload 2

import torch 
import sys
sys.path.insert(1, os.path.join("..", "data"))
sys.path.insert(1, os.path.join("..", "utils"))
from data_utils import Dataset
from plot_utils import plot_image
from torch.utils.data import DataLoader

# Path to the data folder (update the variable to your path).
path_data=os.path.join("..", "data")
# Seed value
seed=1001

dataset=Dataset(path_data=path_data, seed=seed)
dataset.read_data()

dataset.get_statistics()

batch_size=32

# Train loader
train_loader = DataLoader(dataset.get_split("train"), batch_size=batch_size, pin_memory=False, shuffle=True)
# Cross validation data loader
valid_loader = DataLoader(dataset.get_split("valid"), batch_size=batch_size, pin_memory=False, shuffle=True)
# Test data loader
test_loader = DataLoader(dataset.get_split("test"), batch_size=batch_size, pin_memory=False, shuffle=True)

classes = ('cloud', 'edge', 'good')

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid 
import torch.nn.functional as F 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def resize_tensor_images(images, size=(256, 256)):
    # Resize the batch of images
    return F.interpolate(images, size=size, mode='bilinear', align_corners=False)

def compute_mean_std(loader):
    # Computation of mean and standard deviation of batches
    mean = 0.
    std = 0.
    total_images_count = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count

    return mean, std

def normalize_images(images, mean, std):
    # Normalizing images with previously computed mean and standard deviation
    normalized_images = (images - mean.view(-1, 1, 1)) / std.view(-1, 1, 1)
    return normalized_images

def normalize_individual_image(image):
    # Calculate the mean and std for each channel of the image
    mean = image.mean(dim=[1, 2])
    std = image.std(dim=[1, 2])

    # Ensure std is not zero to avoid division by zero
    std = std.clamp(min=1e-9)

    # Normalize the image
    normalized_image = (image - mean[:, None, None]) / std[:, None, None]
    return normalized_image
    
def tensor_to_numpy(tensor):
    # Rescale the tensor to 0-1 range
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    # Move the tensor to CPU if it's on GPU
    tensor = tensor.cpu()
    # Convert to numpy and transpose from CxHxW to HxWxC for visualization
    numpy_image = tensor.numpy()
    numpy_image = np.transpose(numpy_image, (1, 2, 0))

    return numpy_image

batches_VAL = []
batches_TRL = []
batches_TST = []

UNPRO_batches_VAL = []
UNPRO_batches_TRL = []
UNPRO_batches_TST = []

normalization = 'whole batch normalization'

if normalization == 'none':
    Normalization = 0

    for batch in valid_loader:
        images_VAL, labels_VAL = batch
        resized_images_VAL = resize_tensor_images(images_VAL)
        UNPRO_batches_VAL.append((resized_images_VAL, labels_VAL))
        normalized_images_VAL = resized_images_VAL / 256 
        batches_VAL.append((normalized_images_VAL, labels_VAL))
    
    for batch in train_loader:
        images_TRL, labels_TRL = batch
        resized_images_TRL = resize_tensor_images(images_TRL)
        UNPRO_batches_TRL.append((resized_images_TRL, labels_TRL))
        normalized_images_TRL = resized_images_TRL / 256
        batches_TRL.append((normalized_images_TRL, labels_TRL))


    for batch in test_loader:
        images_TST, labels_TST = batch
        resized_images_TST = resize_tensor_images(images_TST)
        UNPRO_batches_TST.append((resized_images_TST, labels_TST))
        normalized_images_TST = resized_images_TST / 256
        batches_TST.append((normalized_images_TST, labels_TST))

if normalization == 'meannormalization':
    Normalization = 1

    for batch in valid_loader:
        images_VAL, labels_VAL = batch
        resized_images_VAL = resize_tensor_images(images_VAL)
        UNPRO_batches_VAL.append((resized_images_VAL, labels_VAL))
        normalized_images_VAL = torch.stack([normalize_individual_image(img) for img in resized_images_VAL])
        batches_VAL.append((normalized_images_VAL, labels_VAL))

    for batch in train_loader:
        images_TRL, labels_TRL = batch
        resized_images_TRL = resize_tensor_images(images_TRL)
        UNPRO_batches_TRL.append((resized_images_TRL, labels_TRL))
        normalized_images_TRL = torch.stack([normalize_individual_image(img) for img in resized_images_TRL])
        batches_TRL.append((normalized_images_TRL, labels_TRL))

    for batch in test_loader:
        images_TST, labels_TST = batch
        resized_images_TST = resize_tensor_images(images_TST)
        UNPRO_batches_TST.append((resized_images_TST, labels_TST))
        normalized_images_TST = torch.stack([normalize_individual_image(img) for img in resized_images_TST])
        batches_TST.append((normalized_images_TST, labels_TST))

if normalization == 'whole batch normalization':
    Normalization = 2
    
    mean, std = compute_mean_std(test_loader)
    for batch in valid_loader:
        images_VAL, labels_VAL = batch
        resized_images_VAL = resize_tensor_images(images_VAL)
        UNPRO_batches_VAL.append((resized_images_VAL, labels_VAL))
        normalized_alldata_images_VAL = normalize_images(resized_images_VAL, mean, std)

        # Append the normalized images and their corresponding labels to the list
        batches_VAL.append((normalized_alldata_images_VAL, labels_VAL))



    
    for batch in train_loader:
        images_TRL, labels_TRL = batch
        resized_images_TRL = resize_tensor_images(images_TRL)
        UNPRO_batches_TRL.append((resized_images_TRL, labels_TRL))
        normalized_alldata_images_TRL = normalize_images(resized_images_TRL, mean, std)

        # Append the normalized images and their corresponding labels to the list
        batches_TRL.append((normalized_alldata_images_TRL, labels_TRL))
        

   
    for batch in test_loader:
        images_TST, labels_TST = batch
        resized_images_TST = resize_tensor_images(images_TST)
        UNPRO_batches_TST.append((resized_images_TST, labels_TST))
        normalized_alldata_images_TST = normalize_images(resized_images_TST, mean, std)

        # Append the normalized images and their corresponding labels to the list
        batches_TST.append((normalized_alldata_images_TST, labels_TST))

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, padding=2)  # Reduced filters
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3, padding=2) # Reduced filters
        self.bn2 = nn.BatchNorm2d(12)
        self.conv3 = nn.Conv2d(12, 18, kernel_size=3, padding=2) # Reduced filters
        self.bn3 = nn.BatchNorm2d(18)
        self.conv4 = nn.Conv2d(18, 27, kernel_size=5, padding=2) # Reduced filters
        self.bn4 = nn.BatchNorm2d(27)
        self.pool = nn.MaxPool2d(2, 2)

        # Adjusted layer sizes for fully connected layers
        self.fcs = nn.ModuleList([nn.Linear(in_size, out_size) for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:])])

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        for layer in self.fcs:
            if isinstance(layer, nn.Linear):
                x = layer(x)
            elif isinstance(layer, nn.Dropout):
                x = F.relu(x)
                x = layer(x)
        
        return x

# Adjusted flattened size and layer sizes
flattened_size = 256 * (16) * (16)  # Adjust this based on the output size of the last conv layer
layer_sizes = [6912, 400, 150, 20, 3] 
net = Net(layer_sizes)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

n_epochs = 25

train_losses = []
val_losses = []

for epoch in range(n_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    net.train()  # Set the model to training mode
    for i, data in enumerate(batches_TRL, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()  # Add the batch's loss to the running total

    epoch_loss = running_loss / len(batches_TRL)  # Calculate the average loss for this epoch
    train_losses.append(epoch_loss)

    # Validation phase
    running_val_loss = 0.0
    net.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No gradients needed for validation
        for i, data in enumerate(batches_VAL, 0):
            inputs, labels = data

            outputs = net(inputs)
            val_loss = criterion(outputs, labels)

            running_val_loss += val_loss.item()

    epoch_val_loss = running_val_loss / len(batches_VAL)  # Calculate the average validation loss for this epoch
    val_losses.append(epoch_val_loss)

    # Print epoch statistics
    print(f'Epoch {epoch+1} - Training Loss: {epoch_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}')

print('Finished Training')

layer_info = '-'.join(map(str, layer_sizes[:-1]))
txt = './N{}_e{}_n{}_b{}_s{}.pth'.format(Normalization, n_epochs, layer_info, batch_size, seed,)
PATH = txt
# PATH = './test1.pth'
torch.save(net.state_dict(), PATH)

net = Net(layer_sizes)
net.load_state_dict(torch.load(PATH))

dataiter_norm_VAL = iter(batches_VAL)
images_norm_VAL, labels_norm_VAL = next(dataiter_norm_VAL)
dataiter_norm_TRL = iter(batches_TRL)
images_norm_TRL, labels_norm_TRL = next(dataiter_norm_TRL)
dataiter_norm_TST = iter(batches_TST)
images_norm_TST, labels_norm_TST = next(dataiter_norm_TST)


dataiter_og_VAL = iter(UNPRO_batches_VAL)
images_og_VAL, labels_og_VAL = next(dataiter_og_VAL)
dataiter_og_TRL = iter(UNPRO_batches_TRL)
images_og_TRL, labels_og_TRL = next(dataiter_og_TRL)
dataiter_og_TST = iter(UNPRO_batches_TST)
images_og_TST, labels_og_TST = next(dataiter_og_TST)

outputs_VAL = net(images_norm_VAL)
outputs_TRL = net(images_norm_TRL)
outputs_TST = net(images_norm_TST)

_, predicted_VAL = torch.max(outputs_VAL, 1)


# print('Predicted: ', ' '.join(f'{classes[predicted_VAL[j]]:5s}'
                            #   for j in range(batch_size)))


predictions = [classes[predicted_VAL[j]] for j in range(batch_size)]


_, predicted_TRL = torch.max(outputs_TRL, 1)
_, predicted_TST = torch.max(outputs_TST, 1)

predictions_TST = [classes[predicted_TST[j]] for j in range(batch_size)]

# Calculate the number of correctly predicted labels
correct_predictions_VAL = (predicted_VAL == labels_norm_VAL).sum().item()
correct_predictions_TRL = (predicted_TRL == labels_norm_TRL).sum().item()
correct_predictions_TST = (predicted_TST == labels_norm_TST).sum().item()
# Calculate the total number of labels
total_labels_VAL = labels_norm_VAL.size(0)
total_labels_TRL = labels_norm_TRL.size(0)
total_labels_TST = labels_norm_TST.size(0)
# Calculate the accuracy as a percentage
accuracy = 100 * correct_predictions_VAL / total_labels_VAL
accuracy_train = 100 * correct_predictions_TRL / total_labels_TRL
accuracy_test = 100 * correct_predictions_TST / total_labels_TST
print('Validation Accuracy: {:.2f}%'.format(accuracy))
print('Train accuracy: {:.2f}%'.format(accuracy_train))
print('Test accuracy: {:.2f}%'.format(accuracy_test))

fig, axs = plt.subplots(batch_size, 4, figsize=(12, 4 * batch_size))



for i in range(len(images_norm_TST)):
    og_img = tensor_to_numpy(images_og_TST[i])
    norm_img = tensor_to_numpy(images_norm_TST[i])
    
    axs[i, 0].imshow(og_img)
    axs[i, 0].axis('off')

    # Plot the second image
    axs[i, 1].imshow(norm_img)  # Assuming there are always pairs of images
    axs[i, 1].axis('off')

    # Plot the first text box
    axs[i, 2].text(0.5, 0.5, classes[labels_norm_TST[i]], ha='center', va='center', fontsize=12, color='black')
    axs[i, 2].axis('off')

    # Plot the second text box
    axs[i, 3].text(0.5, 0.5, 'prediction: ' + str(predictions_TST[i]), ha='center', va='center', fontsize=12, color='black')
    axs[i, 3].axis('off')
# plt.tight_layout()
    # plt.tight_layout()
    if predictions_TST[i] == classes[labels_norm_TST[i]]:
        row_color = 'lightblue'  # Set to blue if the condition is true
    else:
        row_color = 'yellow'  # Set to yellow if the condition is false

    rect = patches.Rectangle((0, 0), 1, 1, linewidth=0, edgecolor='none', facecolor=row_color, alpha=0.5)
    axs[i, 0].add_patch(rect)
    axs[i, 1].add_patch(patches.Rectangle((0, 0), 1, 1, linewidth=0, edgecolor='none', facecolor=row_color, alpha=0.5))
    axs[i, 2].add_patch(patches.Rectangle((0, 0), 1, 1, linewidth=0, edgecolor='none', facecolor=row_color, alpha=0.5))
    axs[i, 3].add_patch(patches.Rectangle((0, 0), 1, 1, linewidth=0, edgecolor='none', facecolor=row_color, alpha=0.5))

plt.show()