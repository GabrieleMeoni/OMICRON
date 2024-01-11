import torch.nn as nn

# color image -> grayscale image


###### ATEMPT 1 ######
# class OMICRONClassifier(nn.Module):
#     def __init__(self):
#         super(OMICRONClassifier, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(32 * 64 * 64, 256)
#         self.fc2 = nn.Linear(256, 3)  # 3 classes: Cloud, Edge, Good

#     def forward(self, x):
#         x = self.pool(nn.functional.relu(self.conv1(x)))
#         x = self.pool(nn.functional.relu(self.conv2(x)))
#         x = x.view(-1, 32 * 64 * 64)
#         x = nn.functional.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x



### ATEMPT 2 ###

# common base for both classifiers
class Classifier_base(nn.Module):
    def __init__(self):
        super(Classifier_base, self).__init__() 
                # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
       x = self.pool1(self.relu1(self.conv1(x)))
       x = self.pool2(self.relu2(self.conv2(x)))        
       return x

# branch for edge classifier
class Classifier_edge(nn.Module):
    def __init__(self):
        super(Classifier_edge, self,).__init__() 
    
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 32 * 32, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, 1)  # Output size 1 for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu4(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# branch for cloud classifier
class Classifier_cloud(nn.Module):
    def __init__(self):
        super(Classifier_cloud, self,).__init__() 

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

         # Fully connected layers
        self.fc1 = nn.Linear(256 * 32 * 32, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, 1)  # Output size 1 for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu4(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x     

class Classifier_good(nn.Module):
    def __init__(self):
        super(Classifier_good, self,).__init__() 

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

         # Fully connected layers
        self.fc1 = nn.Linear(256 * 32 * 32, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, 1)  # Output size 1 for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu4(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x      

class CompositeModel(nn.Module):
    def __init__(self):
        super(CompositeModel, self).__init__()
        self.base = Classifier_base()
        self.edge = Classifier_edge()
        self.cloud = Classifier_cloud()
        self.good = Classifier_good()

    def forward(self, x):
        x = self.base(x)
        output_edge = self.edge(x)
        output_cloud = self.cloud(x)
        ouput_good = self.good(x)
        return output_edge, output_cloud,ouput_good

