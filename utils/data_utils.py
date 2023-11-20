import os
import pandas as pd
import torch
from glob import glob
from sklearn.model_selection import train_test_split
from torchvision.io import read_image
from torchvision.transforms import Resize
from tqdm import tqdm

class SplitDataset(torch.utils.data.Dataset):
    """Implements a simple dataset modelling one of the three splits "train", "valid", "test", according to torch dataset model.
    """
    def __init__(self, X, y):
        """Initialization.

        Args:
            X (torch.tensor): images.
            y (torch.tensor): targets.
        """
        self.X, self.y = X,y

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.y)

    def __getitem__(self, index):
        """Generates one sample of data.

        Args:
            index (int): index of the element.

        Returns:
            torch.tensor, torch.tensor: X and Y elements at requested index.
        """
        # Select sample
        X = self.X[index]

        # Load data and get label
        y = self.y[index]

        return X, y.to(torch.int64)

class Dataset():
    """Creates Dataset which reads images from target directories.
    """
    def __init__(self, path_data="../data/", edge_px_to_discard=[10, 10], target_size=[256, 256], train_percentage=0.7, valid_percentage=0.15, seed=42, device=torch.device("cpu")):
        """Initialization.

        Args:
            path_data (str, optional): path to directory containing all the other subdirectories (e.g., Cloud, Edge, Good). Defaults to "../data/".
            edge_px_to_discard (list, optional): number of pixels to discard in each image from top, bottom and left and right. Defaults to [10, 10].
            target_size (list, optional): target size of each image. Each image will be resized to this target. Defaults to [256, 256].
            train_percentage (int, optional): percentage of images of train split.  Defaults to 0.7 (70%).
            valid_percentage (int, optional): percentage of image of cross_validation split.  Defaults to 0.15 (15%).
            seed (int, optional): seed to perform pseudo-randomic train/cross validation/test split. Defaults to 42.
            device (torch.device, optional): device. Defaults to torch.device("cpu").
        """
        # Train images placeholder.
        self.X_train= None
        # Train targets placeholder.
        self.y_train= None
        # Cross validation images placeholder.
        self.X_valid= None
        # Cross validation targets placeholder.
        self.y_valid= None
        # Test images placeholder.
        self.X_test= None
        # Test targets placeholder.
        self.y_test= None
        # Path to the main directory.
        self.path_data=path_data
        # Number of edge pixels to be discarded.
        self.edge_px_to_discard=edge_px_to_discard
        # Target image size. Images will be reshaped.
        self.target_size=target_size
        # Creaating a resize function by using the corresponding torchvision transform.
        self.resize_transform=Resize(target_size, antialias=True)
        # Train percentage.
        self.train_percentage=train_percentage
        # Cross validation percentage
        self.valid_percentage=valid_percentage
        # Seed for pseudorandomic split.
        self.seed=seed
        # Target device. Use torch.device("cuda") if you want to use GPUs.
        self.device=device
        # Dataset ready (i.e., images were loaded).
        self.dataset_ready=False

    def __clean_image__(self, image):
        """Cleans each image by: (first step)discarding top, bottom, left, right pixel. (second step) Resizing to the target size.

        Args:
            image (torch.tensor): input image.

        Returns:
            torch.tensor: cleaned image.
        """
        image=image[:, self.edge_px_to_discard[0]:-self.edge_px_to_discard[0], self.edge_px_to_discard[1]:-self.edge_px_to_discard[1]]
        return self.resize_transform(image)

    def __read_images_from_directory__(self, directory):
        """Reads images from a target directory and moves them to the target device.

        Args:
            directory (str): target directory. Must be either "Cloud", "Edge", "Good".

        Raises:
            ValueError: target directory not existing.

        Returns:
            torch.tensor: tensor containing target images read from directory. Tensor shape [N images, 3, target_size[0], target_size[1]].
        """
        # Testing if directory is existing.
        if not(os.path.exists(os.path.join(self.path_data, directory))):
            raise ValueError(f"{directory} directory missing.")
        else:
            # Parsing images and retrieving names.
            directory_images_path=glob(os.path.join(self.path_data, directory,"*"))
            # Create a placeholder for the target images.
            images=torch.zeros([len(directory_images_path), 3, self.target_size[0], self.target_size[1]], device=self.device)
            # Reading images.
            for n,path in tqdm(enumerate(directory_images_path), desc=f"Parsing class: {directory}"):
                image=read_image(path)
                # Removing unwanted side pixels and reshaping image.
                images[n]=self.__clean_image__(image)
        return images

    def read_data(self):
        """Reads target images from the "Cloud", "Edge", "Good" directory and creates the corresponding targets in one-hot encoding.
        """
        # Read cloud, edge, and good images.
        cloud_images=self.__read_images_from_directory__("Cloud")
        edge_images=self.__read_images_from_directory__("Edge")
        good_images=self.__read_images_from_directory__("Good")
        # Create a placeholder to stack all the images.
        X=torch.zeros([len(cloud_images) + len(good_images) + len(edge_images), cloud_images[0].shape[0], self.target_size[0], self.target_size[1]]).to(self.device)
        # Place all the images into the placeholder.
        X[:len(cloud_images)]=cloud_images
        X[len(cloud_images): len(cloud_images)+len(edge_images)]=edge_images
        X[len(cloud_images)+len(edge_images):]=good_images
        # Create a placeholder for target.
        y=torch.zeros([X.shape[0]]).to(self.device)
        # Assign 0,1,2 respectively to Cloud, Edge, and Good images.
        y[len(cloud_images):len(cloud_images)+len(edge_images)]=1
        y[len(cloud_images)+len(edge_images):]=2
        # Pseudorandomic split of train and test.
        train_idx, test_idx =train_test_split(list(range(len(X))), test_size=(1 - (self.train_percentage + self.valid_percentage)), random_state=self.seed)
        # Splitting Train+valid
        X_train_valid, y_train_valid = X[train_idx], y[train_idx]
        # Splitting test and moving class placeholders.
        self.X_test, self.y_test = X[test_idx], y[test_idx]
        # Splitting into train and cross validation
        train_idx, valid_idx = train_test_split(list(range(len(X_train_valid))), test_size=self.valid_percentage * len(X)/ len(X_train_valid), random_state=self.seed)
        # Splitting train and moving class placeholders.
        self.X_train, self.y_train = X_train_valid[train_idx], y_train_valid[train_idx]
        # Splitting cross validation and moving class placeholders.
        self.X_valid, self.y_valid = X_train_valid[valid_idx], y_train_valid[valid_idx]
        # Setting self.dataset_ready to True
        self.dataset_ready=True

    def get_split(self, split="train"):
        """Returns one of the three splits between "train", "valid", and "test".

        Args:
            split (str, optional): requested split. Defaults to "train".

        Raises:
            ValueError: Unsopported split.

        Returns:
            SplitDataset: dataset modelling one split according to the torch Dataset standard.
        """
        if split == "train":
            return SplitDataset(X=self.X_train, y=self.y_train)
        elif split == "valid":
            return SplitDataset(X=self.X_valid, y=self.y_valid)
        elif split == "test":
            return SplitDataset(X=self.X_test, y=self.y_test)
        else:
            raise ValueError(f"Split: {split} not supported. Please, enter one of the following: ""train"",""valid"",""test"".")

    def get_statistics(self):
        """Returning statistics about classes distribution in the different splits through a dataset.

        Raises:
            ValueError: "Impossible to get statistics from an empty dataset."

        Returns:
            pandas dataframe: test stats.
        """

        if self.dataset_ready:
            # Creating tmp variables to simplify readibility
            y_train_tmp = self.y_train
            y_valid_tmp = self.y_valid
            y_test_tmp = self.y_test

            # Creating empty dictionary
            stats_dict = {"train" : {"cloud" : 0, "edge" : 0, "good" : 0}, "valid" : {"cloud" : 0, "edge" : 0, "good" : 0}, "test" : {"cloud" : 0, "edge" : 0, "good" : 0}}

            # Calculating stats
            stats_dict["train"]["cloud"], stats_dict["train"]["edge"], stats_dict["train"]["good"] = len(y_train_tmp[y_train_tmp == 0]), len(y_train_tmp[y_train_tmp == 1]), len(y_train_tmp[y_train_tmp == 2])
            stats_dict["valid"]["cloud"], stats_dict["valid"]["edge"], stats_dict["valid"]["good"] = len(y_valid_tmp[y_valid_tmp == 0]), len(y_valid_tmp[y_valid_tmp == 1]), len(y_valid_tmp[y_valid_tmp == 2])
            stats_dict["test"]["cloud"], stats_dict["test"]["edge"], stats_dict["test"]["good"] = len(y_test_tmp[y_test_tmp == 0]), len(y_test_tmp[y_test_tmp == 1]), len(y_test_tmp[y_test_tmp == 2])
            return pd.DataFrame.from_dict(stats_dict)

        else:
            raise ValueError("Impossible to get statistics from an empty dataset.")
