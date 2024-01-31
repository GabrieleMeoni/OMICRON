from torchvision.io import read_image
import os
from glob import glob
import plot_utils
import matplotlib.pyplot as plt


def main(unfiltered_folder):
    """Main function to run the script. It will loop through all the images in the folder and ask for a label. The label
    can be: "c" for cloud, "e" for edge, "g" for good, or empty to skip the image. The script will move the image to the
    corresponding folder. After looping through all the images, it will ask if you want to loop again.
    Note: the script will not check if the image is already classified, so be careful to not label the same image
    multiple times.
    :param unfiltered_folder: name of the folder containing the unfiltered images"""
    data_folder = os.path.join("..", "data")
    png_folder = os.path.join(data_folder, unfiltered_folder)
    cloud_folder = os.path.join(data_folder, "Cloud")
    good_folder = os.path.join(data_folder, "Good")
    edge_folder = os.path.join(data_folder, "Edge")
    loop = True
    while loop:
        images_path = sorted(glob(os.path.join(png_folder, "*")))
        total_images = len(images_path)
        for i, image_path in enumerate(images_path):
            image = read_image(image_path)
            fig = plot_utils.plot_image(image, equalize=True, downsampling_factor=(2, 2), figsize=(20, 8),
                                        title=f"Image {i+1}/{total_images} - {os.path.basename(image_path)}")
            plt.show()
            label = input("Label [c/e/g] Enter for skip: ")
            label = label.lower().strip()
            if label == "c":
                os.rename(image_path, os.path.join(cloud_folder, os.path.basename(image_path)))
                print(f"Moved image {os.path.basename(image_path)} to Cloud folder.")
            elif label == "e":
                os.rename(image_path, os.path.join(edge_folder, os.path.basename(image_path)))
                print(f"Moved image {os.path.basename(image_path)} to Edge folder.")
            elif label == "g":
                os.rename(image_path, os.path.join(good_folder, os.path.basename(image_path)))
                print(f"Moved image {os.path.basename(image_path)} to Good folder.")
            else:
                print("Skipping image.")
            plt.clf()
        print("\033[96m===========Finished with the set===========")
        loop = input("Continue to loop through again? [Y/N] \033[0m").lower().strip() != "n"


if __name__ == "__main__":
    main(unfiltered_folder="Siddarth")
