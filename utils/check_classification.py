import os


def main():
    data_folder = os.path.join("..", "data")
    cloud_folder = os.path.join(data_folder, "Cloud")
    good_folder = os.path.join(data_folder, "Good")
    edge_folder = os.path.join(data_folder, "Edge")
    good_set = [i[:-11] for i in os.listdir(good_folder)]
    cloud_set = [i[:-11] for i in os.listdir(cloud_folder)]
    edge_set = [i[:-11] for i in os.listdir(edge_folder)]
    all_images = set(i[:-11] for i in os.listdir(cloud_folder) + os.listdir(good_folder) + os.listdir(edge_folder))

    print(all_images)
    # check if image is in two datasets
    i = 0
    for image in all_images:
        if image in good_set and image in cloud_set:
            print(f"Image {image} is in both good and cloud")
            i += 1
        if image in good_set and image in edge_set:
            print(f"Image {image} is in both good and edge")
            i += 1
        if image in cloud_set and image in edge_set:
            print(f"Image {image} is in both cloud and edge")
            i += 1

    print(f"Total number of images in two datasets: {i}")


if __name__ == "__main__":
    main()
