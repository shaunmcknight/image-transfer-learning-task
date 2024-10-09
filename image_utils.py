import os
import numpy as np
import matplotlib.pyplot as plt

"""
Helper functions to deal with text files, image processing and visualization.

"""

def load_image_list(txt_file):
    with open(txt_file, 'r') as file:
        img_list = [line.strip() for line in file.readlines()]  # Strip newline characters
    return img_list

def get_image_paths(dataset, data_dir):
    # For each image in the dataset, store its relative path (folder/filename.jpg)
    return [os.path.relpath(path, data_dir) for path, _ in dataset.samples]

def preprocess_for_display(image, title=None):
    # Unnormalize the image
    image = image.numpy().transpose((1, 2, 0))  # Convert from Tensor (C, H, W) to NumPy array (H, W, C)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean  # Reverse normalization
    image = np.clip(image, 0, 1)  # Clip pixel values to [0, 1] for valid image display
    return image

# Function to display sample iamge from each class
def show_examples(dataset, class_names):
    class_samples = {class_name: None for class_name in class_names}

    # Loop through the dataset and get the first example for each class
    for i in range(len(dataset)):
        image, label = dataset[i]
        class_name = class_names[label]
        if class_samples[class_name] is None:
            class_samples[class_name] = image
        if None not in class_samples.values():
            break

    # Plot each image with its corresponding class name
    fig, axes = plt.subplots(1, len(class_samples), figsize=(15, 5))
    for ax, (class_name, image) in zip(axes, class_samples.items()):
        ax.axis('off')
        ax.imshow(preprocess_for_display(image))
        ax.set_title(class_name)
    plt.show()
