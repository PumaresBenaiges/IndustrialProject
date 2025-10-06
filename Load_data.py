# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 17:33:05 2025

@author: Ann Marie Roob
"""

import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt


# Define folder path
base_folder = "./IDP Group A"

# Define wavelength list (450 nm to 950 nm in 20 nm steps)
wavelengths = list(range(450, 951, 20))

# Define subfolders
sample_folders = [f"d{i}" for i in range(1, 15)]  # d1 to d14
dark_folder = "dark"
white_folder = "white"
reference_folder = "reference image"


# Load white and dark images
def load_cube(folder_path):
    cube = []
    for wl in wavelengths:
        file_path = os.path.join(folder_path, f"Image_Cube_{wl}.tif")
        img = tiff.imread(file_path)
        cube.append(img.astype(np.float32))
        print(img.shape)

    return np.stack(cube, axis=0)  # Shape: (bands, height, width)


# Load dark and white reference cubes
dark_cube = load_cube(os.path.join(base_folder, dark_folder))
# white_cube = load_cube(os.path.join(base_folder, white_folder))
# reference_cube = load_cube(os.path.join(base_folder, reference_folder))

# # Avoid division by zero
# denominator = white_cube - dark_cube
# denominator[denominator == 0] = 1e-6

# # Load and compute reflectance for each sample
# reflectance_cubes = {}

# for folder in sample_folders:
#     sample_cube = load_cube(os.path.join(base_folder, folder))
#     R = (sample_cube - dark_cube) / denominator
#     reflectance_cubes[folder] = R  # Shape: (bands, height, width)

# # Example: Access reflectance cube for d1
# R_d1 = reflectance_cubes["d1"]

# # %%
# import matplotlib.pyplot as plt


# def plot_example_reflectance():
#     # Choose sample and wavelength index
#     sample_name = "d5"
#     wavelength_index = 8  # e.g., corresponds to 10th wavelength in the list
#     wavelength = wavelengths[wavelength_index]  # e.g., 650 nm

#     # Get reflectance cube
#     R = reflectance_cubes[sample_name]

#     # Extract 2D image at selected wavelength
#     reflectance_image = R[wavelength_index, :, :]

#     # Plot the image
#     plt.figure(figsize=(6, 5))
#     plt.imshow(reflectance_image, cmap="grey")
#     plt.colorbar(label="Reflectance")
#     plt.title(f"Reflectance of {sample_name} at {wavelength} nm")
#     plt.axis("off")
#     plt.tight_layout()
#     plt.show()


# def plot_original_tif():
#     # Path to your TIFF image
#     tiff_path = "./IDP Group A/d5/Image_Cube_610"  # replace with actual path

#     # Load the image
#     image = tiff.imread(tiff_path)

#     # Display the image
#     plt.figure(figsize=(6, 5))
#     plt.imshow(image, cmap="gray")  # or 'viridis' or any other colormap
#     plt.colorbar(label="Pixel Value")
#     plt.title("TIFF Image Defect 1 at 610 nm")  # Customize this as needed
#     plt.axis("off")
#     plt.tight_layout()
#     plt.show()


# plot_example_reflectance()
# plot_original_tif()
