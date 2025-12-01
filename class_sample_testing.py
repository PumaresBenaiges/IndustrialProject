import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import cv2
import math
import pandas as pd
from CSL_2025_Python_codes import spim2XYZ, XYZ2Lab, spim2rgb, XYZ2RGB
from functions import *

class sample_testing:
    """
    This class is used to classify a set of samples into defective and non-defective.
    Inputs:
        - Folder_name (string): folder that conatins the spectral data of the samples. 
            It needs to have the subfolders white, dark, reference and then a folder for each sample.
        - Vis (boolean, optional): if it is True it will show plots of intermediate steps 
                                   if its False no plots will be displayed.
        - Trial (int, optional): Id of the trail, in case the same samples have been captured in different trials.
        - Operator (int, optionaln): Id of the operator, in case the same samples have been captured by different operators.
    """
    def __init__(self, folder_name, vis=False, trial=0, operator=0):
        self.vis = vis
        self.trial = trial
        self.operator = operator
        self.base_folder = folder_name
        sample_folders = [
            folder.name
            for folder in os.scandir(self.base_folder)
            if folder.is_dir() and folder.name not in ["dark", "white", "reference", "reference image"]
        ]
        self.sample_folders = sorted(sample_folders, key=lambda x: int(''.join(filter(str.isdigit, x))))

        self.wavelengths = list(
            range(450, 951, 20)
        )  # Bands set in the lab when capturing images
        self.band = 530  # Band for image alignment
        print("Trial: ", trial, " Operator: ", operator)

    def load_data(self):
        """
        Loads dark, white and reference image. Applies white correction to reference.
        """
        # Load dark, white and reference cubes
        dark_cube = load_cube(os.path.join(self.base_folder, "dark"))
        white_cube = load_cube(os.path.join(self.base_folder, "white"))
        self.reference_cube = load_cube(os.path.join(self.base_folder, "reference"))
        try: 
            self.reference_cube = load_cube(os.path.join(self.base_folder, "reference"))
        except Exception:
            self.reference_cube = load_cube(os.path.join(self.base_folder, "reference image"))

        # Load samples
        denominator = white_cube - dark_cube
        denominator[denominator == 0] = 1e-6  # Avoid division by zero
        self.reference_cube = (self.reference_cube - dark_cube) / denominator
        self.sample_cubes = {}
        for folder in self.sample_folders:
            sample_cube = load_cube(os.path.join(self.base_folder, folder))
            self.sample_cubes[folder] = (
                sample_cube - dark_cube
            ) / denominator  # Shape: (bands, height, width)
        self.bands, self.h, self.w = self.reference_cube.shape

    def process_reference(self):
        """
        For the reference image:
        - Interpolates spectral data 
        - Converts to LAB and RGB, 
        - Finds radius and center of IC
        - Creates mask of IC
        """
        print("Processing reference...")
        R_sample = self.reference_cube

        # Single band 
        ref_single_band = R_sample[self.wavelengths.index(self.band), :, :]
        self.ref_binary = gray_to_binary_image(ref_single_band)

        # Interpolation
        R_sample_interp, new_wavelengths = interpolate_spectral_cube(
            R_sample[:, : self.h // 2, : self.w // 2], self.wavelengths, wl_min=360, wl_max=830, wl_step=10
        )
        R_sample_interp = np.transpose(R_sample_interp, (1, 2, 0))  # (H, W, bands)

        # Transform to LAB and RGB
        ref_XYZ = spim2XYZ(R_sample_interp, new_wavelengths, "D65")
        self.ref_LAB = XYZ2Lab(ref_XYZ, new_wavelengths, "D65")
        self.ref_rgb = XYZ2RGB(ref_XYZ)

        # Detect center and radius of IC
        ref_initial_center = detect_circle_center(self.ref_rgb, self.vis)
        ref_initial_mask = compute_IC_mask(
            self.ref_rgb, vis=self.vis, pixel_IC=ref_initial_center
        )
        ref_circle_info = get_circle_info(ref_initial_mask, self.ref_rgb, self.vis)
        self.ref_center = (
            int(np.round(ref_circle_info["centroid"][0])),
            int(np.round(ref_circle_info["centroid"][1])),
        )
        self.ref_radius = (
            int(np.round(ref_circle_info["radius"])) - 5
        )

        # Create mask of IC region
        self.ref_mask = np.zeros(self.ref_LAB.shape[:2], dtype=np.uint8)
        cv2.circle(self.ref_mask, self.ref_center, self.ref_radius, 1, -1)

        # Only for visualization of results
        self.IC_visualization1 = [
            crop_circle(self.ref_rgb, self.ref_center, self.ref_radius)
        ]
        output = self.ref_rgb.copy()
        cv2.circle(output, self.ref_center, self.ref_radius, (0, 255, 0), 2)
        cv2.circle(output, self.ref_center, 2, (0, 0, 255), -1)
        self.IC_visualization2 = [output]
  

    def process_samples(self):
        """
        For the samples:
        - Interpolates
        - Converts to RGB
        """
        self.samples_rgb = {}
        for sample in self.sample_folders:
            sample_cube = self.sample_cubes[sample]
            cube_interp, self.new_wavelengths = interpolate_spectral_cube(
                sample_cube, self.wavelengths, wl_min=360, wl_max=830, wl_step=10
            )
            cube_interp = np.transpose(cube_interp, (1, 2, 0))  # (H, W, b)
            sample_XYZ = spim2XYZ(cube_interp, self.new_wavelengths, "D65")
            self.samples_rgb[sample] = XYZ2RGB(sample_XYZ)

    def align_samples_to_reference(self):
        """
        Find the transformation matrix for each sample to align it with reference.
        """
        self.homographies = {}
        for sample in self.sample_folders:
            sample_cube = self.sample_cubes[sample]
            sample_single_band = sample_cube[self.wavelengths.index(self.band), :, :]
            sample_binary = gray_to_binary_image(sample_single_band)
            H = align_and_visualise_homography(
                self.ref_binary, sample_binary, sample, visualise=self.vis
            )
            self.homographies[sample] = H

            if self.vis:
                aligned, _ = align_and_blend_RGB_homography(
                    self.ref_rgb, self.samples_rgb[sample], H, sample
                )
                overlay_mask(
                    self.samples_rgb[sample],
                    self.ref_mask,
                    sample,
                    color=(1, 0, 0),
                    alpha=0.5,
                )
                overlay_mask(
                    aligned,
                    self.ref_mask,
                    f"aligned_{sample}",
                    color=(1, 0, 0),
                    alpha=0.5,
                )

    def compute_CTQs(self):
        """
        For each sample:
        - Aligns sample to reference.
        - Finds radius of IC (CTQ2).
        - Converts spectra to LAB.
        - Computes CIEDE2000 difference (CTQ1).
        """
        self.deltaE_results = []  # CTQ1
        self.samples_radius = []  # CTQ2

        for sample in self.sample_folders:
            print(f"Processing {sample}...")
            # CTQ2 (in RGB)
            top_left = self.sample_cubes[sample][:, : self.h // 2, : self.w // 2]

            # Align RGB sample to reference
            sample_rgb = extract_RGB(top_left, self.wavelengths)
            h, w, _ = sample_rgb.shape
            A_3x3 = self.homographies[sample]
            A_2x3 = A_3x3[:2, :]
            aligned = cv2.warpAffine(sample_rgb, A_2x3, (w, h))

            # Compute defect mask of IC to find circle radius
            sample_initial_mask = compute_IC_mask(
                sample_rgb, A_2x3, self.vis, self.ref_center
            )
            sample_circle_info = get_circle_info(sample_initial_mask, aligned, self.vis)
            self.samples_radius.append(
                sample_circle_info["radius"]
            )

            # Masked IC region for visualization
            aligned_vis = cv2.warpAffine(self.samples_rgb[sample][:, : self.h // 2, : self.w // 2], A_2x3, (w, h))
            self.IC_visualization1.append(
                crop_circle(aligned_vis, self.ref_center, self.ref_radius)
            )

            # Circle Visualization
            center = (
                int(sample_circle_info["centroid"][0]),
                int(sample_circle_info["centroid"][1]),
            )
            radi = int(sample_circle_info["radius"])
            output = aligned_vis.copy()
            cv2.circle(output, center, radi, (0, 255, 0), 2)
            cv2.circle(output, center, 2, (0, 0, 255), -1)
            self.IC_visualization2.append(output)

            # CTQ1 (all spectra)
            cube = self.sample_cubes[sample][:, : self.h // 2, : self.w // 2]
            bands, h, w = cube.shape

            # Align full cube, interpolate and convert to LAB
            cube_aligned = np.stack(
                [cv2.warpAffine(cube[j], A_3x3[:2, :], (w, h)) for j in range(bands)]
            )
            cube_interp, new_wavelengths = interpolate_spectral_cube(
                cube_aligned, self.wavelengths, wl_min=360, wl_max=830, wl_step=10
            )
            cube_interp = np.transpose(cube_interp, (1, 2, 0))  # (H, W, b)
            sample_XYZ = spim2XYZ(cube_interp, new_wavelengths, "D65")
            sample_LAB = XYZ2Lab(sample_XYZ, new_wavelengths, "D65")

            # Cielab delta E with reference
            DE_mean = calculate_delta_E(self.ref_LAB, sample_LAB, mask=self.ref_mask)
            self.deltaE_results.append(DE_mean)

    def save_results(self):
        """
        Saves results in a excel file with columns:
        - ID: Id of sample
        - trial: number of trial
        - operator: number of operator
        - DE_mean: Delta Em mean value of IC region
        - D_radius: difference of radius of sample with reference radius
        - CTQ1: True if DE_mean is over threshold, False otherwise
        - CTQ2: True if D_radius is over threshold, False otherwise
        - Defect?: True if CTQ1 or CTQ2 is True or if both are True
        """
        # Save results to df
        result_df = pd.DataFrame()
        result_df["ID"] = self.sample_folders
        result_df["trial"] = self.trial
        result_df["operator"] = self.operator
        result_df["DE_mean"] = self.deltaE_results
        result_df["D_radius"] = abs(
            np.array(self.samples_radius) - (self.ref_radius + 5)
        )
        result_df["CTQ1"] = np.array(self.deltaE_results) > 2.8
        result_df["CTQ2"] = result_df["D_radius"] > 1.2
        result_df["Defect?"] = result_df["CTQ1"] | result_df["CTQ2"]
    
        return result_df

    def test_sample(self):
        """
        Calls all steps to classify a group of samples.
        """
        self.load_data()
        self.process_reference()
        self.process_samples()
        self.align_samples_to_reference()
        self.compute_CTQs()
        res = self.save_results()
        # self.plot_IC_regions()
        return res

    def plot_IC_regions(self):
        """
        Visualization of detected IC regions of each sample.
        """
        titles = ["reference"] + self.sample_folders
        plt.figure(figsize=(18, 12))
        cols = 5
        rows = math.ceil((len(self.sample_folders) + 1) / cols)
        for idx, (img, title) in enumerate(zip(self.IC_visualization1, titles)):
            plt.subplot(rows, cols, idx + 1)
            plt.imshow(img)
            plt.title(title, fontsize=22, fontweight="bold")
            plt.axis("off")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(18, 12))
        rows = math.ceil((len(self.sample_folders) + 1) / cols)
        for idx, (img, title) in enumerate(zip(self.IC_visualization2, titles)):
            plt.subplot(rows, cols, idx + 1)
            cx, cy = self.ref_center
            img2 = img[cy-100:cy+100,cx-100:cx+100,:]
            img2 = np.clip(img2, 0,1)
            plt.imshow(img2)
            plt.title(title, fontsize=18, fontweight="bold")
            plt.axis("off")
        plt.tight_layout()
        plt.show()        

    def plot_average_spectras(self):
        """
        Visualization of mean spectra of IC region.
        """
        # Reference spectras
        R_sample_interp, new_wavelengths = interpolate_spectral_cube(
            self.reference_cube[:, : self.h // 2, : self.w // 2], self.wavelengths, wl_min=360, wl_max=830, wl_step=10
        )
        inter_spectra = average_reflectance_in_circle(
            R_sample_interp, self.ref_center, self.ref_radius
        )
        interpolated_spectra = [inter_spectra]
        original_spectra = [average_reflectance_in_circle(
            self.reference_cube, self.ref_center, self.ref_radius
        )]
        # Samples spectras
        for sample in self.sample_folders:
            cube = self.sample_cubes[sample][:, : self.h // 2, : self.w // 2]
            bands, h, w = cube.shape
            cube_aligned = np.stack(
                [cv2.warpAffine(cube[j], self.homographies[sample][:2, :], (w, h)) for j in range(bands)]
            )
            cube_interp, new_wavelengths = interpolate_spectral_cube(
                cube_aligned, self.wavelengths, wl_min=360, wl_max=830, wl_step=10
            )
            avg_cube = average_reflectance_in_circle(cube_interp, self.ref_center, self.ref_radius)
            avg_cube2 = average_reflectance_in_circle(cube_aligned, self.ref_center, self.ref_radius)
            interpolated_spectra.append(avg_cube)
            original_spectra.append(avg_cube2)
        
        titles = ["reference"] + self.sample_folders 
        plt.figure(figsize=(10, 6))
        for name, spectrum in zip(titles, original_spectra):
            plt.plot(self.wavelengths, spectrum, label=name)

        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Average Reflectance")
        plt.title("Average Spectral Reflectance of IC region")
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 6))
        for name, spectrum in zip(titles, interpolated_spectra):
            plt.plot(self.new_wavelengths, spectrum, label=name)

        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Average Reflectance")
        plt.title("Average Spectral Reflectance of IC region")
        plt.legend()
        plt.show()