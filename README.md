# IndustrialProject

This project focuses on classifying printed lenses as defective or non-defective based on two CTQ (Critical to Quality) measurements. 
The printed lenses are captured using a hyperspectral camera, and this codebase processes the resulting images to perform automated quality classification.
There is a set of train samples and a set of test samples captured in 6 differents trials.

## Project Overview
- Capture hyperspectral images of printed lenses
- Extract two CTQ features from each sample
- Classification with a threshold to determine whether a lens is defective
- Provide scripts and utilities to streamline the full workflow (data loading, preprocessing, classification)

## File structure
- **rename_data.py**: Used to rename some fodlers.
- **functions.py**: Contains utility functions for image processing, CTQs extraction, and other core operations used throughout the project.
- **class_sample_testing.py**: Implements a class-based structure for handling sample testing, including loading samples, running classification, and showing and saving results.
- **CSL_2025_Python_codes.py**: Functions provided during Color Science Laboratory course, adapted and used for spectral conversion to XYZ, LAB and RGB.
- **main.ipynb** and **main.py**: Both files execute the main workflow of the project.
  - **main.ipynb**: Interactive notebook version for exploration and visualization
  - **main.py**: Script version for automated or command-line execution
- **train_results.xlsx**: Generated when running main. Contains the CTQ and classification results for the train samples.
- **train_results.xlsx**: Results for the test samples in each trial.
  
## Folder structure 
- `main_folder/`
  - `train/`
      - `dark/`
      - `white/`
      - `reference/`
      - One folder per sample
  - `test1/`
     - `operator1/` (same structure as `train`)
    - `operator2/` (same structure as `train`)
    - `operator3/`(same structure as `train`)
  - `test2/` (same structure as `test1`)
- functions.py
- class_sample_testing.py
- main.ipynb
- main.py
 
## How to run the code
Choose wether you want to run in notebook or python script.
- In the notebook version (main.ipynb): just run the notebook. Change manually the folder where you have data stored if needed.
- In the script version (main.py):
    - Run: `python3 main.py "../IDP Group A"`
    - Args:
      - **Folder with spectral data**: string, optional, by default: "../IDP Group A"




