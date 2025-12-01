# IndustrialProject

This project focuses on developing a measurement system to classifying printed lenses as defective or non-defective based on two CTQ (Critical to Quality) measurements. 
The printed lenses are captured using a hyperspectral camera, and this codebase processes the resulting images to perform automated quality classification.
There is a set of train samples and a set of test samples captured in 6 differents trials.

## Project Overview
### Exeperimental workflow
- Phase 1:
  - Extract CTQs from train data
  - Define thresholds
- Phase 2:
  - Measure test data on different trials
  - Classify test data
  - Perform GAGE R&R analysis
 
### Proposed solution steps
- (1) Capture hyperspectral images of printed lenses
- (2) Scripts for automated lense classification
    - Extract two CTQ features from each sample
    - Classification with a threshold to determine whether a lens is defective

## File structure
- **rename_data.py**: Used to rename some folders.
- **functions.py**: Contains utility functions for image processing, CTQs extraction, and other core operations used throughout the project.
- **class_sample_testing.py**: Implements a class-based structure for handling sample testing, including loading samples, running classification, and showing and saving results.
- **CSL_2025_Python_codes.py**: Functions provided during Color Science Laboratory course, adapted and used for spectral conversion to XYZ, LAB and RGB.
- **main.ipynb** and **main.py**: Both files execute the main workflow of the project.
  - **main.ipynb**: Interactive notebook version for exploration and visualization
  - **main.py**: Script version for automated or command-line execution
- **train_results.xlsx**: Generated when running main. Contains the CTQ and classification results for the train samples.
- **train_results.xlsx**: Generated when running main. Conatins results for the test samples in each trial.
  
## Folder structure required
- `IDP Group A/`
  - `train/`
      - `dark/`
      - `white/`
      - `reference/`
      - One folder per sample, the name of the folder will be used as the sample id.
  - `test1/`
     - `operator1/` (same structure as `train`)
    - `operator2/` (same structure as `train`)
    - `operator3/`(same structure as `train`)
  - `test2/` (same structure as `test1`)
- functions.py
- CSL_2025_Python_codes.py
- class_sample_testing.py
- main.ipynb
- main.py
 
## How to run the code
Choose wether you want to run in notebook or python script.
- In the notebook version (main.ipynb): just run the notebook. Change manually the folder where you have data stored if needed.
- In the script version (main.py):
    - Run: `python3 main.py path_to_folder`
    - Args:
      - **path_to_folder** (string, optional): Path of the folder where you have the spectral data, by default: "./IDP Group A"




