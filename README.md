# Deep Learning Based Stereo Depth Estimation

This repository contains an implementation of stereo depth estimation using [PSMNet](https://github.com/JiaRenChang/PSMNet) with a focus on comparing deep learning predictions with ground truth depth maps. The code preprocesses stereo images and corresponding ground truth (from a PFM file), runs inference with PSMNet, and computes quantitative error metrics (MAE, RMSE) for evaluation.

## Overview

Stereo depth estimation aims to reconstruct a dense depth map from a pair of stereo images. This project:
- Loads and preprocesses stereo images (left and right) ensuring they match PSMNet's input size requirements.
- Runs a pretrained PSMNet model to predict a disparity map.
- Loads and preprocesses a ground truth depth/disparity map from a PFM file.
- Compares the predicted output and ground truth using error metrics.
- Visualizes the results with side-by-side comparisons.

## Repository Structure

your-project/ 
├── PSMNet/ # PSMNet model code (submodule folder) 
├── main.py # Main script for inference and evaluation 
├── README.md # This file 
├── LICENSE # Open source license (e.g., MIT) 
└── requirements.txt # Dependencies for the project


## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/your-project.git
   cd your-project

2. **Install dependencies:**
Make sure you have Python (>=3.6) installed, then install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
    The requirements.txt file should include packages such as:

    - torch
    - torchvision
    - numpy
    - opencv-python
    - matplotlib

3. **PSMNet Submodule**

    ``` bash
    git submodule add https://github.com/JiaRenChang/PSMNet.git PSMNet

4. **Pretrained Weights**\
    The pretrained weights file pretrained_model_KITTI2015.tar has been downloaded from PSMNet GitHub repo. You can download updated and make suitable changes in code if needed.\
    [Google Drive](https://drive.google.com/file/d/1pHWjmhKMG4ffCrpcsp_MTXMJXhgl3kF9/view)

## Usage
```bash
python main.py
```
The script will:

- Preprocess stereo images (im0.png and im1.png).
- Run inference with PSMNet to predict a disparity map.
- Load and preprocess a ground truth depth map from disp0.pfm.
- Compute and print quantitative metrics (MAE, RMSE).
- Save and display the comparison plots.

### Customisation

Replace im0.png, im1.png and disp0.png with your own stereo images and ground truth files\
Resizing Settings:
The stereo images are resized to the nearest multiples of 16 to meet PSMNet requirements, while the ground truth is resized accordingly to maintain the original aspect ratio for fair comparison.\

Disparity to Depth\
Note that PSMNet outputs a disparity map. To compute actual depth values, convert disparity to depth using:\
```mathematica
Depth = (Focal Length × Baseline) / Disparity
```

## Contributing
Contributions are welcome! If you find issues or have suggestions, please open an issue or submit a pull request. When contributing:
- Follow the existing coding style
- Update comments and documentation as necessary

## License
This project is licensed under the MIT License -- See the LICENSE file for details.

## Acknowledgements
- PSMNet : The deep learning model used for stereo depth estimation
- ChatGPT has been used to help generate parts of this code and documentation
