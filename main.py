
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nnc
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Inputes
LEFT_IMAGE = "im0.png"
RIGHT_IMAGE = "im1.png"
DISPARITY_GT = "disp0.pfm"

# ------------------------------
# Add PSMNet to Python path
# ------------------------------
PSMNET_PATH = os.path.join(os.path.dirname(__file__), "PSMNet")
sys.path.append(PSMNET_PATH)  # Allow importing from PSMNet folder

from models import stackhourglass  # Import PSMNet model architecture

# ------------------------------
# Load PSMNet Model
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = stackhourglass(192)  # Initialize PSMNet with maximum disparity of 192
model = nn.DataParallel(model)
model.to(device)

# Load pretrained weights (using map_location for CPU compatibility)
model_path = "pretrained_model_KITTI2015.tar"
model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
model.eval()  # Set to evaluation mode

# ------------------------------
# Preprocessing for Stereo Images (for inference)
# ------------------------------
def preprocess_stereo(imgL, imgR):
    """
    Preprocesses stereo images to be fed into the deep learning model.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    imgL = transform(imgL).unsqueeze(0).to(device)
    imgR = transform(imgR).unsqueeze(0).to(device)
    return imgL, imgR

# ------------------------------
# Load and preprocess stereo images
# ------------------------------
imgL = cv2.imread(LEFT_IMAGE)  # Left image
imgR = cv2.imread(RIGHT_IMAGE)  # Right image
imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)
# Resize stereo images to be multiples of 16 (PSMNet requirement)
H, W, _ = imgL.shape
target_H = (H // 16) * 16
target_W = (W // 16) * 16
imgL = cv2.resize(imgL, (target_W, target_H))
imgR = cv2.resize(imgR, (target_W, target_H))
imgL_tensor, imgR_tensor = preprocess_stereo(imgL, imgR)

# ------------------------------
# Run Inference with PSMNet
# ------------------------------
with torch.no_grad():
    output = model(imgL_tensor, imgR_tensor)
    predicted_depth = output.squeeze().cpu().numpy()

# Normalize predicted disparity for visualization
predicted_depth_norm = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())

plt.figure(figsize=(6, 5))
plt.imshow(predicted_depth_norm, cmap="inferno")
plt.colorbar(label="Predicted Disparity")
plt.title("Predicted Disparity Map from PSMNet")
plt.axis("off")
plt.savefig("predicted_disparity_map.png", bbox_inches="tight")
plt.pause(0.001)

# ------------------------------
# Load and preprocess ground truth depth map from PFM
# ------------------------------
def read_pfm(file):
    with open(file, "rb") as f:
        header = f.readline().decode().rstrip()
        if header == "Pf":
            color = False
        elif header == "PF":
            color = True
        else:
            raise ValueError("Not a PFM file.")
        dims = f.readline().decode("utf-8").rstrip()
        width, height = map(int, dims.split())
        scale = float(f.readline().decode())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(f, endian + "f")
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data, abs(scale)

pfm_path = DISPARITY_GT  # Ground truth disparity map
disparity_map, _ = read_pfm(pfm_path)
# Handle NaN or infinite values (replace with 0)
disparity_map = np.nan_to_num(disparity_map, nan=0.0, posinf=0.0, neginf=0.0)

# ------------------------------
# Quantitative Metrics: Compare DL prediction with Ground Truth
# ------------------------------
# Resize for comparison
predicted_depth_resized = cv2.resize(predicted_depth, (target_W, target_H), interpolation=cv2.INTER_LINEAR)
gt_disparity_resized = cv2.resize(disparity_map, (target_W, target_H), interpolation=cv2.INTER_LINEAR)

MAE = np.mean(np.abs(predicted_depth_resized - gt_disparity_resized))
RMSE = np.sqrt(np.mean((predicted_depth_resized - gt_disparity_resized) ** 2))

print(f"Mean Absolute Error (MAE): {MAE:.4f}")
print(f"Root Mean Squared Error (RMSE): {RMSE:.4f}")

# ------------------------------
# Display side-by-side comparisons
# ------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].imshow(gt_disparity_resized, cmap="inferno")
axs[0].set_title("Ground Truth Disparity Map")
axs[0].axis("off")
axs[1].imshow(predicted_depth_resized, cmap="inferno")
axs[1].set_title("Predicted Disparity Map")
axs[1].axis("off")
plt.savefig("Comparison.png", bbox_inches="tight")

plt.show()
