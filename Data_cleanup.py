from PIL import Image
import numpy as np
import os
import glob
import matplotlib.pyplot as plt


def get_image_pixels_from_folders_normalized(folders, extensions=("jpg", "jpeg", "png", "gif")):
    image_pixels = []
    for folder in folders:
        for ext in extensions:
            for image_path in glob.glob(os.path.join(folder, f"*.{ext}")):
                try:
                    img = Image.open(image_path)  # Open the image
                    img = img.convert("RGB")  # Ensure the image is in RGB format
                    img_pixels = np.array(img, dtype=np.float32) / 255.0  # Convert to NumPy array and normalize to [0, 1]
                    image_pixels.append(img_pixels)
                except Exception as e:
                    print(f"Failed to process {image_path}: {e}")
    return image_pixels


folders = ["DataScienceProject/GNN_Dataset/seg_train/seg_train/buildings"]
image_pixels_list = get_image_pixels_from_folders_normalized(folders)

def show_img(img):
  plt.imshow(img.permute(1, 2, 0))
  plt.show()

print(image_pixels_list[0])
plt.imshow(image_pixels_list[0])
plt.axis('off')  # Hide axis labels for better visualization
plt.show()