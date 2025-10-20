import os
import numpy as np
from dsvr.datasets.robot_data import RoboticsDatasetV2, RoboticsDatasetV3
import argparse
from PIL import Image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="Path to dataset .npz saved by RoboticsDataset")
    ap.add_argument("output", help="Path to output data saved by RoboticsDataset")
    ap.add_argument("label", type=int, help="Integer value representing the label")
    args = ap.parse_args()

    # Define the input and output paths
    output_folder = args.output

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder+"/depth", exist_ok=True)
    os.makedirs(output_folder+"/rgb", exist_ok=True)
    os.makedirs(output_folder+"/masks", exist_ok=True)

    # Load the dataset
    data = RoboticsDatasetV3.load(args.npz)

    # Check for required keys in the dataset
    # if "depth" not in data or "color" not in data:
    #     raise ValueError("The dataset must contain 'depth' and 'color' keys.")

    # Extract depth and color images
    depth_images = data.depth
    color_images = data.images
    label_images = data.seg_mask

    # Save depth and color images as PNGs
    for i, (label, depth, color) in enumerate(zip(label_images, depth_images, color_images)):
        depth_image_path = os.path.join(output_folder, f"/depth/{i:04d}.png")
        color_image_path = os.path.join(output_folder, f"/rgb/{i:04d}.png")
        label_image_path = os.path.join(output_folder, f"/masks/{i:04d}.png")

        # Save depth image (normalize to 0-255 for PNG format)
        depth_normalized = (255 * (depth - depth.min()) / (depth.max() - depth.min())).astype(np.uint8)
        Image.fromarray(depth_normalized).save(output_folder+depth_image_path)

        # label images
        print(np.unique(label, return_counts=True))
        # Create a binary mask where True if pixel label is in target_labels
        seg = np.where(label==args.label, 1, 0)

        # Convert to uint8 black and white image
        bw = (seg * 255).astype(np.uint8)
        Image.fromarray(bw).save(output_folder+label_image_path)

        # Save color image
        Image.fromarray(color).save(output_folder+color_image_path)

    print(f"Images saved to {output_folder}")

if __name__ == "__main__":
    main()
