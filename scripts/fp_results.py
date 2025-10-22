import os
import numpy as np
import sys
from dsvr.results.robot_results import VisionInferenceResultV3

def read_pose_matrices_from_directory(directory):
    # List all files in the directory and sort them sequentially
    files = sorted(f for f in os.listdir(directory) if f.endswith('.txt'))
    result = VisionInferenceResultV3.empty(1, image_shape=(480, 640, 1))


    for i, file in enumerate(files):
        file_path = os.path.join(directory, file)
        # Read the 4x4 pose matrix from the file
        particle_poses = []
        particle_images = []
        particle_pdfs = []
        particle_times = []
        particle_pixelwise = []
        with open(file_path, 'r') as f:
            matrix = []
            for line in f:
                matrix.append([float(x) for x in line.strip().split()])
            pose_matrix = np.array(matrix)
            particle_poses.append(pose_matrix)
            particle_images.append(np.zeros((480, 640, 1)))  # Placeholder for image
            particle_pixelwise.append(np.zeros((480, 640, 1)))  # Placeholder for pixelwise score
            particle_pdfs.append(0.0)
            # extract number before '.txt' as time and convert to int
            time_stamp = int(file.split('.txt')[0])
            particle_times.append(time_stamp)

        result.add_measurement(
            measurement_id=i,
            poses=particle_poses,
            times=particle_times,
            images=particle_images,
            pixelwise_score=particle_pixelwise,
            unnormalised_log_pdfs=particle_pdfs)

    return result

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        sys.exit(1)

    results = read_pose_matrices_from_directory(directory)
    print(results.summary())
    results.save("/home/aditya/Documents/working/oct2025/vision_inference_results.npz")
