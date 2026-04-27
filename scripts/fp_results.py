import os
import numpy as np
import argparse
from dsvr.datasets.robot_data import RoboticsDatasetV3
from dsvr.results.robot_results import VisionInferenceResultV3

FP_DEMO = "/home/aditya/Documents/workspace/FoundationPose/demo_data"
DATASET_BASE = "/home/aditya/Documents/workspace/inverse_graphics/output/March2026"
OUTPUT_BASE = "/mnt/big_beast/output/April2026"

SCENARIOS = [f"scenario{i}" for i in range(9)]


def process_scenario(ob_in_cam_dir, dataset_path, output_path):
    ds = RoboticsDatasetV3.load(dataset_path)
    image_ts = ds.image_ts
    x_cam_times, x_cam_mats = ds.se3_traj["X_Camera"]

    files = sorted(f for f in os.listdir(ob_in_cam_dir) if f.endswith(".txt"))
    if not files:
        print(f"No pose files in {ob_in_cam_dir}, skipping")
        return

    result = VisionInferenceResultV3.empty(1, image_shape=(480, 640, 1))

    for i, fname in enumerate(files):
        frame_idx = int(fname.split(".txt")[0])
        ts = image_ts[frame_idx]
        cam_idx = int(np.argmin(np.abs(x_cam_times - ts)))
        X_Camera = x_cam_mats[cam_idx]

        with open(os.path.join(ob_in_cam_dir, fname)) as f:
            T_ob_in_cam = np.array([[float(x) for x in line.strip().split()] for line in f])

        X_world_ob = X_Camera @ T_ob_in_cam

        result.add_measurement(
            measurement_id=i,
            poses=[X_world_ob],
            times=[ts],
            images=[np.zeros((480, 640, 1))],
            pixelwise_score=[np.zeros((480, 640, 1))],
            unnormalised_log_pdfs=[0.0],
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.save(output_path)
    print(f"Saved {len(files)} frames -> {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("geom", help="geometry name, e.g. data_sim_rectangle")
    args = ap.parse_args()

    geom = args.geom
    for scenario in SCENARIOS:
        ob_in_cam_dir = os.path.join(FP_DEMO, geom, scenario, "results", "ob_in_cam")
        dataset_path = os.path.join(DATASET_BASE, geom, scenario, "data", "dataset.npz")
        output_path = os.path.join(OUTPUT_BASE, f"fp_{geom}", scenario, "results", "vision_inference_results.npz")

        if not os.path.isdir(ob_in_cam_dir):
            print(f"Skipping {scenario}: no ob_in_cam dir at {ob_in_cam_dir}")
            continue
        if not os.path.isfile(dataset_path):
            print(f"Skipping {scenario}: no dataset.npz at {dataset_path}")
            continue

        print(f"Processing {geom}/{scenario}")
        process_scenario(ob_in_cam_dir, dataset_path, output_path)


if __name__ == "__main__":
    main()
