"""
Robot configuration visualizer using Rerun.

Loads a URDF file and joint states from a dataset npz file,
then visualizes the robot configuration over time.

Based on: https://rerun.io/examples/robotics/animated_urdf
"""
from __future__ import annotations

import argparse
import os
from typing import Optional, Dict

import numpy as np
import rerun as rr
from rerun.urdf import UrdfTree

from dsvr.datasets.robot_data import RoboticsDatasetV2, RoboticsDatasetV3


def main():
    parser = argparse.ArgumentParser(
        description="Visualize robot configurations from a dataset using Rerun."
    )
    parser.add_argument(
        "data_npz",
        type=str,
        help="Path to dataset .npz file containing joint states",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "iiwa14_no_collision.urdf"),
        help="Path to URDF file (default: iiwa14_no_collision.urdf in script directory)",
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default="world/robot",
        help="Root entity path for the robot in Rerun (default: world/robot)",
    )
    parser.add_argument(
        "--v3",
        action="store_true",
        help="Use RoboticsDatasetV3 instead of V2",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start frame index (default: 0)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End frame index (default: last frame)",
    )
    args = parser.parse_args()

    # Initialize Rerun
    rr.init("robot_visualizer", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Load dataset
    if args.v3:
        ds = RoboticsDatasetV3.load(args.data_npz)
    else:
        ds = RoboticsDatasetV2.load(args.data_npz)

    print(f"Dataset summary: {ds.summary()}")

    if ds.joint_states is None or ds.joint_ts is None:
        print("Error: Dataset does not contain joint states.")
        return

    # Log the URDF file as a static resource
    rr.log_file_from_path(args.urdf, entity_path_prefix=args.root_path, static=True)

    # Load the URDF tree structure for animation
    urdf_tree = UrdfTree.from_file_path(args.urdf)

    # Get joint names from URDF tree
    joint_names = [joint.name for joint in urdf_tree.joints() if joint.joint_type in ("revolute", "continuous", "prismatic")]
    print(f"Loaded URDF with actuated joints: {joint_names}")

    # Build joint name to index mapping
    joint_name_to_idx: Dict[str, int] = {name: i for i, name in enumerate(joint_names)}

    # Log world origin axes
    rr.log(
        "world/origin/axes",
        rr.Arrows3D(
            vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ),
        static=True,
    )

    # Iterate through joint configurations
    num_configs = ds.joint_states.shape[0]
    start_idx = max(0, args.start)
    end_idx = min(num_configs, args.end) if args.end is not None else num_configs
    print(f"Logging frames {start_idx} to {end_idx} (of {num_configs} total)...")

    for i in range(start_idx, end_idx):
        t = ds.joint_ts[i]
        q = ds.joint_states[i]

        # Set time for this frame
        rr.set_time("time", duration=t)

        # Log transforms for each joint
        for joint in urdf_tree.joints():
            if joint.name in joint_name_to_idx:
                idx = joint_name_to_idx[joint.name]
                if idx < len(q):
                    angle = q[idx]
                    # compute_transform returns a transform ready to log
                    transform = joint.compute_transform(angle)
                    rr.log(f"{args.root_path}/{joint.child_link}", transform)

    print("Done.")


if __name__ == "__main__":
    main()
