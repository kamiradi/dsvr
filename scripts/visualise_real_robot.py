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
        default="/world/robot",
        help="Root entity path for the robot in Rerun (default: /world/robot)",
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
    parser.add_argument(
        "--hole-mesh",
        type=str,
        default=os.path.expanduser("~/Documents/workspace/assembly_description/urdf/meshes/rectangular_hole.obj"),
        help="Path to hole mesh OBJ file",
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
        "/world/origin/",
        rr.Arrows3D(
            vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ),
        rr.Transform3D(translation=[0, 0, 0]),
        static=True,
    )

    # Camera setup — use depth intrinsics from dataset metadata
    import json as _json
    cam_ent = "/world/camera"
    meta = _json.loads(str(ds.metadata_json)) if ds.metadata_json is not None else {}
    depth_cam = meta.get("depth_camera", {})
    width  = depth_cam.get("width",  848)
    height = depth_cam.get("height", 480)
    K_flat = depth_cam.get("K", None)
    if K_flat is not None:
        K = np.array(K_flat, dtype=float).reshape(3, 3)
    else:
        fov_y = np.pi / 4
        f = 0.5 * height / np.tan(0.5 * fov_y)
        K = np.array([[f, 0., width / 2.], [0., f, height / 2.], [0., 0., 1.]], dtype=float)
    print(f"Depth camera: {width}x{height}\nK=\n{K}")

    # Log hole mesh at last known X_Hole pose (static ground truth)
    if (hasattr(ds, 'se3_traj') and ds.se3_traj is not None
            and 'X_Hole' in ds.se3_traj and os.path.exists(args.hole_mesh)):
        hole_times, hole_mats = ds.se3_traj['X_Hole']
        rr.log(
            "/world/ground_truth",
            rr.Transform3D(
                mat3x3=hole_mats[-1][:3, :3],
                translation=hole_mats[-1][:3, 3],
            ),
            static=True,
        )
        rr.log("/world/ground_truth/geom", rr.Asset3D(
            path=args.hole_mesh,
            albedo_factor=[0, 255, 255, 255],  # Cyan, fully opaque
        ), static=True)

    has_camera = (hasattr(ds, 'se3_traj') and ds.se3_traj is not None
                  and 'X_Camera' in ds.se3_traj)

    if has_camera:
        # Camera-primary loop: iterate over camera trajectory, look up closest joint state
        # (matches visualise_reconstruction.py structure)
        cam_times, cam_mats = ds.se3_traj['X_Camera']
        num_frames = cam_mats.shape[0]
        start_idx = max(0, args.start)
        end_idx = min(num_frames, args.end) if args.end is not None else num_frames
        print(f"Logging frames {start_idx} to {end_idx} (of {num_frames} total)...")

        count = 0
        total_frames = end_idx - start_idx
        for k in range(start_idx, end_idx):
            d = cam_times[k]
            rr.set_time("time", duration=float(d))

            X_Cam = cam_mats[k]

            # Find closest image timestamp
            ts_ind = 0
            for ts in range(len(ds.image_ts)):
                if ds.image_ts[ts] >= d:
                    ts_ind = ts
                    break

            # Camera transform + pinhole
            rr.log(cam_ent, rr.Transform3D(mat3x3=X_Cam[:3, :3], translation=X_Cam[:3, 3]))
            rr.log(cam_ent, rr.Pinhole(resolution=[width, height], image_from_camera=K,
                                        camera_xyz=rr.ViewCoordinates.RDF))
            if ds.images is not None:
                rr.log(f"{cam_ent}/rgb", rr.Image(ds.images[ts_ind]))
            if ds.depth is not None and ds.depth_ts is not None:
                depth_ind = int(np.searchsorted(ds.depth_ts, d, side="right")) - 1
                depth_ind = max(0, min(depth_ind, len(ds.depth_ts) - 1))
                rr.log(f"{cam_ent}/depth", rr.DepthImage(ds.depth[depth_ind], meter=1.0))

            # Robot joint states (look up closest joint timestamp)
            joint_idx = 0
            for ji in range(len(ds.joint_ts)):
                if ds.joint_ts[ji] >= d:
                    joint_idx = ji
                    break
            q = ds.joint_states[joint_idx]
            for joint in urdf_tree.joints():
                if joint.name in joint_name_to_idx:
                    idx = joint_name_to_idx[joint.name]
                    if idx < len(q):
                        rr.log(f"{args.root_path}/{joint.child_link}",
                               joint.compute_transform(q[idx]))

            count += 1
            print(f"\rProcessing frame {count}/{total_frames}...", end="", flush=True)

    else:
        # Joint-only loop (no camera data)
        num_configs = ds.joint_states.shape[0]
        start_idx = max(0, args.start)
        end_idx = min(num_configs, args.end) if args.end is not None else num_configs
        print(f"Logging frames {start_idx} to {end_idx} (of {num_configs} total)...")

        for i in range(start_idx, end_idx):
            t = ds.joint_ts[i]
            q = ds.joint_states[i]

            rr.set_time("time", duration=t)

            for joint in urdf_tree.joints():
                if joint.name in joint_name_to_idx:
                    idx = joint_name_to_idx[joint.name]
                    if idx < len(q):
                        rr.log(f"{args.root_path}/{joint.child_link}",
                               joint.compute_transform(q[idx]))

    # Log F/T time series
    if ds.ft is not None and ds.ft_ts is not None:
        ft_channels = [
            ("ft/force/x", "Fx [N]",  [220,  50,  50]),
            ("ft/force/y", "Fy [N]",  [ 50, 200,  50]),
            ("ft/force/z", "Fz [N]",  [ 50, 100, 220]),
            ("ft/torque/x","Tx [Nm]", [220, 150,  50]),
            ("ft/torque/y","Ty [Nm]", [180,  50, 220]),
            ("ft/torque/z","Tz [Nm]", [ 50, 200, 200]),
        ]
        for path, label, color in ft_channels:
            rr.log(path, rr.SeriesLines(colors=[color], names=[label]), static=True)
        print("Logging F/T data...")
        for i in range(len(ds.ft_ts)):
            rr.set_time("time", duration=float(ds.ft_ts[i]))
            for j, (path, _, _) in enumerate(ft_channels):
                rr.log(path, rr.Scalars(float(ds.ft[i, j])))

    # Coordinate frame setup (matching visualise_reconstruction.py)
    rr.log("/", rr.CoordinateFrame("root"), static=True)
    rr.log("/", rr.Transform3D(translation=[0., 0., 0.],
                               parent_frame="root",
                               child_frame="world"), static=True)
    rr.log("/world", rr.Transform3D(translation=[0., 0., 0.],
                                    parent_frame="root"), static=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
