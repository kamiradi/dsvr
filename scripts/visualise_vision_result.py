import rerun as rr
import sys
import os
import numpy as np
from dsvr.datasets.robot_data import RoboticsDatasetV2
from dsvr.results.robot_results import VisionInferenceResultV3
import argparse
from urllib.parse import quote as _q, unquote as _uq
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from typing import Dict
from rerun.urdf import UrdfTree


ap = argparse.ArgumentParser()
ap.add_argument("data_npz", help="Path to dataset .npz saved by RoboticsDataset")
ap.add_argument("result_npz", help="Path to result .npz saved by Result")
ap.add_argument(
    "--urdf",
    type=str,
    default=os.path.expanduser("~/Documents/workspace/iiwa_description/urdf/iiwa14.urdf"),
    help="Path to URDF file (default: ~/Documents/workspace/iiwa_description/urdf/iiwa14.urdf)",
)
ap.add_argument(
    "--root-path",
    type=str,
    default="/world/robot",
    help="Root entity path for the robot in Rerun (default: world/robot)",
)
ap.add_argument(
    "--start",
    type=int,
    default=800,
    help="Start frame index (default: 800)",
)
ap.add_argument(
    "--end",
    type=int,
    default=1000,
    help="End frame index (default: 1000)",
)
ap.add_argument(
    "--step",
    type=int,
    default=4,
    help="Frame step (default: 4)",
)
ap.add_argument(
    "--peg-mesh",
    type=str,
    default=os.path.expanduser("~/Documents/workspace/assembly_description/urdf/meshes/rectangular_peg.obj"),
    help="Path to peg mesh OBJ file",
)
ap.add_argument(
    "--seg-mask",
    action="store_true",
    help="Visualise segmentation masks (requires RoboticsDatasetV3 with seg_mask data)",
)
args = ap.parse_args()

# some useful colors
particle_color = [10, 10, 10]
ground_truth_color = [0, 255, 0]
estimate_color = [0, 0, 200, 100]

rr.init("results", spawn=True)
# Load dataset
res = VisionInferenceResultV3.load(args.result_npz)
if args.seg_mask:
    from dsvr.datasets.robot_data import RoboticsDatasetV3
    ds = RoboticsDatasetV3.load(args.data_npz)
    assert ds.seg_mask is not None, "Dataset has no seg_mask data"
else:
    ds = RoboticsDatasetV2.load(args.data_npz)
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

# Load and setup URDF robot visualization
rr.log_file_from_path(args.urdf, entity_path_prefix=args.root_path, static=True)
urdf_tree = UrdfTree.from_file_path(args.urdf)

# Get joint names from URDF tree
joint_names = [joint.name for joint in urdf_tree.joints() if joint.joint_type in ("revolute", "continuous", "prismatic")]
print(f"Loaded URDF with actuated joints: {joint_names}")

# Build joint name to index mapping
joint_name_to_idx: Dict[str, int] = {name: i for i, name in enumerate(joint_names)}

# initialize camera parameters
cam_ent = "/world/camera"
width = 640
height = 480
fov_y = np.pi/4
f_x = f_y = 0.5 * height / np.tan(0.5 * fov_y)
c_x, c_y = (width / 2.0 , height / 2.0 )
s = 0.0
K = np.array([[f_x, s,  c_x],
              [0.0, f_y, c_y],
              [0.0, 0.0, 1.0]], dtype=float)

times, mats = ds.se3_traj['X_Camera']
hole_times, hole_mats = ds.se3_traj['X_Hole']
peg_times, peg_mats = ds.se3_traj['X_Peg']
T = mats.shape[0]
count=0
half_size = np.array([0.04, 0.04, 0.025])

# Log static identity transforms for parent entities to establish transform hierarchy
# rr.log("world/origin", rr.Transform3D(translation=[0, 0, 0]), static=True)
rr.log(
    "/world/origin/",
    rr.Arrows3D(
        vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    ),
    rr.Transform3D(translation=[0, 0, 0]),
    static=True)
# rr.log("/world/particles", rr.Transform3D(translation=[0, 0, 0]), static=True)
# rr.log("/world/sampled_image", rr.Transform3D(translation=[0, 0, 0]), static=True)
# rr.log("/world/pixelwise", rr.Transform3D(translation=[0, 0, 0]), static=True)
# rr.log(cam_ent, rr.Transform3D(translation=[0, 0, 0]))

# Log peg mesh as static asset
peg_ent = "/world/peg"
rr.log(peg_ent, rr.Asset3D(path=args.peg_mesh), static=True)

frame_range = range(args.start, args.end, args.step)
total_frames = len(frame_range)
for k in frame_range:
    d = times[k]
    # Set time FIRST before logging any data for this frame
    rr.set_time("frame", duration=float(d))
    X_Cam = mats[k]
    p_WCam_W = X_Cam[:3, 3]
    R_WCam = X_Cam[:3, :3]

    # extract image closest to current transform
    ts_ind = 0
    for ts in range(len(ds.image_ts)):
        if ds.image_ts[ts] >= d:
            ts_ind = ts
            break
    # Explicitly draw a coordinate frame at the world origin
    # rr.log("world/origin", rr.Transform3D(translation=[0,0,0]))

    # # Add axes (length in meters)
    # rr.log(
    #     "world/origin/axes",
    #     rr.Arrows3D(
    #        vectors=[[1,0,0],[0,1,0],[0,0,1]],
    #         colors=[[255,0,0],[0,255,0],[0,0,255]],  # X=red, Y=green, Z=blue
    #         origins=[[0,0,0],[0,0,0],[0,0,0]]),
    #     rr.Transform3D(translation=[0,0,0]))

    # log particle images for the 2 best particles only
    weights = res.unnormalised_log_pdfs[ts_ind]
    top2 = np.argsort(weights)[-2:]
    for i in top2:
        rr.log(f"/world/sampled_image/sample_image_{i}", rr.DepthImage(res.images[ts_ind][i],
                                                                      meter=1.0, colormap="turbo"))
    for i in top2:
        rr.log(f"/world/pixelwise/pixelwise_score_{i}", rr.DepthImage(res.pixelwise_score[ts_ind][i],
                                                           meter=1.0,
                                                           colormap="viridis"))
    # iterating through particles
    for i, t in enumerate(res.poses[ts_ind]):
        translation = t[:3, 3]
        rotation = t[:3, :3]
        gt = f"/world/ground_truth"
        ent = f"/world/particles/particle_{i}"
        rr.log(
            f"{ent}",
            rr.Transform3D(
                mat3x3=rotation,
                translation=translation,
            ),
        )
        rr.log(
            f"{gt}",
            rr.Transform3D(
                mat3x3=hole_mats[0][:3, :3],
                translation=hole_mats[0][:3, 3],
            ),
        )
        weights = res.unnormalised_log_pdfs[ts_ind]
        # print weight stats like min, max mean
        norm = Normalize(vmin=min(weights), vmax=max(weights))
        normalized_weight = norm(weights[i])
        particle_color = cm.viridis(normalized_weight)

        rr.log(f"{ent}/geom", rr.Boxes3D(
            centers=[[0.0, 0.0, 0.0]],
            half_sizes=[half_size],
            colors=[particle_color],   # optional tint
        ))
        # Orientation line: local X-axis projected onto XY plane
        x_axis_world = rotation[:, 0]
        xy_dir = np.array([x_axis_world[0], x_axis_world[1], 0.0])
        xy_norm = np.linalg.norm(xy_dir)
        if xy_norm > 1e-6:
            xy_dir /= xy_norm
        arrow_len = 0.04
        rr.log(f"/world/orientations/particle_{i}", rr.Arrows3D(
            origins=[translation],
            vectors=[xy_dir * arrow_len],
            colors=[[255, 255, 255]],
        ))
        rr.log(f"{gt}/geom", rr.Boxes3D(
            centers=[[0.0, 0.0, 0.0]],
            half_sizes=[half_size],
            colors=[ground_truth_color],   # optional tint
        ))

    for i, score in enumerate(res.unnormalised_log_pdfs[ts_ind]):
        rr.log(f"/world/scores/score_{i}", rr.Scalars(score))
        #rr.log("world/ft/fx", rr.Scalars(fx))

    # log relevant dataset
    rr.log(
        f"{cam_ent}",
        rr.Transform3D(
            mat3x3=R_WCam,
            translation=p_WCam_W,
        ),
    )
    rr.log(
        f"{cam_ent}",
        rr.Pinhole(
            resolution=[640, 480],
            image_from_camera=K,
            camera_xyz=rr.ViewCoordinates.RDF
        ),
    )
    rr.log(f"{cam_ent}/rgb", rr.Image(ds.images[ts_ind]))
    rr.log(f"{cam_ent}/depth", rr.DepthImage(ds.depth[ts_ind], meter=1.0))

    # Log segmentation mask
    if args.seg_mask and ds.seg_mask is not None:
        seg_idx = 0
        for si in range(len(ds.seg_mask_ts)):
            if ds.seg_mask_ts[si] >= d:
                seg_idx = si
                break
        rr.log(f"{cam_ent}/seg_mask", rr.SegmentationImage(ds.seg_mask[seg_idx]))

    # rr.log(f"{cam_ent}/rgb", rr.Transform3D(translation=[0, 0, 0]), static=True)
    # rr.log(f"{cam_ent}/depth", rr.Transform3D(translation=[0, 0, 0]), static=True)

    # Log peg pose
    X_Peg = peg_mats[k]
    rr.log(
        peg_ent,
        rr.Transform3D(
            mat3x3=X_Peg[:3, :3],
            translation=X_Peg[:3, 3],
        ),
    )

    # Log robot joint states if available
    if ds.joint_states is not None and ds.joint_ts is not None:
        # Find closest joint state timestamp to current frame time
        joint_idx = 0
        for ji in range(len(ds.joint_ts)):
            if ds.joint_ts[ji] >= d:
                joint_idx = ji
                break
        q = ds.joint_states[joint_idx]

        # Log transforms for each joint
        for joint in urdf_tree.joints():
            if joint.name in joint_name_to_idx:
                idx = joint_name_to_idx[joint.name]
                if idx < len(q):
                    angle = q[idx]
                    transform = joint.compute_transform(angle)
                    # can you print the data type of the transform?
                    rr.log(f"{args.root_path}/{joint.child_link}", transform)

    count+=1
    print(f"\rProcessing frame {count}/{total_frames}...", end="", flush=True)

rr.log("/", rr.CoordinateFrame("root"), static=True)
rr.log("/", rr.Transform3D(translation=[0., 0., 0.],
                           parent_frame="root",
                           child_frame="world"), static=True)
rr.log("/world", rr.Transform3D(translation=[0., 0., 0.],
                                parent_frame="root",), static=True)
print(f"\rDone! Processed {count} frames.")
