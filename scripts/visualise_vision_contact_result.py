import rerun as rr
import sys
import os
import numpy as np
from pathlib import Path
import yaml
from dsvr.datasets.robot_data import RoboticsDatasetV2
from dsvr.results.robot_results import VisionInferenceResultV3, ContactResidualResult
import argparse
from urllib.parse import quote as _q, unquote as _uq
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.cm as mcm
from matplotlib.colors import Normalize
from typing import Dict
from rerun.urdf import UrdfTree


def compute_ess(log_weights: np.ndarray) -> float:
    """Compute Effective Sample Size from unnormalized log weights."""
    log_weights = np.asarray(log_weights, dtype=float)
    log_weights -= log_weights.max()  # numerical stability
    weights = np.exp(log_weights)
    weights /= weights.sum()
    return float(1.0 / np.sum(weights ** 2))


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
    default=None,
    help="End frame index (default: last frame in dataset)",
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
    "--hole-mesh",
    type=str,
    default=os.path.expanduser("~/Documents/workspace/assembly_description/urdf/meshes/rectangular_hole.obj"),
    help="Path to hole mesh OBJ file",
)
ap.add_argument(
    "--seg-mask",
    action="store_true",
    help="Visualise segmentation masks (requires RoboticsDatasetV3 with seg_mask data)",
)
ap.add_argument(
    "--contact-residuals",
    type=str,
    default=None,
    help="Path to ContactResidualResult .npz file (optional)",
)
args = ap.parse_args()

# Derive experiment name and load config
# Expected path: .../output/{MonthYear}/{experiment_name}/scenario*/results/*.npz
_result_path = Path(args.result_npz)
experiment_name = _result_path.parent.parent.parent.name
_config_name = experiment_name if experiment_name.endswith(".yaml") else experiment_name + ".yaml"
_config_path = Path(os.path.expanduser("~/Documents/workspace/inverse_graphics/config/experiment")) / _config_name

# some useful colors
particle_color = [10, 10, 10]
ground_truth_color = [0, 255, 255]  # Cyan
estimate_color = [0, 0, 200, 100]

rr.init("results", spawn=True)

# Log experiment config as markdown
if _config_path.exists():
    _config_text = _config_path.read_text()
    _md = f"# `{experiment_name}`\n\n```yaml\n{_config_text}\n```"
else:
    _md = f"# `{experiment_name}`\n\n_Config not found: `{_config_path}`_"
rr.log("/experiment/config", rr.TextDocument(_md, media_type=rr.MediaType.MARKDOWN), static=True)

# Load dataset
res = VisionInferenceResultV3.load(args.result_npz)
if args.seg_mask:
    from dsvr.datasets.robot_data import RoboticsDatasetV3
    ds = RoboticsDatasetV3.load(args.data_npz)
    assert ds.seg_mask is not None, "Dataset has no seg_mask data"
else:
    ds = RoboticsDatasetV2.load(args.data_npz)
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

# Load contact residuals if provided
cr = None
if args.contact_residuals is not None:
    cr = ContactResidualResult.load(args.contact_residuals)
    print(f"Contact residuals: {cr.summary()}")

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
hole_ent = "/world/ground_truth"
arrow_len = 0.04

# Log static ground truth orientation arrow (cyan, same as GT mesh)
_gt_rot = hole_mats[0][:3, :3]
_gt_pos = hole_mats[0][:3, 3]
_gt_x = _gt_rot[:, 0]
_gt_xy = np.array([_gt_x[0], _gt_x[1], 0.0])
_gt_xy_norm = np.linalg.norm(_gt_xy)
if _gt_xy_norm > 1e-6:
    _gt_xy /= _gt_xy_norm
# rr.log(
#     "/world/ground_truth/orientation",
#     rr.Arrows3D(origins=[_gt_pos], vectors=[_gt_xy * arrow_len], colors=[[0, 255, 255]]),
#     static=True,
# )
rr.log(
    "/world/ground_truth/orientation",
    rr.Arrows3D(origins=[[0, 0, 0.06]], vectors=[_gt_xy * arrow_len], colors=[[0, 255, 255]]),
    static=True,
)

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

end = min(args.end, T) if args.end is not None else T
frame_range = range(args.start, end, args.step)
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
    ts_ind = min(ts_ind, len(res.unnormalised_log_pdfs) - 1)
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

    # select top-2 particles by weight, best-first
    weights = res.unnormalised_log_pdfs[ts_ind]
    top2 = np.argsort(weights)[-2:][::-1]
    top10 = set(np.argsort(weights)[-10:])
    norm = Normalize(vmin=min(weights), vmax=max(weights))

    # log particle images for the 2 best particles by rank
    for rank, i in enumerate(top2):
        rr.log(f"/world/sampled_image/sample_image_{rank+1}", rr.DepthImage(res.images[ts_ind][i],
                                                                      meter=1.0, colormap="turbo"))
    for rank, i in enumerate(top2):
        rr.log(f"/world/pixelwise/pixelwise_score_{rank+1}", rr.DepthImage(res.pixelwise_score[ts_ind][i],
                                                           meter=1.0,
                                                           colormap="viridis"))
    # iterating through all particles
    gt = "/world/ground_truth"
    for i, t in enumerate(res.poses[ts_ind]):
        translation = t[:3, 3]
        rotation = t[:3, :3]
        ent = f"/world/particles/particle_{i}"
        rr.log(
            ent,
            rr.Transform3D(
                mat3x3=rotation,
                translation=translation,
            ),
        )
        rr.log(
            gt,
            rr.Transform3D(
                mat3x3=hole_mats[0][:3, :3],
                translation=hole_mats[0][:3, 3],
            ),
        )
        alpha = int(norm(weights[i]) * 200 + 55)  # range 55-255
        rr.log(f"{ent}/geom", rr.Asset3D(
            path=args.hole_mesh,
            albedo_factor=[255, 165, 0, alpha],  # Orange with weight-based alpha
        ))
        # Orientation line: local X-axis projected onto XY plane
        x_axis_world = rotation[:, 0]
        xy_dir = np.array([x_axis_world[0], x_axis_world[1], 0.0])
        xy_norm = np.linalg.norm(xy_dir)
        if xy_norm > 1e-6:
            xy_dir /= xy_norm
        arrow_color = [255, 0, 0] if i in top10 else [0, 0, 0]
        rr.log(f"/world/orientations/particle_{i}", rr.Arrows3D(
            origins=[translation + np.array([0, 0, 0.06])],
            vectors=[xy_dir * arrow_len],
            colors=[arrow_color],
        ))
        rr.log(f"{gt}/geom", rr.Asset3D(
            path=args.hole_mesh,
            albedo_factor=[0, 255, 255, 255],  # Cyan, fully opaque
        ))

    for i, score in enumerate(weights):
        rr.log(f"/world/scores/score_{i}", rr.Scalars(score))
    ess = compute_ess(weights)
    rr.log("/world/ess", rr.Scalars(ess))
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

    # Contact residuals: coloured point cloud in world frame
    if cr is not None:
        cr_idx = int(np.clip(
            np.searchsorted(cr.times, d, side="right") - 1,
            0, cr.T - 1,
        ))
        peg_idx = int(np.clip(
            np.searchsorted(peg_times, d, side="right") - 1,
            0, len(peg_times) - 1,
        ))
        R_peg = peg_mats[peg_idx, :3, :3]
        t_peg = peg_mats[peg_idx, :3, 3]
        pts_W = (R_peg @ cr.mesh_points.T).T + t_peg
        likelihoods = cr.residuals[cr_idx]
        lo, hi = likelihoods.min(), likelihoods.max()
        norm_cr = (likelihoods - lo) / max(hi - lo, 1e-6)
        colors_cr = (mcm.turbo(norm_cr)[:, :3] * 255).astype(np.uint8)
        rr.log("/world/contact_residuals", rr.Points3D(pts_W, colors=colors_cr, radii=0.001))
        hist_counts, _ = np.histogram(likelihoods, bins=50)
        rr.log("/world/contact_residuals/distribution", rr.BarChart(hist_counts.astype(np.float32)))

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
