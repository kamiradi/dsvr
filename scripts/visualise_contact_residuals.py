import rerun as rr
import sys
import os
import numpy as np
from dsvr.datasets.robot_data import RoboticsDatasetV2
from dsvr.results.robot_results import ContactResidualResult
import argparse
from matplotlib import cm
from matplotlib.colors import Normalize
from typing import Dict
from rerun.urdf import UrdfTree


ap = argparse.ArgumentParser()
ap.add_argument("data_npz", help="Path to dataset .npz saved by RoboticsDataset")
ap.add_argument("result_npz", help="Path to ContactResidualResult .npz")
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
    default=0,
    help="Start frame index (default: 0)",
)
ap.add_argument(
    "--end",
    type=int,
    default=-1,
    help="End frame index (default: -1 for all)",
)
ap.add_argument(
    "--step",
    type=int,
    default=10,
    help="Frame step (default: 10)",
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
args = ap.parse_args()

# Colors
contact_point_color = [255, 255, 255]  # Red for contact point
ground_truth_color = [0, 0, 0]   # Green for ground truth
force_color = [255, 165, 0]        # Orange for force vector

rr.init("contact_residuals", spawn=True)

# Load dataset and results
print(f"Loading dataset from {args.data_npz}")
ds = RoboticsDatasetV2.load(args.data_npz)
print(f"Loading contact residuals from {args.result_npz}")
res = ContactResidualResult.load(args.result_npz)

print(f"Dataset summary:")
print(f"  F/T samples: {len(ds.ft_ts)}")
print(f"Result summary:")
print(res.summary())

rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

# Load and setup URDF robot visualization
rr.log_file_from_path(args.urdf, entity_path_prefix=args.root_path, static=True)
urdf_tree = UrdfTree.from_file_path(args.urdf)

# Get joint names from URDF tree
joint_names = [joint.name for joint in urdf_tree.joints() if joint.joint_type in ("revolute", "continuous", "prismatic")]
print(f"Loaded URDF with actuated joints: {joint_names}")

# Build joint name to index mapping
joint_name_to_idx: Dict[str, int] = {name: i for i, name in enumerate(joint_names)}

# Initialize camera parameters
cam_ent = "/world/camera"
width = 640
height = 480
fov_y = np.pi/4
f_x = f_y = 0.5 * height / np.tan(0.5 * fov_y)
c_x, c_y = (width / 2.0 , height / 2.0)
K = np.array([[f_x, 0.0, c_x],
              [0.0, f_y, c_y],
              [0.0, 0.0, 1.0]], dtype=float)

# Load trajectories from dataset
cam_times, cam_mats = ds.se3_traj['X_Camera']
hole_times, hole_mats = ds.se3_traj['X_Hole']
peg_times, peg_mats = ds.se3_traj['X_Peg']
ftsense_times, ftsense_mats = ds.se3_traj['X_Ftsense']
con_times, con_mats = ds.se3_traj['X_Con']

# Log static elements
rr.log(
    "/world/origin/",
    rr.Arrows3D(
        vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    ),
    rr.Transform3D(translation=[0, 0, 0]),
    static=True)

# Log peg mesh as static asset
peg_ent = "/world/peg"
rr.log(peg_ent, rr.Asset3D(path=args.peg_mesh), static=True)

# Log hole mesh as static asset (at ground truth position)
hole_ent = "/world/hole"
rr.log(hole_ent, rr.Asset3D(path=args.hole_mesh), static=True)
# Set hole transform to ground truth
X_Hole_gt = hole_mats[0]
rr.log(
    hole_ent,
    rr.Transform3D(
        mat3x3=X_Hole_gt[:3, :3],
        translation=X_Hole_gt[:3, 3],
    ),
    static=True
)

# Log ground truth hole box
half_size = np.array([0.04, 0.04, 0.025])
rr.log(
    "/world/ground_truth",
    rr.Transform3D(
        mat3x3=hole_mats[0][:3, :3],
        translation=hole_mats[0][:3, 3],
    ),
    static=True
)
rr.log("/world/ground_truth/geom", rr.Boxes3D(
    centers=[[0.0, 0.0, 0.0]],
    half_sizes=[half_size],
    colors=[ground_truth_color],
), static=True)

# Determine frame range
end_idx = args.end if args.end > 0 else res.T
frame_range = range(args.start, min(end_idx, res.T), args.step)
total_frames = len(frame_range)

print(f"Visualizing {total_frames} frames from {args.start} to {end_idx} (step {args.step})")

count = 0
for k in frame_range:
    t = res.times[k]
    rr.set_time("frame", duration=float(t))

    # Get residuals for this timestep
    residuals = res.residuals[k]  # (P,)
    ft_value = res.ft_values[k]   # (6,)
    has_contact = res.contact_mask[k] > 0
    min_idx = res.min_residual_indices[k]

    # Find closest peg pose by time
    peg_idx = np.searchsorted(peg_times, t, side='right') - 1
    peg_idx = max(0, min(peg_idx, len(peg_mats) - 1))
    X_Peg = peg_mats[peg_idx]

    # Find closest X_Ftsense pose by time
    ftsense_idx = np.searchsorted(ftsense_times, t, side='right') - 1
    ftsense_idx = max(0, min(ftsense_idx, len(ftsense_mats) - 1))
    X_Ftsense = ftsense_mats[ftsense_idx]
    t_ftsense = X_Ftsense[:3, 3]

    # Transform mesh points from local to world frame using X_Peg
    R_peg = X_Peg[:3, :3]
    t_peg = X_Peg[:3, 3]
    mesh_points_world = (R_peg @ res.mesh_points.T).T + t_peg

    # Color mesh points by residual
    if has_contact:
        # Normalize residuals for coloring (log scale for better visualization)
        finite_residuals = residuals[np.isfinite(residuals)]
        if len(finite_residuals) > 0:
            # Use log scale and clip for visualization
            log_residuals = np.log1p(np.clip(residuals, 0, 1e6))
            norm = Normalize(vmin=np.min(log_residuals[np.isfinite(log_residuals)]),
                           vmax=np.max(log_residuals[np.isfinite(log_residuals)]))
            colors = cm.jet_r(norm(log_residuals))[:, :3] * 255  # RGB, 0-255
        else:
            # Fallback: lowest color in colormap
            colors = np.full((len(res.mesh_points), 3), cm.jet(0.0)[:3]) * 255
    else:
        # No contact: show all points with lowest color in colormap
        colors = np.full((len(res.mesh_points), 3), cm.jet_r(1.0)[:3]) * 255

    # Log residual point cloud (transformed to world frame)
    rr.log(
        "/world/residual_cloud",
        rr.Points3D(
            positions=mesh_points_world,
            colors=colors.astype(np.uint8),
            radii=0.002,
        ),
    )

    if has_contact:
        # Log estimated contact point (larger, red) - also transformed
        contact_point_local = res.mesh_points[min_idx]
        contact_point_world = R_peg @ contact_point_local + t_peg
        rr.log(
            "/world/contact_point",
            rr.Points3D(
                positions=[contact_point_world],
                colors=[contact_point_color],
                radii=0.005,
            ),
        )

        # Log min residual as scalar
        rr.log("/world/metrics/min_residual", rr.Scalars(res.min_residuals[k]))

        # Log ground truth contact location from X_Con
        con_idx = np.searchsorted(con_times, t, side='right') - 1
        con_idx = max(0, min(con_idx, len(con_mats) - 1))
        gt_contact_pos = con_mats[con_idx][:3, 3]
        rr.log(
            "/world/gt_contact_point",
            rr.Points3D(
                positions=[gt_contact_pos],
                colors=[ground_truth_color],
                radii=0.005,
            ),
        )

    # Log force vector at X_Ftsense origin (in world frame)
    force = ft_value[:3]
    force_norm = np.linalg.norm(force)
    if force_norm > 1e-6:
        # Scale force for visualization (0.01 m per N)
        force_scaled = force * 0.01
        rr.log(
            "/world/force_vector",
            rr.Arrows3D(
                vectors=[force_scaled],
                origins=[t_ftsense],
                colors=[force_color],
            ),
        )

    # Log F/T values as scalars
    rr.log("/world/ft/force_x", rr.Scalars(ft_value[0]))
    rr.log("/world/ft/force_y", rr.Scalars(ft_value[1]))
    rr.log("/world/ft/force_z", rr.Scalars(ft_value[2]))
    rr.log("/world/ft/torque_x", rr.Scalars(ft_value[3]))
    rr.log("/world/ft/torque_y", rr.Scalars(ft_value[4]))
    rr.log("/world/ft/torque_z", rr.Scalars(ft_value[5]))
    rr.log("/world/ft/force_magnitude", rr.Scalars(np.linalg.norm(ft_value[:3])))

    # Find closest camera frame to current time
    cam_idx = np.searchsorted(cam_times, t, side='right') - 1
    cam_idx = max(0, min(cam_idx, len(cam_mats) - 1))
    X_Cam = cam_mats[cam_idx]

    # Log camera pose and images
    rr.log(
        cam_ent,
        rr.Transform3D(
            mat3x3=X_Cam[:3, :3],
            translation=X_Cam[:3, 3],
        ),
    )
    rr.log(
        cam_ent,
        rr.Pinhole(
            resolution=[640, 480],
            image_from_camera=K,
            camera_xyz=rr.ViewCoordinates.RDF
        ),
    )

    # Find closest image timestamp
    img_idx = 0
    for idx in range(len(ds.image_ts)):
        if ds.image_ts[idx] >= t:
            img_idx = idx
            break
    rr.log(f"{cam_ent}/rgb", rr.Image(ds.images[img_idx]))
    rr.log(f"{cam_ent}/depth", rr.DepthImage(ds.depth[img_idx], meter=1.0))

    # Log peg pose
    rr.log(
        peg_ent,
        rr.Transform3D(
            mat3x3=X_Peg[:3, :3],
            translation=X_Peg[:3, 3],
        ),
    )

    # Log robot joint states if available
    if ds.joint_states is not None and ds.joint_ts is not None:
        joint_idx = np.searchsorted(ds.joint_ts, t, side='right') - 1
        joint_idx = max(0, min(joint_idx, len(ds.joint_states) - 1))
        q = ds.joint_states[joint_idx]

        for joint in urdf_tree.joints():
            if joint.name in joint_name_to_idx:
                idx = joint_name_to_idx[joint.name]
                if idx < len(q):
                    angle = q[idx]
                    transform = joint.compute_transform(angle)
                    rr.log(f"{args.root_path}/{joint.child_link}", transform)

    count += 1
    print(f"\rProcessing frame {count}/{total_frames}...", end="", flush=True)

rr.log("/", rr.CoordinateFrame("root"), static=True)
rr.log("/", rr.Transform3D(translation=[0., 0., 0.],
                           parent_frame="root",
                           child_frame="world"), static=True)
rr.log("/world", rr.Transform3D(translation=[0., 0., 0.],
                                parent_frame="root",), static=True)
print(f"\rDone! Processed {count} frames.")
