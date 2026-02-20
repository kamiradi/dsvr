import rerun as rr
import os
import numpy as np
from dsvr.datasets.robot_data import RoboticsDatasetV3
import argparse
from typing import Dict
from rerun.urdf import UrdfTree
from vis_utils import depth_to_pointcloud

ap = argparse.ArgumentParser()
ap.add_argument("data_npz", help="Path to dataset .npz")
ap.add_argument("--mesh", type=str, required=True, help="Path to reconstruction .ply")
ap.add_argument(
    "--urdf",
    type=str,
    default=os.path.expanduser("~/Documents/workspace/iiwa_description/urdf/iiwa14.urdf"),
)
ap.add_argument("--root-path", type=str, default="/world/robot")
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
ap.add_argument("--prior-mesh", type=str, default=None, help="Path to prior shape .ply")
ap.add_argument("--hole-interior", type=str, default=None, help="Path to hole interior .ply (from isolate_hole.py)")
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--end", type=int, default=-1)
ap.add_argument("--step", type=int, default=4)
args = ap.parse_args()

rr.init("reconstruction", spawn=True)
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

# Load dataset
ds = RoboticsDatasetV3.load(args.data_npz)

# Load and setup URDF robot
rr.log_file_from_path(args.urdf, entity_path_prefix=args.root_path, static=True)
urdf_tree = UrdfTree.from_file_path(args.urdf)
joint_names = [j.name for j in urdf_tree.joints() if j.joint_type in ("revolute", "continuous", "prismatic")]
joint_name_to_idx: Dict[str, int] = {name: i for i, name in enumerate(joint_names)}

# Camera intrinsics
cam_ent = "/world/camera"
width, height = 640, 480
fov_y = np.pi / 4
f_x = f_y = 0.5 * height / np.tan(0.5 * fov_y)
c_x, c_y = width / 2.0, height / 2.0
K = np.array([[f_x, 0.0, c_x],
              [0.0, f_y, c_y],
              [0.0, 0.0, 1.0]], dtype=float)

# Trajectories
times, mats = ds.se3_traj['X_Camera']
hole_times, hole_mats = ds.se3_traj['X_Hole']
peg_times, peg_mats = ds.se3_traj['X_Peg']
T = mats.shape[0]

# Static elements: world origin axes, reconstruction mesh, GT hole pose
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

# Load mesh and log with vertex colors via Mesh3D
import trimesh
_mesh = trimesh.load(args.mesh)
rr.log("/world/reconstruction",
       rr.Mesh3D(
           vertex_positions=_mesh.vertices,
           triangle_indices=_mesh.faces,
           vertex_colors=_mesh.visual.vertex_colors[:, :3],
       ), static=True)

if args.prior_mesh is not None:
    _prior = trimesh.load(args.prior_mesh)
    rr.log("/world/prior_shape",
           rr.Mesh3D(
               vertex_positions=_prior.vertices,
               triangle_indices=_prior.faces,
               vertex_colors=np.full((len(_prior.vertices), 3), 180, dtype=np.uint8),
           ), static=True)

if args.hole_interior is not None:
    _interior = trimesh.load(args.hole_interior, force="mesh")
    rr.log("/world/hole_interior",
           rr.Mesh3D(
               vertex_positions=_interior.vertices,
               triangle_indices=_interior.faces,
               vertex_colors=np.full((len(_interior.vertices), 3), [255, 165, 0], dtype=np.uint8),
           ), static=True)

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

# Log peg mesh as static asset
peg_ent = "/world/peg"
rr.log(peg_ent, rr.Asset3D(path=args.peg_mesh), static=True)

# Frame loop
end = args.end if args.end > 0 else T
frame_range = range(args.start, end, args.step)
total_frames = len(frame_range)
count = 0

for k in frame_range:
    d = times[k]
    rr.set_time("frame", duration=float(d))

    X_Cam = mats[k]

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
    rr.log(f"{cam_ent}/rgb", rr.Image(ds.images[ts_ind]))
    rr.log(f"{cam_ent}/depth", rr.DepthImage(ds.depth[ts_ind], meter=1.0))

    # Point cloud from depth in world frame
    pts_world = depth_to_pointcloud(ds.depth[ts_ind], f_x, f_y, c_x, c_y, T_wc=X_Cam)
    if pts_world.shape[0] > 0:
        rr.log("/world/pointcloud", rr.Points3D(pts_world, colors=[200, 200, 255]))

    # Peg pose
    rr.log(peg_ent, rr.Transform3D(mat3x3=peg_mats[k][:3, :3], translation=peg_mats[k][:3, 3]))

    # Robot joint states
    if ds.joint_states is not None and ds.joint_ts is not None:
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

rr.log("/", rr.CoordinateFrame("root"), static=True)
rr.log("/", rr.Transform3D(translation=[0., 0., 0.],
                           parent_frame="root",
                           child_frame="world"), static=True)
rr.log("/world", rr.Transform3D(translation=[0., 0., 0.],
                                parent_frame="root"), static=True)
print(f"\rDone! Processed {count} frames.")
