import rerun as rr
import os
import numpy as np
from dsvr.datasets.robot_data import RoboticsDatasetV3
import argparse
from typing import Dict
from rerun.urdf import UrdfTree
from vis_utils import depth_to_pointcloud

ap = argparse.ArgumentParser()
ap.add_argument("--data-path", required=True, help="Path to dataset.npz (e.g. /exp/data/dataset.npz)")
ap.add_argument("--result-path", required=True, help="Path to results directory (e.g. /exp/results)")
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
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--end", type=int, default=-1)
ap.add_argument("--step", type=int, default=4)
args = ap.parse_args()

def _maybe(d, f):
    p = os.path.join(d, f)
    return p if os.path.exists(p) else None

results_dir        = args.result_path
data_npz           = args.data_path
mesh_path          = os.path.join(results_dir, "reconstruction_measurement_10.ply")
if not os.path.exists(mesh_path):
    mesh_path      = os.path.join(results_dir, "reconstruction_measurement_9.ply")
if not os.path.exists(mesh_path):
    mesh_path      = os.path.join(results_dir, "reconstruction.ply")
envelope_mesh_path = _maybe(results_dir, "reconstruction_with_envelope.ply")
occ_pc_path        = _maybe(results_dir, "occupancy_pointcloud.ply")
scalar_fields_path = _maybe(results_dir, "scalar_fields.npz")

rr.init("reconstruction", spawn=True)
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

# Load dataset
ds = RoboticsDatasetV3.load(data_npz)

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

# Static elements: world origin axes
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
_mesh = trimesh.load(mesh_path)
rr.log("/world/reconstruction",
       rr.Mesh3D(
           vertex_positions=_mesh.vertices,
           triangle_indices=_mesh.faces,
           vertex_colors=_mesh.visual.vertex_colors[:, :3],
       ), static=True)

# Reconstruction with envelope
if envelope_mesh_path is not None:
    _env = trimesh.load(envelope_mesh_path)
    rr.log("/world/reconstruction_with_envelope",
           rr.Mesh3D(
               vertex_positions=_env.vertices,
               triangle_indices=_env.faces,
               vertex_colors=_env.visual.vertex_colors[:, :3],
           ), static=True)

# Occupancy point cloud
if occ_pc_path is not None:
    _occ = trimesh.load(occ_pc_path)
    occ_colors = np.array(_occ.colors[:, :3], dtype=np.uint8) if hasattr(_occ, 'colors') and _occ.colors is not None else None
    rr.log("/world/occupancy_pointcloud",
           rr.Points3D(np.array(_occ.vertices), colors=occ_colors, radii=0.004),
           static=True)

if scalar_fields_path is not None:
    import gpytoolbox as gpt
    from scipy.ndimage import map_coordinates
    _sf = np.load(scalar_fields_path)
    corner = _sf['corner']          # min corner of the grid, shape (3,)
    gs = _sf['gs']                  # number of cells per axis, shape (3,)
    h = _sf['h']                    # cell spacing per axis, shape (3,)
    scalar_mean = _sf['scalar_mean']
    grid_size = gs * h              # total extent in each axis
    grid_center = corner + grid_size / 2.0
    rr.log(
        "/world/spsr_grid",
        rr.Boxes3D(
            centers=[grid_center],
            half_sizes=[grid_size / 2.0],
            colors=[[100, 200, 255, 40]],  # pale blue, mostly transparent
        ),
        static=True,
    )

    G = gpt.fd_grad(gs=gs, h=h)
    grad_flat = G @ scalar_mean
    n_x = int((gs[0] - 1) * gs[1] * gs[2])
    n_y = int(gs[0] * (gs[1] - 1) * gs[2])
    grad_x = grad_flat[:n_x].reshape(gs[0] - 1, gs[1], gs[2], order='F')
    grad_y = grad_flat[n_x:n_x + n_y].reshape(gs[0], gs[1] - 1, gs[2], order='F')
    grad_z = grad_flat[n_x + n_y:].reshape(gs[0], gs[1], gs[2] - 1, order='F')
    corner_x = corner + np.array([0.5 * h[0], 0.0, 0.0])
    corner_y = corner + np.array([0.0, 0.5 * h[1], 0.0])
    corner_z = corner + np.array([0.0, 0.0, 0.5 * h[2]])

    step = max(1, int(gs[0] / 8))
    idx = [np.arange(step // 2, int(gs[d]), step) for d in range(3)]
    ii, jj, kk = np.meshgrid(*idx, indexing='ij')
    origins = np.stack([
        corner[0] + ii * h[0],
        corner[1] + jj * h[1],
        corner[2] + kk * h[2],
    ], axis=-1).reshape(-1, 3)

    gc_x = ((origins - corner_x[None, :]) / h[None, :]).T
    gc_y = ((origins - corner_y[None, :]) / h[None, :]).T
    gc_z = ((origins - corner_z[None, :]) / h[None, :]).T
    vecs = np.stack([
        map_coordinates(grad_x, gc_x, order=1, mode='nearest'),
        map_coordinates(grad_y, gc_y, order=1, mode='nearest'),
        map_coordinates(grad_z, gc_z, order=1, mode='nearest'),
    ], axis=1)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-4 * norms.max()
    arrow_len = h[0] * step * 0.8
    vecs_norm = np.where(norms > 1e-12, vecs / norms * arrow_len, 0.0)
    rr.log(
        "/world/prior_gradient",
        rr.Arrows3D(
            origins=origins[valid],
            vectors=vecs_norm[valid],
            colors=[0, 0, 0],
        ),
        static=True,
    )

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
