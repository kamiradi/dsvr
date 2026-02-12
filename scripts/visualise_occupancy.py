import rerun as rr
import numpy as np
from dsvr.datasets.robot_data import RoboticsDatasetV2
import pyoctomap
import argparse
from vis_utils import depth_to_pointcloud, filter_pointcloud_bbox

ap = argparse.ArgumentParser(description="Visualise OctoMap occupancy grid from dataset depth")
ap.add_argument("npz", help="Path to dataset .npz saved by RoboticsDatasetV2")
ap.add_argument("--resolution", type=float, default=0.005, help="OctoMap voxel resolution in meters")
ap.add_argument("--frame", type=int, default=0, help="Depth frame index to use (-1 for all frames)")
ap.add_argument("--bbox-min", type=float, nargs=3, default=None, help="Bounding box min [x y z]")
ap.add_argument("--bbox-max", type=float, nargs=3, default=None, help="Bounding box max [x y z]")
args = ap.parse_args()

# Load dataset
ds = RoboticsDatasetV2.load(args.npz)

# Camera intrinsics (matching visualise_dataset.py)
width, height = 640, 480
fov_y = np.pi / 4
f_x = f_y = 0.5 * height / np.tan(0.5 * fov_y)
c_x, c_y = width / 2.0, height / 2.0

# Camera poses
cam_times, cam_mats = ds.se3_traj['X_Camera']

# Determine which frames to process
if args.frame == -1:
    frame_indices = range(len(cam_times))
else:
    frame_indices = [args.frame]

# Build point cloud from selected frames
all_points = []
for i in frame_indices:
    pts = depth_to_pointcloud(ds.depth[i], f_x, f_y, c_x, c_y, T_wc=cam_mats[i])
    all_points.append(pts)
all_points = np.concatenate(all_points, axis=0)

# Optional bounding box filter
if args.bbox_min is not None and args.bbox_max is not None:
    all_points = filter_pointcloud_bbox(
        all_points, np.array(args.bbox_min), np.array(args.bbox_max)
    )

print(f"Point cloud: {all_points.shape[0]} points")

# Build OctoMap
tree = pyoctomap.OcTree(args.resolution)
sensor_origin = np.array([0.0, 0.0, 0.0])
tree.insertPointCloud(all_points.astype(np.float64), sensor_origin, lazy_eval=True)
tree.updateInnerOccupancy()

# Extract occupied voxels
centers = []
for leaf in tree.begin_leafs():
    if tree.isNodeOccupied(leaf):
        centers.append(leaf.getCoordinate())

centers = np.array(centers)
half_size = args.resolution / 2.0
print(f"Occupied voxels: {len(centers)}")

# Visualize in Rerun
rr.init("occupancy_map", spawn=True)

# Log world origin axes
rr.log("world/origin", rr.Transform3D(translation=[0, 0, 0]))
rr.log("world/origin/axes", rr.Arrows3D(
    vectors=[[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
))

# Log raw point cloud for comparison
rr.log("world/points", rr.Points3D(all_points, colors=[200, 200, 255], radii=0.001))

# Log occupied voxels as boxes
half_sizes = np.full((len(centers), 3), half_size)
rr.log("world/occupancy", rr.Boxes3D(
    centers=centers,
    half_sizes=half_sizes,
    colors=[100, 200, 100],
))
