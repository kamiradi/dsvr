import rerun as rr
import rerun.blueprint as rrb
import numpy as np
from robot_data import RoboticsDatasetV2
import argparse
from urllib.parse import quote as _q, unquote as _uq
from vis_utils import depth_to_pointcloud, filter_pointcloud_bbox

ap = argparse.ArgumentParser()
ap.add_argument("npz", help="Path to dataset .npz saved by RoboticsDataset")
args = ap.parse_args()

rr.init("dataset_visual", spawn=True)
# Load dataset
ds = RoboticsDatasetV2.load(args.npz)

bbox_min = np.array([0.0, -0.2, 0.005])
bbox_max = np.array([0.8, 0.2, 0.05])

cam_ent = "world/camera"
image_cam_ent = cam_ent+"/image"
depth_cam_ent = cam_ent+"/depth"
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
T = mats.shape[0]
for i in range(T):
    d = times[i]
    t = mats[i]
    translation = t[:3, 3]
    rotation = t[:3, :3]
    # Explicitly draw a coordinate frame at the world origin
    rr.log("world/origin", rr.Transform3D(translation=[0,0,0]))

    # Add axes (length in meters)
    rr.log("world/origin/axes", rr.Arrows3D(
        vectors=[[1,0,0],[0,1,0],[0,0,1]],
        colors=[[255,0,0],[0,255,0],[0,0,255]],  # X=red, Y=green, Z=blue
        origins=[[0,0,0],[0,0,0],[0,0,0]],
    ))
    rr.log(
        f"{cam_ent}",
        rr.Transform3D(
            mat3x3=rotation,
            translation=translation,
        ),
    )
    # extract image closest to time T
    # only works if images are fewer than transforms
    ts_ind = 0
    for k in range(len(ds.image_ts)):
        if ds.image_ts[k] >= d:
            ts_ind = k
            break
    rr.log(f"{cam_ent}/rgb", rr.Image(ds.images[ts_ind]))
    rr.log(f"{cam_ent}/depth", rr.DepthImage(ds.depth[ts_ind], meter=1.0))

    points = depth_to_pointcloud(ds.depth[ts_ind], f_x, f_y, c_x, c_y, T_wc=t)
    filtered_points = filter_pointcloud_bbox(points, bbox_min, bbox_max)
    rr.log(f"world/points", rr.Points3D(filtered_points, colors=[200,200,255]))

    rr.log(
        f"{cam_ent}",
        rr.Pinhole(
            resolution=[640, 480],
            image_from_camera=K,
            camera_xyz=rr.ViewCoordinates.RDF
        ),
    )
    rr.set_time("frame", duration=float(d))

# for img, t in zip(ds.images, ds.image_ts):
#     rr.set_time("frame", duration=float(t))
#     rr.log(f"{cam_ent}/rgb", rr.Image(img))


# for d, t in zip(ds.depth, ds.depth_ts):
#     rr.set_time("frame", duration=float(t))
#     rr.log(f"{cam_ent}/depth", rr.DepthImage(d, meter=0.5))



for sample, t in zip(ds.ft, ds.ft_ts):
    fx, fy, fz, tx, ty, tz = [float(x) for x in sample]
    rr.set_time("frame", duration=float(t))
    # Scalar time series
    rr.log("world/ft/fx", rr.Scalars(fx))
    rr.log("world/ft/fy", rr.Scalars(fy))
    rr.log("world/ft/fz", rr.Scalars(fz))
    rr.log("world/ft/tx", rr.Scalars(tx))
    rr.log("world/ft/ty", rr.Scalars(ty))
    rr.log("world/ft/tz", rr.Scalars(tz))
