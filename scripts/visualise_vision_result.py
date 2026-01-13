import rerun as rr
import sys
import numpy as np
from dsvr.datasets.robot_data import RoboticsDatasetV2
from dsvr.results.robot_results import VisionInferenceResultV3
import argparse
from urllib.parse import quote as _q, unquote as _uq
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize


ap = argparse.ArgumentParser()
ap.add_argument("data_npz", help="Path to dataset .npz saved by RoboticsDataset")
ap.add_argument("result_npz", help="Path to result .npz saved by Result")
args = ap.parse_args()

# some useful colors
particle_color = [10, 10, 10]
ground_truth_color = [0, 255, 0]
estimate_color = [0, 0, 200, 100]

# Save .rrd file in same directory as result_npz
from pathlib import Path
result_path = Path(args.result_npz)
rrd_path = result_path.with_suffix(".rrd")

rr.init("results")
rr.save(str(rrd_path))
# Load dataset
res = VisionInferenceResultV3.load(args.result_npz)
ds = RoboticsDatasetV2.load(args.data_npz)
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP)

# initialize camera parameters
cam_ent = "world/camera"
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
count=0
half_size = np.array([0.04, 0.04, 0.025])
# for measurement in res.measurement_ids:
# iterate through all transforms
# for k in range(T):
frame_range = range(800, 1000, 4)
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
    rr.log("world/origin", rr.Transform3D(translation=[0,0,0]))

    # Add axes (length in meters)
    rr.log("world/origin/axes", rr.Arrows3D(
        vectors=[[1,0,0],[0,1,0],[0,0,1]],
        colors=[[255,0,0],[0,255,0],[0,0,255]],  # X=red, Y=green, Z=blue
        origins=[[0,0,0],[0,0,0],[0,0,0]],
    ))

    # log particles poses and generated images
    for i, image in enumerate(res.images[ts_ind]):
        rr.log(f"world/sampled_image/sample_image_{i}", rr.DepthImage(image, meter=0.5, colormap="turbo"))
    for i, pixel_image in enumerate(res.pixelwise_score[ts_ind]):
        rr.log(f"world/pixelwise/pixelwise_score_{i}", rr.DepthImage(pixel_image,
                                                           meter=1,
                                                           colormap="viridis"))
    # iterating through particles
    for i, t in enumerate(res.poses[ts_ind]):
        translation = t[:3, 3]
        rotation = t[:3, :3]
        gt = f"world/ground_truth"
        ent = f"world/particles/particle_{i}"
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
        rr.log(f"{gt}/geom", rr.Boxes3D(
            centers=[[0.0, 0.0, 0.0]],
            half_sizes=[half_size],
            colors=[ground_truth_color],   # optional tint
        ))

    for i, score in enumerate(res.unnormalised_log_pdfs[ts_ind]):
        rr.log(f"world/scores/score_{i}", rr.Scalars(score))
        #rr.log("world/ft/fx", rr.Scalars(fx))

    # log relevant dataset
    rr.log(f"{cam_ent}/rgb", rr.Image(ds.images[ts_ind]))
    rr.log(f"{cam_ent}/depth", rr.DepthImage(ds.depth[ts_ind], meter=1.0))
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

    count+=1
    print(f"\rProcessing frame {count}/{total_frames}...", end="", flush=True)

print(f"\rDone! Saved {count} frames to: {rrd_path}")
print(f"View with: rerun {rrd_path}")


