import rerun as rr
import rerun.blueprint as rrb
import numpy as np
from robot_data import RoboticsDatasetV2, VisionInferenceResultV3
import argparse
from urllib.parse import quote as _q, unquote as _uq

ap = argparse.ArgumentParser()
ap.add_argument("data_npz", help="Path to dataset .npz saved by RoboticsDataset")
ap.add_argument("result_npz", help="Path to result .npz saved by RoboticsDataset")
args = ap.parse_args()

rr.init("results", spawn=True)
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
for k in range(T):

    d = times[k]
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
    # for i, image in enumerate(res.images[ts_ind]):
    #     rr.log(f"world/sample_image_{i}", rr.DepthImage(image, meter=0.5, colormap="turbo"))
    # for i, pixel_image in enumerate(res.pixelwise_score[ts_ind]):
    #     rr.log(f"world/pixelwise_score_{i}", rr.DepthImage(pixel_image,
    #                                                        meter=1,
    #                                                        colormap="viridis"))
    for i, t in enumerate(res.poses[ts_ind]):
        translation = t[:3, 3]
        rotation = t[:3, :3]
        ent = f"world/particle_{i}"
        rr.log(
            f"{ent}",
            rr.Transform3D(
                mat3x3=rotation,
                translation=translation,
            ),
        )
        rr.log(f"{ent}/geom", rr.Boxes3D(
            centers=[[0.0, 0.0, 0.0]],
            half_sizes=[half_size],
            colors=[[200, 200, 255]],   # optional tint
        ))

    for i, score in enumerate(res.unnormalised_log_pdfs[ts_ind]):
        rr.log(f"world/score_{i}", rr.Scalars(score))
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

    # log time
    rr.set_time("frame", duration=float(d))
    count+=1


