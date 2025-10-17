import numpy as np

def filter_pointcloud_bbox(
    pts: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    return_mask: bool = False,
):
    """
    Filter a point cloud with an axis-aligned bounding box.

    Args:
        pts: (N,3) array of 3D points.
        bbox_min: (3,) array, lower corner [xmin, ymin, zmin].
        bbox_max: (3,) array, upper corner [xmax, ymax, zmax].
        return_mask: if True, also return the boolean mask.

    Returns:
        pts_in: (M,3) points inside the bounding box.
        (optional) mask: (N,) bool array, True if point is inside.
    """
    pts = np.asarray(pts)
    bbox_min = np.asarray(bbox_min)
    bbox_max = np.asarray(bbox_max)

    # elementwise check: min <= pt <= max
    mask = np.all((pts >= bbox_min) & (pts <= bbox_max), axis=1)

    if return_mask:
        return pts[mask], mask
    return pts[mask]

def depth_to_pointcloud(
    depth: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    T_wc: np.ndarray | None = None,
    mask_invalid: bool = True,
    return_indices: bool = False,
):
    """
    Convert a depth map to a point cloud.

    Args:
        depth: (H, W) array of depths in meters (float32/float64). Z=0 or NaN treated as invalid if mask_invalid.
        fx, fy, cx, cy: pinhole intrinsics.
        T_wc: (4, 4) homogeneous transform of the CAMERA in the WORLD frame (world_from_cam). If None, points are in camera frame.
        mask_invalid: if True, drop non-positive/NaN depths.
        return_indices: if True, also return the (v,u) pixel indices for each 3D point.

    Returns:
        pts: (N, 3) array of 3D points (camera frame if T_wc is None, else world frame).
        (optional) idx: (N, 2) int array of (v,u) pixel coords corresponding to pts.
    """
    assert depth.ndim == 2, "depth must be HxW"
    H, W = depth.shape

    # Build a grid of pixel coordinates (u along width/x, v along height/y)
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)

    z = depth.astype(np.float64)
    if mask_invalid:
        valid = np.isfinite(z) & (z > 0)
    else:
        valid = np.isfinite(z)

    if not np.any(valid):
        if return_indices:
            return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int32)
        return np.empty((0, 3), dtype=np.float64)

    uu = uu[valid]
    vv = vv[valid]
    z = z[valid]

    # Back-project to camera coordinates (x right, y down, z forward), pinhole model
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    pts_c = np.stack([x, y, z], axis=-1)  # (N,3)

    if T_wc is not None:
        assert T_wc.shape == (4, 4), "T_wc must be 4x4 homogeneous (world_from_cam)"
        # Homogenize and transform
        ones = np.ones((pts_c.shape[0], 1), dtype=pts_c.dtype)
        pts_c_h = np.concatenate([pts_c, ones], axis=1)  # (N,4)
        pts_w_h = pts_c_h @ T_wc.T                          # (N,4)
        pts = pts_w_h[:, :3]
    else:
        pts = pts_c

    if return_indices:
        idx = np.stack([vv.astype(np.int32), uu.astype(np.int32)], axis=-1)
        return pts.astype(np.float64), idx
    return pts.astype(np.float64)

# --------- Example usage ----------
if __name__ == "__main__":
    # depth: meters; K = [[fx,0,cx],[0,fy,cy],[0,0,1]]
    depth = np.random.uniform(0.5, 2.0, size=(480, 640)).astype(np.float32)
    fx, fy, cx, cy = 525.0, 525.0, 319.5, 239.5

    # Camera pose in world (example): T_wc
    T_wc = np.eye(4)
    T_wc[:3, 3] = [0.0, 0.0, 1.0]  # camera is 1m above world origin

    pts_world = depth_to_pointcloud(depth, fx, fy, cx, cy, T_wc=T_wc)
    print(pts_world.shape)  # (N, 3)
