import rerun as rr
import numpy as np

_MESHES = "/home/aditya/Documents/workspace/assembly_description/urdf/meshes"

SPACING = 0.15  # metres between hole centres along X

HOLES = [
    ("arch",            f"{_MESHES}/arch_hole.obj",           [255,  80,  80, 255]),  # red
    ("ellipse",         f"{_MESHES}/ellipse_hole.obj",         [ 80, 200,  80, 255]),  # green
    ("ellipse_teeth",   f"{_MESHES}/ellipse_hole_teeth.obj",   [ 80, 120, 255, 255]),  # blue
    ("rectangle",       f"{_MESHES}/rectangular_hole.obj",     [255, 200,  50, 255]),  # yellow
    ("rectangle_teeth", f"{_MESHES}/rectangle_hole_teeth.obj", [200,  80, 255, 255]),  # purple
]

rr.init("hole_meshes", spawn=True)
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

rr.log(
    "/world/origin",
    rr.Arrows3D(
        vectors=[[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]],
        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    ),
    static=True,
)

n = len(HOLES)
offsets = (np.arange(n) - (n - 1) / 2.0) * SPACING

for i, (name, obj_path, colour) in enumerate(HOLES):
    ent = f"/world/{name}"
    rr.log(ent, rr.Transform3D(translation=[offsets[i], 0.0, 0.0]), static=True)
    rr.log(f"{ent}/geom", rr.Asset3D(path=obj_path, albedo_factor=colour), static=True)
