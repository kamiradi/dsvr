#!/usr/bin/env python3
"""Experimental script to visualize robot URDF with joint states from dataset."""

import argparse
from pathlib import Path
import numpy as np
import rerun as rr
from dsvr.datasets.robot_data import RoboticsDatasetV2

try:
    import yourdfpy
    HAS_YOURDFPY = True
except ImportError:
    HAS_YOURDFPY = False
    print("Warning: yourdfpy not installed. Install with: pip install yourdfpy")

ap = argparse.ArgumentParser()
ap.add_argument("data_npz", help="Path to dataset .npz")
ap.add_argument("--urdf", type=str, required=True, help="Path to URDF file")
args = ap.parse_args()

print("Initializing rerun...")
rr.init("robot_viz", spawn=True)

# Log coordinate frame and test geometry
print("Logging test geometry...")
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
rr.log("world/origin", rr.Points3D([[0, 0, 0]], colors=[[255, 0, 0]], radii=[0.1]), static=True)
rr.log("world/x_axis", rr.Arrows3D(origins=[[0,0,0]], vectors=[[1,0,0]], colors=[[255,0,0]]), static=True)
rr.log("world/y_axis", rr.Arrows3D(origins=[[0,0,0]], vectors=[[0,1,0]], colors=[[0,255,0]]), static=True)
rr.log("world/z_axis", rr.Arrows3D(origins=[[0,0,0]], vectors=[[0,0,1]], colors=[[0,0,255]]), static=True)
print("You should see a red point and RGB axes at origin")

# Load dataset
print(f"Loading dataset from {args.data_npz}...")
ds = RoboticsDatasetV2.load(args.data_npz)

if ds.joint_states is None or ds.joint_ts is None:
    print("ERROR: No joint states in dataset")
    exit(1)

print(f"Loaded {len(ds.joint_states)} joint states, shape: {ds.joint_states.shape}")
print(f"First joint state: {ds.joint_states[0]}")

if not HAS_YOURDFPY:
    print("Cannot animate robot without yourdfpy")
    exit(1)

# Load URDF
print(f"Loading URDF from {args.urdf}...")
robot = yourdfpy.URDF.load(args.urdf)

# Get actuated joint names
joint_names = list(robot.actuated_joint_names)
print(f"URDF actuated joints ({len(joint_names)}): {joint_names}")

DoF = ds.joint_states.shape[1]
if len(joint_names) != DoF:
    print(f"Warning: URDF has {len(joint_names)} joints but dataset has {DoF} DoF")
    joint_names = joint_names[:DoF]

# Log link names
link_names = [link.name for link in robot.robot.links]
print(f"URDF links ({len(link_names)}): {link_names}")

def mat_to_transform3d(T: np.ndarray) -> rr.Transform3D:
    """Convert 4x4 matrix to rr.Transform3D."""
    return rr.Transform3D(mat3x3=T[:3, :3], translation=T[:3, 3])


# Prepare mesh data (load once, transform each frame)
print("Loading meshes...")
mesh_data = {}  # geom_name -> (vertices, faces, colors)
scene = robot.scene
if scene is not None:
    print(f"  Scene has {len(scene.geometry)} geometries")
    for geom_name, geom in scene.geometry.items():
        try:
            if hasattr(geom, 'vertices') and hasattr(geom, 'faces'):
                vertices = np.array(geom.vertices, dtype=np.float32)
                faces = np.array(geom.faces, dtype=np.uint32)

                # Get vertex colors if available
                vertex_colors = None
                if hasattr(geom, 'visual') and hasattr(geom.visual, 'vertex_colors'):
                    vertex_colors = np.array(geom.visual.vertex_colors)

                mesh_data[geom_name] = (vertices, faces, vertex_colors)
                print(f"  Loaded mesh: {geom_name} ({len(vertices)} verts)")
        except Exception as e:
            print(f"  Could not load mesh {geom_name}: {e}")
else:
    print("  No scene geometry found")

# Animate through joint states (subset to avoid crashing rerun)
print("Animating robot...")
start_idx, end_idx, step = 700, 901, 4
for idx in range(start_idx, min(end_idx, len(ds.joint_states)), step):
    q, t = ds.joint_states[idx], ds.joint_ts[idx]
    rr.set_time_seconds("time", float(t))

    # Update joint configuration
    cfg = {name: float(val) for name, val in zip(joint_names, q)}

    try:
        # Update robot configuration - this updates scene graph transforms
        robot.update_cfg(cfg)

        # Log each geometry with its current world transform
        for geom_name, (vertices, faces, vertex_colors) in mesh_data.items():
            try:
                # Get world transform for this geometry from scene graph
                T = robot.scene.graph.get(geom_name)[0]

                # Transform vertices to world frame
                verts_homogeneous = np.hstack([vertices, np.ones((len(vertices), 1))])
                verts_world = (T @ verts_homogeneous.T).T[:, :3]

                rr.log(
                    f"world/robot/{geom_name}",
                    rr.Mesh3D(
                        vertex_positions=verts_world.astype(np.float32),
                        triangle_indices=faces,
                        vertex_colors=vertex_colors,
                    ),
                )
            except Exception as e:
                if idx == 0:
                    print(f"  Could not transform {geom_name}: {e}")

    except Exception as e:
        if idx == 0:
            print(f"FK error: {e}")

    if idx % 100 == 0:
        print(f"  Frame {idx} (range {start_idx}-{end_idx}, step {step}), t={t:.3f}")

print("Done! Check the rerun viewer.")
print("Tips: Press 'F' to focus on geometry, use timeline to scrub through animation")
