from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Iterable
import numpy as np
import os
from urllib.parse import quote as _q, unquote as _uq

@dataclass
class ADDResultV2:
    """
    Hierarchical ADD results: K measurements, each with N particles.

    Fixed shapes:
      - add_measures:  (K, N)         float
      - gt_poses:      (K, N, 4, 4)   float
      - est_poses:     (K, N, 4, 4)   float
      - measurement_ids: (K,)         int  (per-measurement)
    """
    add_measures: np.ndarray
    gt_poses: np.ndarray
    est_poses: np.ndarray
    measurement_ids: np.ndarray

    # ---------- Constructors ----------
    @classmethod
    def empty(cls, num_particles: int, float_dtype=float, id_dtype=np.int64) -> "ADDResult":
        N = int(num_particles)
        if N <= 0:
            raise ValueError("num_particles must be > 0")
        return cls(
            add_measures=np.empty((0, N), dtype=float_dtype),
            gt_poses=np.empty((0, N, 4, 4), dtype=float_dtype),
            est_poses=np.empty((0, N, 4, 4), dtype=float_dtype),
            measurement_ids=np.empty((0,), dtype=id_dtype),
        )

    # ---------- Append one measurement (with N particles) ----------
    def add_measurement(
        self,
        measurement_id: int,
        add_measures: np.ndarray,   # (N,)
        gt_poses: np.ndarray,       # (N, 4, 4)
        est_poses: np.ndarray,      # (N, 4, 4)
    ) -> None:
        add_measures = np.asarray(add_measures)
        gt_poses = np.asarray(gt_poses)
        est_poses = np.asarray(est_poses)

        K_curr = self.add_measures.shape[0]
        N_expected = self.add_measures.shape[1] if K_curr > 0 else add_measures.shape[0]

        # Shape checks
        if add_measures.shape != (N_expected,):
            raise ValueError(f"add_measures shape {add_measures.shape} != expected {(N_expected,)}")
        if gt_poses.shape != (N_expected, 4, 4):
            raise ValueError(f"gt_poses shape {gt_poses.shape} != expected {(N_expected, 4, 4)}")
        if est_poses.shape != (N_expected, 4, 4):
            raise ValueError(f"est_poses shape {est_poses.shape} != expected {(N_expected, 4, 4)}")

        # Append along K
        self.add_measures = np.concatenate([self.add_measures, add_measures[None, ...]], axis=0)
        self.gt_poses = np.concatenate([self.gt_poses, gt_poses[None, ...]], axis=0)
        self.est_poses = np.concatenate([self.est_poses, est_poses[None, ...]], axis=0)
        self.measurement_ids = np.concatenate(
            [self.measurement_ids, np.array([measurement_id], dtype=self.measurement_ids.dtype)], axis=0
        )

    # ---------- Accessors ----------
    def get_measurement(self, k: int) -> Dict[str, Any]:
        """Return dict for measurement k with arrays of shape (N, ...)."""
        return {
            "measurement_id": int(self.measurement_ids[k]),
            "add_measures": self.add_measures[k],  # (N,)
            "gt_poses": self.gt_poses[k],          # (N, 4, 4)
            "est_poses": self.est_poses[k],        # (N, 4, 4)
        }

    # ---------- I/O ----------
    def save(self, path: str) -> None:
        """Compressed, pickle-free save."""
        np.savez_compressed(
            path,
            add_measures=self.add_measures,
            gt_poses=self.gt_poses,
            est_poses=self.est_poses,
            measurement_ids=self.measurement_ids,
        )

    @classmethod
    def load(cls, path: str) -> "ADDResult":
        """Pickle-free load (arrays must be numeric/fixed-shape)."""
        data = np.load(path, allow_pickle=False)
        return cls(
            add_measures=data["add_measures"],
            gt_poses=data["gt_poses"],
            est_poses=data["est_poses"],
            measurement_ids=data["measurement_ids"],
        )

    # ---------- Convenience ----------
    def __len__(self) -> int:
        """Total number of particles = K * N."""
        return int(self.add_measures.shape[0] * self.add_measures.shape[1])

    @property
    def K(self) -> int:
        """Number of measurements."""
        return self.add_measures.shape[0]

    @property
    def N(self) -> int:
        """Particles per measurement (fixed)."""
        return self.add_measures.shape[1] if self.K > 0 else 0

    # ---------- Summary ----------
    def summary(self) -> str:
        if self.K == 0:
            return "ADDResult (hierarchical): [empty]"
        add = self.add_measures
        add_min = float(add.min()) if add.size else None
        add_max = float(add.max()) if add.size else None
        add_mean = float(add.mean()) if add.size else None
        mid_min, mid_max = (int(self.measurement_ids.min()), int(self.measurement_ids.max())) if self.measurement_ids.size else (None, None)
        return (
            "ADDResult (hierarchical):\n"
            f"  Measurements (K): {self.K}\n"
            f"  Particles per measurement (N): {self.N}\n"
            f"  Pose shape: (4, 4)\n"
            f"  ADD stats: min={add_min}, max={add_max}, mean={add_mean}\n"
            f"  Measurement ID range: ({mid_min}, {mid_max})"
        )


@dataclass
class VisionInferenceResultV3:
    """
    Hierarchical result: K measurements, each with N particles.

    Fixed shapes:
      - poses:                 (K, N, 4, 4)       float
      - times:                 (K, N)             float (seconds)
      - unnormalised_log_pdfs: (K, N)             float
      - images:                (K, N, H, W, C)    (e.g., uint8 RGB or float32 depth)
      - pixelwise_score:       (K, N, H, W, C)    (e.g., uint8 RGB or float32 depth)
      - measurement_ids:       (K,)               int
    """
    poses: np.ndarray
    times: np.ndarray
    unnormalised_log_pdfs: np.ndarray
    images: np.ndarray
    pixelwise_score: np.ndarray
    measurement_ids: np.ndarray

    # ---------- Constructors ----------
    @classmethod
    def empty(
        cls,
        num_particles: int, #make optional
        image_shape: Tuple[int, int, int],
        image_dtype=np.uint8,
        pixelwise_dtype=np.uint8,
        float_dtype=float,
        id_dtype=np.int64,
    ) -> "VisionInferenceResult":
        H, W, C = image_shape
        N = int(num_particles)
        if N <= 0:
            raise ValueError("num_particles must be > 0")

        return cls(
            poses=np.empty((0, N, 4, 4), dtype=float_dtype),
            times=np.empty((0, N), dtype=float_dtype),
            unnormalised_log_pdfs=np.empty((0, N), dtype=float_dtype),
            images=np.empty((0, N, H, W, C), dtype=image_dtype),
            pixelwise_score=np.empty((0, N, H, W, C), dtype=pixelwise_dtype),
            measurement_ids=np.empty((0,), dtype=id_dtype),
        )

    # ---------- Append one measurement (with N particles) ----------
    def add_measurement(
        self,
        measurement_id: int,
        poses: np.ndarray,                 # (N, 4, 4)
        times: np.ndarray,                 # (N,)
        images: np.ndarray,                # (N, H, W, C)
        pixelwise_score: np.ndarray,       # (N, H, W, C)
        unnormalised_log_pdfs: np.ndarray  # (N,)
    ) -> None:
        poses = np.asarray(poses)
        times = np.asarray(times)
        images = np.asarray(images)
        pixelwise_score = np.asarray(pixelwise_score)
        unnormalised_log_pdfs = np.asarray(unnormalised_log_pdfs)

        #K_curr = self.poses.shape[0]
        N_expected = self.poses.shape[1]
        img_expected = self.images.shape[2:] 

        # Shape checks
        if poses.shape != (N_expected, 4, 4):
            raise ValueError(f"poses shape {poses.shape} != expected {(N_expected, 4, 4)}")
        if times.shape != (N_expected,):
            raise ValueError(f"times shape {times.shape} != expected {(N_expected,)}")
        if unnormalised_log_pdfs.shape != (N_expected,):
            raise ValueError(f"unnormalised_log_pdfs shape {unnormalised_log_pdfs.shape} != expected {(N_expected,)}")
        if images.shape != (N_expected, *img_expected): #if K_curr > 0 else images.ndim == 4:
            # if K_curr == 0:
            #     raise ValueError("images must be (N, H, W, C) on first insert")
            raise ValueError(f"images shape {images.shape} != expected {(N_expected, *img_expected)}")

        # Append along K (measurement) axis
        self.poses = np.concatenate([self.poses, poses[None, ...]], axis=0)
        self.times = np.concatenate([self.times, times[None, ...]], axis=0)
        self.unnormalised_log_pdfs = np.concatenate(
            [self.unnormalised_log_pdfs, unnormalised_log_pdfs[None, ...]], axis=0
        )
        self.images = np.concatenate([self.images, images[None, ...]], axis=0)
        self.pixelwise_score = np.concatenate([self.pixelwise_score,
                                              pixelwise_score[None, ...]],
                                              axis=0)
        self.measurement_ids = np.concatenate(
            [self.measurement_ids, np.array([measurement_id], dtype=self.measurement_ids.dtype)], axis=0
        )

    # ---------- Accessors ----------
    def get_measurement(self, k: int) -> Dict[str, Any]:
        """Return dict for measurement k with arrays of shape (N, ...)."""
        return {
            "measurement_id": int(self.measurement_ids[k]),
            "poses": self.poses[k],                   # (N, 4, 4)
            "times": self.times[k],                   # (N,)
            "images": self.images[k],                 # (N, H, W, C)
            "pixelwise_score": self.pixelwise_score[k], # (N, H, W, C)
            "unnormalised_log_pdfs": self.unnormalised_log_pdfs[k],  # (N,)
        }

    # ---------- I/O ----------
    def save(self, path: str) -> None:
        """Compressed, pickle-free save."""
        np.savez_compressed(
            path,
            poses=self.poses,
            times=self.times,
            unnormalised_log_pdfs=self.unnormalised_log_pdfs,
            images=self.images,
            pixelwise_score=self.pixelwise_score,
            measurement_ids=self.measurement_ids,
        )

    @classmethod
    def load(cls, path: str) -> "VisionInferenceResult":
        """Pickle-free load (arrays must be numeric/fixed-shape)."""
        data = np.load(path, allow_pickle=False)
        return cls(
            poses=data["poses"],
            times=data["times"],
            unnormalised_log_pdfs=data["unnormalised_log_pdfs"],
            images=data["images"],
            pixelwise_score=data["pixelwise_score"],
            measurement_ids=data["measurement_ids"],
        )

    # ---------- Convenience ----------
    def __len__(self) -> int:
        """Total number of particles = K * N."""
        return int(self.poses.shape[0] * self.poses.shape[1])

    @property
    def K(self) -> int:
        """Number of measurements."""
        return self.poses.shape[0]

    @property
    def N(self) -> int:
        """Particles per measurement (fixed)."""
        return self.poses.shape[1] if self.K > 0 else 0

    def summary(self) -> str:
        if self.K == 0:
            return "VisionInferenceResult (hierarchical): [empty]"
        img_shape = self.images.shape[2:]
        img_dtype = self.images.dtype
        t_min, t_max = (self.times.min(), self.times.max()) if self.times.size else (None, None)
        lp_min, lp_max = (self.unnormalised_log_pdfs.min(), self.unnormalised_log_pdfs.max()) if self.unnormalised_log_pdfs.size else (None, None)
        mid_min, mid_max = (self.measurement_ids.min(), self.measurement_ids.max()) if self.measurement_ids.size else (None, None)
        return (
            "VisionInferenceResult (hierarchical):\n"
            f"  Measurements (K): {self.K}\n"
            f"  Particles per measurement (N): {self.N}\n"
            f"  Pose shape: (4, 4)\n"
            f"  Image shape: {img_shape}, dtype={img_dtype}\n"
            f"  Time range: ({t_min}, {t_max})\n"
            f"  Measurement ID range: ({mid_min}, {mid_max})\n"
            f"  Unnormalised log-pdf range: ({lp_min}, {lp_max})"
        )


@dataclass
class ForceTorqueInferenceResultV2:
    """
    Hierarchical result: K measurements, each with N particles.

    Fixed shapes:
      - poses:                 (K, N, 4, 4)       float
      - times:                 (K, N)             float (seconds)
      - ft_trajectories:       (K, N, T, 6)       float (Fx,Fy,Fz,Tx,Ty,Tz)
      - unnormalised_log_pdfs: (K, N)             float
      - measurement_ids:       (K,)             int  (per-particle, kept to match prior API)

    Notes:
      * N (particles per measurement) is fixed across the dataset.
      * T (trajectory length) is fixed across the dataset.
      * Use add_measurement(...) to append one full measurement (N particles).
      * You can pass a scalar measurement_id or an (N,) array; scalars broadcast.
    """
    poses: np.ndarray
    times: np.ndarray
    ft_trajectories: np.ndarray
    unnormalised_log_pdfs: np.ndarray
    measurement_ids: np.ndarray

    # ---------- Constructors ----------
    @classmethod
    def empty(
        cls,
        num_particles: int,
        traj_len: int,
        float_dtype=float,
        id_dtype=np.int64,
    ) -> "ForceTorqueInferenceResult":
        N = int(num_particles)
        T = int(traj_len)
        if N <= 0:
            raise ValueError("num_particles must be > 0")
        if T <= 0:
            raise ValueError("traj_len must be > 0")

        return cls(
            poses=np.empty((0, N, 4, 4), dtype=float_dtype),
            times=np.empty((0, N), dtype=float_dtype),
            ft_trajectories=np.empty((0, N, T, 6), dtype=float_dtype),
            unnormalised_log_pdfs=np.empty((0, N), dtype=float_dtype),
            measurement_ids=np.empty((0,), dtype=id_dtype),
        )

    # ---------- Append one measurement (with N particles) ----------
    def add_measurement(
        self,
        poses: np.ndarray,                   # (N, 4, 4)
        times: np.ndarray,                   # (N,)
        ft_trajectories: np.ndarray,         # (N, T, 6)
        unnormalised_log_pdfs: np.ndarray,   # (N,)
        measurement_id: int
    ) -> None:
        poses = np.asarray(poses)
        times = np.asarray(times)
        ft_trajectories = np.asarray(ft_trajectories)
        unnormalised_log_pdfs = np.asarray(unnormalised_log_pdfs)

        K_curr = self.poses.shape[0]
        N_expected = self.poses.shape[1] if K_curr > 0 else poses.shape[0]
        T_expected = self.ft_trajectories.shape[2] if K_curr > 0 else ft_trajectories.shape[1]

        # Validate shapes
        if poses.shape != (N_expected, 4, 4):
            raise ValueError(f"poses shape {poses.shape} != expected {(N_expected, 4, 4)}")
        if times.shape != (N_expected,):
            raise ValueError(f"times shape {times.shape} != expected {(N_expected,)}")
        if ft_trajectories.shape != (N_expected, T_expected, 6):
            raise ValueError(f"ft_trajectories shape {ft_trajectories.shape} != expected {(N_expected, T_expected, 6)}")
        if unnormalised_log_pdfs.shape != (N_expected,):
            raise ValueError(f"unnormalised_log_pdfs shape {unnormalised_log_pdfs.shape} != expected {(N_expected,)}")

        # Measurement IDs: accept scalar or (N,)
        # if np.isscalar(measurement_id):
        #     mids = np.full((N_expected,), int(measurement_id), dtype=self.measurement_ids.dtype)
        # else:
        #     mids = np.asarray(measurement_id)
        #     if mids.shape != (N_expected,):
        #         raise ValueError(f"measurement_id shape {mids.shape} != expected {(N_expected,)}")
        #     if mids.dtype != self.measurement_ids.dtype:
        #         mids = mids.astype(self.measurement_ids.dtype, copy=False)

        # Append along K
        self.poses = np.concatenate([self.poses, poses[None, ...]], axis=0)
        self.times = np.concatenate([self.times, times[None, ...]], axis=0)
        self.ft_trajectories = np.concatenate([self.ft_trajectories, ft_trajectories[None, ...]], axis=0)
        self.unnormalised_log_pdfs = np.concatenate([self.unnormalised_log_pdfs, unnormalised_log_pdfs[None, ...]], axis=0)
        # self.measurement_ids = np.concatenate([self.measurement_ids, mids[None, ...]], axis=0)
        self.measurement_ids = np.concatenate(
            [self.measurement_ids, np.array([measurement_id], dtype=self.measurement_ids.dtype)], axis=0
        )

    # ---------- Accessors ----------
    def get_measurement(self, k: int) -> Dict[str, Any]:
        """Return dict for measurement k with arrays of shape (N, ...)."""
        return {
            "poses": self.poses[k],                       # (N, 4, 4)
            "times": self.times[k],                       # (N,)
            "ft_trajectories": self.ft_trajectories[k],   # (N, T, 6)
            "unnormalised_log_pdfs": self.unnormalised_log_pdfs[k],  # (N,)
            "measurement_ids": self.measurement_ids[k],   # (N,)
        }

    # ---------- I/O ----------
    def save(self, path: str) -> None:
        """Compressed, pickle-free save."""
        np.savez_compressed(
            path,
            poses=self.poses,
            times=self.times,
            ft_trajectories=self.ft_trajectories,
            unnormalised_log_pdfs=self.unnormalised_log_pdfs,
            measurement_ids=self.measurement_ids,
        )

    @classmethod
    def load(cls, path: str) -> "ForceTorqueInferenceResult":
        """Pickle-free load (arrays must be numeric/fixed-shape)."""
        data = np.load(path, allow_pickle=False)
        return cls(
            poses=data["poses"],
            times=data["times"],
            ft_trajectories=data["ft_trajectories"],
            unnormalised_log_pdfs=data["unnormalised_log_pdfs"],
            measurement_ids=data["measurement_ids"],
        )

    # ---------- Convenience ----------
    def __len__(self) -> int:
        """Total number of particles = K * N."""
        return int(self.poses.shape[0] * self.poses.shape[1])

    @property
    def K(self) -> int:
        """Number of measurements."""
        return self.poses.shape[0]

    @property
    def N(self) -> int:
        """Particles per measurement (fixed)."""
        return self.poses.shape[1] if self.K > 0 else 0

    @property
    def T(self) -> int:
        """Trajectory length (fixed)."""
        return self.ft_trajectories.shape[2] if self.K > 0 else 0

    def summary(self) -> str:
        if self.K == 0:
            return "ForceTorqueInferenceResult (hierarchical): [empty]"
        t_min, t_max = (self.times.min(), self.times.max()) if self.times.size else (None, None)
        lp_min, lp_max = (self.unnormalised_log_pdfs.min(), self.unnormalised_log_pdfs.max()) if self.unnormalised_log_pdfs.size else (None, None)
        mid_min, mid_max = (self.measurement_ids.min(), self.measurement_ids.max()) if self.measurement_ids.size else (None, None)
        return (
            "ForceTorqueInferenceResult (hierarchical):\n"
            f"  Measurements (K): {self.K}\n"
            f"  Particles per measurement (N): {self.N}\n"
            f"  Trajectory length (T): {self.T}\n"
            f"  Pose shape: {self.poses.shape}\n"
            f"  F/T shape : {self.ft_trajectories.shape}\n"
            f"  log-pdf shape : {self.unnormalised_log_pdfs.shape}\n"
            f"  Time range: ({t_min}, {t_max})\n"
            f"  Unnormalised log-pdf range: ({lp_min}, {lp_max})\n"
            f"  Measurement ID range (per-particle): ({mid_min}, {mid_max})"
        )

