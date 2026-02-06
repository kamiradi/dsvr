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
    def empty(
        cls, num_particles: int, float_dtype=float, id_dtype=np.int64
    ) -> "ADDResult":
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
        add_measures: np.ndarray,  # (N,)
        gt_poses: np.ndarray,  # (N, 4, 4)
        est_poses: np.ndarray,  # (N, 4, 4)
    ) -> None:
        add_measures = np.asarray(add_measures)
        gt_poses = np.asarray(gt_poses)
        est_poses = np.asarray(est_poses)

        K_curr = self.add_measures.shape[0]
        N_expected = self.add_measures.shape[1] if K_curr > 0 else add_measures.shape[0]

        # Shape checks
        if add_measures.shape != (N_expected,):
            raise ValueError(
                f"add_measures shape {add_measures.shape} != expected {(N_expected,)}"
            )
        if gt_poses.shape != (N_expected, 4, 4):
            raise ValueError(
                f"gt_poses shape {gt_poses.shape} != expected {(N_expected, 4, 4)}"
            )
        if est_poses.shape != (N_expected, 4, 4):
            raise ValueError(
                f"est_poses shape {est_poses.shape} != expected {(N_expected, 4, 4)}"
            )

        # Append along K
        self.add_measures = np.concatenate(
            [self.add_measures, add_measures[None, ...]], axis=0
        )
        self.gt_poses = np.concatenate([self.gt_poses, gt_poses[None, ...]], axis=0)
        self.est_poses = np.concatenate([self.est_poses, est_poses[None, ...]], axis=0)
        self.measurement_ids = np.concatenate(
            [
                self.measurement_ids,
                np.array([measurement_id], dtype=self.measurement_ids.dtype),
            ],
            axis=0,
        )

    # ---------- Accessors ----------
    def get_measurement(self, k: int) -> Dict[str, Any]:
        """Return dict for measurement k with arrays of shape (N, ...)."""
        return {
            "measurement_id": int(self.measurement_ids[k]),
            "add_measures": self.add_measures[k],  # (N,)
            "gt_poses": self.gt_poses[k],  # (N, 4, 4)
            "est_poses": self.est_poses[k],  # (N, 4, 4)
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
        mid_min, mid_max = (
            (int(self.measurement_ids.min()), int(self.measurement_ids.max()))
            if self.measurement_ids.size
            else (None, None)
        )
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
        num_particles: int,  # make optional
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
        poses: np.ndarray,  # (N, 4, 4)
        times: np.ndarray,  # (N,)
        images: np.ndarray,  # (N, H, W, C)
        pixelwise_score: np.ndarray,  # (N, H, W, C)
        unnormalised_log_pdfs: np.ndarray,  # (N,)
    ) -> None:
        poses = np.asarray(poses)
        times = np.asarray(times)
        images = np.asarray(images)
        pixelwise_score = np.asarray(pixelwise_score)
        unnormalised_log_pdfs = np.asarray(unnormalised_log_pdfs)

        # K_curr = self.poses.shape[0]
        N_expected = self.poses.shape[1]
        img_expected = self.images.shape[2:]

        # Shape checks
        if poses.shape != (N_expected, 4, 4):
            raise ValueError(
                f"poses shape {poses.shape} != expected {(N_expected, 4, 4)}"
            )
        if times.shape != (N_expected,):
            raise ValueError(f"times shape {times.shape} != expected {(N_expected,)}")
        if unnormalised_log_pdfs.shape != (N_expected,):
            raise ValueError(
                f"unnormalised_log_pdfs shape {unnormalised_log_pdfs.shape} != expected {(N_expected,)}"
            )
        if images.shape != (
            N_expected,
            *img_expected,
        ):  # if K_curr > 0 else images.ndim == 4:
            # if K_curr == 0:
            #     raise ValueError("images must be (N, H, W, C) on first insert")
            raise ValueError(
                f"images shape {images.shape} != expected {(N_expected, *img_expected)}"
            )

        # Append along K (measurement) axis
        self.poses = np.concatenate([self.poses, poses[None, ...]], axis=0)
        self.times = np.concatenate([self.times, times[None, ...]], axis=0)
        self.unnormalised_log_pdfs = np.concatenate(
            [self.unnormalised_log_pdfs, unnormalised_log_pdfs[None, ...]], axis=0
        )
        self.images = np.concatenate([self.images, images[None, ...]], axis=0)
        self.pixelwise_score = np.concatenate(
            [self.pixelwise_score, pixelwise_score[None, ...]], axis=0
        )
        self.measurement_ids = np.concatenate(
            [
                self.measurement_ids,
                np.array([measurement_id], dtype=self.measurement_ids.dtype),
            ],
            axis=0,
        )

    # ---------- Accessors ----------
    def get_measurement(self, k: int) -> Dict[str, Any]:
        """Return dict for measurement k with arrays of shape (N, ...)."""
        return {
            "measurement_id": int(self.measurement_ids[k]),
            "poses": self.poses[k],  # (N, 4, 4)
            "times": self.times[k],  # (N,)
            "images": self.images[k],  # (N, H, W, C)
            "pixelwise_score": self.pixelwise_score[k],  # (N, H, W, C)
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
        t_min, t_max = (
            (self.times.min(), self.times.max()) if self.times.size else (None, None)
        )
        lp_min, lp_max = (
            (self.unnormalised_log_pdfs.min(), self.unnormalised_log_pdfs.max())
            if self.unnormalised_log_pdfs.size
            else (None, None)
        )
        mid_min, mid_max = (
            (self.measurement_ids.min(), self.measurement_ids.max())
            if self.measurement_ids.size
            else (None, None)
        )
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
        poses: np.ndarray,  # (N, 4, 4)
        times: np.ndarray,  # (N,)
        ft_trajectories: np.ndarray,  # (N, T, 6)
        unnormalised_log_pdfs: np.ndarray,  # (N,)
        measurement_id: int,
    ) -> None:
        poses = np.asarray(poses)
        times = np.asarray(times)
        ft_trajectories = np.asarray(ft_trajectories)
        unnormalised_log_pdfs = np.asarray(unnormalised_log_pdfs)

        K_curr = self.poses.shape[0]
        N_expected = self.poses.shape[1] if K_curr > 0 else poses.shape[0]
        T_expected = (
            self.ft_trajectories.shape[2] if K_curr > 0 else ft_trajectories.shape[1]
        )

        # Validate shapes
        if poses.shape != (N_expected, 4, 4):
            raise ValueError(
                f"poses shape {poses.shape} != expected {(N_expected, 4, 4)}"
            )
        if times.shape != (N_expected,):
            raise ValueError(f"times shape {times.shape} != expected {(N_expected,)}")
        if ft_trajectories.shape != (N_expected, T_expected, 6):
            raise ValueError(
                f"ft_trajectories shape {ft_trajectories.shape} != expected {(N_expected, T_expected, 6)}"
            )
        if unnormalised_log_pdfs.shape != (N_expected,):
            raise ValueError(
                f"unnormalised_log_pdfs shape {unnormalised_log_pdfs.shape} != expected {(N_expected,)}"
            )

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
        self.ft_trajectories = np.concatenate(
            [self.ft_trajectories, ft_trajectories[None, ...]], axis=0
        )
        self.unnormalised_log_pdfs = np.concatenate(
            [self.unnormalised_log_pdfs, unnormalised_log_pdfs[None, ...]], axis=0
        )
        # self.measurement_ids = np.concatenate([self.measurement_ids, mids[None, ...]], axis=0)
        self.measurement_ids = np.concatenate(
            [
                self.measurement_ids,
                np.array([measurement_id], dtype=self.measurement_ids.dtype),
            ],
            axis=0,
        )

    # ---------- Accessors ----------
    def get_measurement(self, k: int) -> Dict[str, Any]:
        """Return dict for measurement k with arrays of shape (N, ...)."""
        return {
            "poses": self.poses[k],  # (N, 4, 4)
            "times": self.times[k],  # (N,)
            "ft_trajectories": self.ft_trajectories[k],  # (N, T, 6)
            "unnormalised_log_pdfs": self.unnormalised_log_pdfs[k],  # (N,)
            "measurement_ids": self.measurement_ids[k],  # (N,)
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
        t_min, t_max = (
            (self.times.min(), self.times.max()) if self.times.size else (None, None)
        )
        lp_min, lp_max = (
            (self.unnormalised_log_pdfs.min(), self.unnormalised_log_pdfs.max())
            if self.unnormalised_log_pdfs.size
            else (None, None)
        )
        mid_min, mid_max = (
            (self.measurement_ids.min(), self.measurement_ids.max())
            if self.measurement_ids.size
            else (None, None)
        )
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


@dataclass
class ForceTorqueInferenceResultV3:
    """
    Hierarchical result: K measurements, each with N particles.

    Fixed shapes:
      - poses:                 (K, N, 4, 4)       float
      - times:                 (K, N)             float (seconds)
      - ft_trajectories:       (K, N, T, 6)       float (Fx,Fy,Fz,Tx,Ty,Tz)
      - gt_ft_trajectories:    (K, N, T, 6)       float (ground truth Fx,Fy,Fz,Tx,Ty,Tz)
      - unnormalised_log_pdfs: (K, N)             float
      - measurement_ids:       (K,)               int  (per-particle, kept to match prior API)
      - probe_poses:           (K, 4, 4)          float (ground truth pose for each measurement)

    Notes:
      * N (particles per measurement) is fixed across the dataset.
      * T (trajectory length) is fixed across the dataset.
      * Use add_measurement(...) to append one full measurement (N particles).
      * You can pass a scalar measurement_id or an (N,) array; scalars broadcast.
    """

    poses: np.ndarray
    times: np.ndarray
    ft_trajectories: np.ndarray
    gt_ft_trajectories: np.ndarray
    unnormalised_log_pdfs: np.ndarray
    measurement_ids: np.ndarray
    probe_poses: np.ndarray

    # ---------- Constructors ----------
    @classmethod
    def empty(
        cls,
        num_particles: int,
        traj_len: int,
        float_dtype=float,
        id_dtype=np.int64,
    ) -> "ForceTorqueInferenceResultV3":
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
            gt_ft_trajectories=np.empty((0, N, T, 6), dtype=float_dtype),
            unnormalised_log_pdfs=np.empty((0, N), dtype=float_dtype),
            measurement_ids=np.empty((0,), dtype=id_dtype),
            probe_poses=np.empty((0, 4, 4), dtype=float_dtype),
        )

    # ---------- Append one measurement (with N particles) ----------
    def add_measurement(
        self,
        poses: np.ndarray,  # (N, 4, 4)
        times: np.ndarray,  # (N,)
        ft_trajectories: np.ndarray,  # (N, T, 6)
        gt_ft_trajectories: np.ndarray,  # (N, T, 6)
        unnormalised_log_pdfs: np.ndarray,  # (N,)
        measurement_id: int,
        probe_pose: np.ndarray,  # (4, 4)
    ) -> None:
        poses = np.asarray(poses)
        times = np.asarray(times)
        ft_trajectories = np.asarray(ft_trajectories)
        gt_ft_trajectories = np.asarray(gt_ft_trajectories)
        unnormalised_log_pdfs = np.asarray(unnormalised_log_pdfs)
        probe_pose = np.asarray(probe_pose)

        K_curr = self.poses.shape[0]
        N_expected = self.poses.shape[1] if K_curr > 0 else poses.shape[0]
        T_expected = (
            self.ft_trajectories.shape[2] if K_curr > 0 else ft_trajectories.shape[1]
        )

        # Validate shapes
        if poses.shape != (N_expected, 4, 4):
            raise ValueError(
                f"poses shape {poses.shape} != expected {(N_expected, 4, 4)}"
            )
        if times.shape != (N_expected,):
            raise ValueError(f"times shape {times.shape} != expected {(N_expected,)}")
        if ft_trajectories.shape != (N_expected, T_expected, 6):
            raise ValueError(
                f"ft_trajectories shape {ft_trajectories.shape} != expected {(N_expected, T_expected, 6)}"
            )
        if gt_ft_trajectories.shape != (N_expected, T_expected, 6):
            raise ValueError(
                f"gt_ft_trajectories shape {gt_ft_trajectories.shape} != expected {(N_expected, T_expected, 6)}"
            )
        if unnormalised_log_pdfs.shape != (N_expected,):
            raise ValueError(
                f"unnormalised_log_pdfs shape {unnormalised_log_pdfs.shape} != expected {(N_expected,)}"
            )
        if probe_pose.shape != (4, 4):
            raise ValueError(f"probe_pose shape {probe_pose.shape} != expected (4, 4)")

        # Append along K
        self.poses = np.concatenate([self.poses, poses[None, ...]], axis=0)
        self.times = np.concatenate([self.times, times[None, ...]], axis=0)
        self.ft_trajectories = np.concatenate(
            [self.ft_trajectories, ft_trajectories[None, ...]], axis=0
        )
        self.gt_ft_trajectories = np.concatenate(
            [self.gt_ft_trajectories, gt_ft_trajectories[None, ...]], axis=0
        )
        self.unnormalised_log_pdfs = np.concatenate(
            [self.unnormalised_log_pdfs, unnormalised_log_pdfs[None, ...]], axis=0
        )
        self.measurement_ids = np.concatenate(
            [
                self.measurement_ids,
                np.array([measurement_id], dtype=self.measurement_ids.dtype),
            ],
            axis=0,
        )
        self.probe_poses = np.concatenate(
            [self.probe_poses, probe_pose[None, ...]], axis=0
        )

    # ---------- Accessors ----------
    def get_measurement(self, k: int) -> Dict[str, Any]:
        """Return dict for measurement k with arrays of shape (N, ...)."""
        return {
            "poses": self.poses[k],  # (N, 4, 4)
            "times": self.times[k],  # (N,)
            "ft_trajectories": self.ft_trajectories[k],  # (N, T, 6)
            "gt_ft_trajectories": self.gt_ft_trajectories[k],  # (N, T, 6)
            "unnormalised_log_pdfs": self.unnormalised_log_pdfs[k],  # (N,)
            "measurement_ids": self.measurement_ids[k],  # (N,)
            "probe_pose": self.probe_poses[k],  # (4, 4)
        }

    # ---------- I/O ----------
    def save(self, path: str) -> None:
        """Compressed, pickle-free save."""
        np.savez_compressed(
            path,
            poses=self.poses,
            times=self.times,
            ft_trajectories=self.ft_trajectories,
            gt_ft_trajectories=self.gt_ft_trajectories,
            unnormalised_log_pdfs=self.unnormalised_log_pdfs,
            measurement_ids=self.measurement_ids,
            probe_poses=self.probe_poses,
        )

    @classmethod
    def load(cls, path: str) -> "ForceTorqueInferenceResultV3":
        """Pickle-free load (arrays must be numeric/fixed-shape)."""
        data = np.load(path, allow_pickle=False)
        return cls(
            poses=data["poses"],
            times=data["times"],
            ft_trajectories=data["ft_trajectories"],
            gt_ft_trajectories=data["gt_ft_trajectories"],
            unnormalised_log_pdfs=data["unnormalised_log_pdfs"],
            measurement_ids=data["measurement_ids"],
            probe_poses=data["probe_poses"],
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
            return "ForceTorqueInferenceResultV3 (hierarchical): [empty]"
        t_min, t_max = (
            (self.times.min(), self.times.max()) if self.times.size else (None, None)
        )
        lp_min, lp_max = (
            (self.unnormalised_log_pdfs.min(), self.unnormalised_log_pdfs.max())
            if self.unnormalised_log_pdfs.size
            else (None, None)
        )
        mid_min, mid_max = (
            (self.measurement_ids.min(), self.measurement_ids.max())
            if self.measurement_ids.size
            else (None, None)
        )
        return (
            "ForceTorqueInferenceResultV3 (hierarchical):\n"
            f"  Measurements (K): {self.K}\n"
            f"  Particles per measurement (N): {self.N}\n"
            f"  Trajectory length (T): {self.T}\n"
            f"  Pose shape: {self.poses.shape}\n"
            f"  F/T shape : {self.ft_trajectories.shape}\n"
            f"  GT F/T shape : {self.gt_ft_trajectories.shape}\n"
            f"  log-pdf shape : {self.unnormalised_log_pdfs.shape}\n"
            f"  Probe pose shape: {self.probe_poses.shape}\n"
            f"  Time range: ({t_min}, {t_max})\n"
            f"  Unnormalised log-pdf range: ({lp_min}, {lp_max})\n"
            f"  Measurement ID range (per-particle): ({mid_min}, {mid_max})"
        )


@dataclass
class ContactResidualResult:
    """
    Result class for contact location inference via residual computation.

    Stores the residuals computed from force/torque measurements against
    sampled points on a mesh surface.

    Fixed shapes:
      - mesh_points:          (P, 3)    float - sampled points on mesh
      - mesh_normals:         (P, 3)    float - normals at each point
      - residuals:            (T, P)    float - residual per timestep per point
      - ft_values:            (T, 6)    float - force/torque values (Fx,Fy,Fz,Tx,Ty,Tz)
      - times:                (T,)      float - timestamps
      - contact_mask:         (T,)      uint8 - binary mask for contact timesteps
      - min_residual_indices: (T,)      int   - index of min residual point per timestep
      - min_residuals:        (T,)      float - minimum residual value per timestep
      - gt_contact_locations: (T, 4, 4) float - ground truth contact locations as SE3
                                                (identity rotation + contact point translation)

    Notes:
      * P is the number of sampled points on the mesh (fixed).
      * T is the trajectory length (number of timesteps).
    """

    mesh_points: np.ndarray
    mesh_normals: np.ndarray
    residuals: np.ndarray
    ft_values: np.ndarray
    times: np.ndarray
    contact_mask: np.ndarray
    min_residual_indices: np.ndarray
    min_residuals: np.ndarray
    gt_contact_locations: np.ndarray

    # ---------- Constructors ----------
    @classmethod
    def empty(
        cls,
        num_points: int,
        float_dtype=np.float32,
    ) -> "ContactResidualResult":
        P = int(num_points)
        if P <= 0:
            raise ValueError("num_points must be > 0")

        return cls(
            mesh_points=np.empty((P, 3), dtype=float_dtype),
            mesh_normals=np.empty((P, 3), dtype=float_dtype),
            residuals=np.empty((0, P), dtype=float_dtype),
            ft_values=np.empty((0, 6), dtype=float_dtype),
            times=np.empty((0,), dtype=float_dtype),
            contact_mask=np.empty((0,), dtype=np.uint8),
            min_residual_indices=np.empty((0,), dtype=np.int64),
            min_residuals=np.empty((0,), dtype=float_dtype),
            gt_contact_locations=np.empty((0, 4, 4), dtype=float_dtype),
        )

    @classmethod
    def from_mesh(
        cls,
        mesh_points: np.ndarray,
        mesh_normals: np.ndarray,
        float_dtype=np.float32,
    ) -> "ContactResidualResult":
        """Create empty result with pre-loaded mesh points and normals."""
        mesh_points = np.asarray(mesh_points, dtype=float_dtype)
        mesh_normals = np.asarray(mesh_normals, dtype=float_dtype)
        P = mesh_points.shape[0]

        if mesh_points.shape != (P, 3):
            raise ValueError(
                f"mesh_points shape {mesh_points.shape} invalid, expected (P, 3)"
            )
        if mesh_normals.shape != (P, 3):
            raise ValueError(
                f"mesh_normals shape {mesh_normals.shape} != mesh_points shape"
            )

        return cls(
            mesh_points=mesh_points,
            mesh_normals=mesh_normals,
            residuals=np.empty((0, P), dtype=float_dtype),
            ft_values=np.empty((0, 6), dtype=float_dtype),
            times=np.empty((0,), dtype=float_dtype),
            contact_mask=np.empty((0,), dtype=np.uint8),
            min_residual_indices=np.empty((0,), dtype=np.int64),
            min_residuals=np.empty((0,), dtype=float_dtype),
            gt_contact_locations=np.empty((0, 4, 4), dtype=float_dtype),
        )

    # ---------- Append timesteps ----------
    def add_timestep(
        self,
        residuals: np.ndarray,  # (P,)
        ft_value: np.ndarray,  # (6,)
        time: float,
        contact: bool,
        gt_contact_location: np.ndarray = None,  # (4, 4) or (3,) for position only
    ) -> None:
        """Add a single timestep of residual data."""
        residuals = np.asarray(residuals, dtype=self.residuals.dtype)
        ft_value = np.asarray(ft_value, dtype=self.ft_values.dtype)

        P = self.mesh_points.shape[0]
        if residuals.shape != (P,):
            raise ValueError(f"residuals shape {residuals.shape} != expected ({P},)")
        if ft_value.shape != (6,):
            raise ValueError(f"ft_value shape {ft_value.shape} != expected (6,)")

        # Handle gt_contact_location - convert 3D point to SE3 if needed
        if gt_contact_location is None:
            gt_se3 = np.eye(4, dtype=self.gt_contact_locations.dtype)
        else:
            gt_contact_location = np.asarray(gt_contact_location)
            if gt_contact_location.shape == (3,):
                # Convert 3D point to SE3 with identity rotation
                gt_se3 = np.eye(4, dtype=self.gt_contact_locations.dtype)
                gt_se3[:3, 3] = gt_contact_location
            elif gt_contact_location.shape == (4, 4):
                gt_se3 = gt_contact_location.astype(self.gt_contact_locations.dtype)
            else:
                raise ValueError(
                    f"gt_contact_location shape {gt_contact_location.shape} != expected (3,) or (4, 4)"
                )

        # Compute min residual info
        min_idx = int(np.argmin(residuals))
        min_val = float(residuals[min_idx])

        self.residuals = np.concatenate([self.residuals, residuals[None, :]], axis=0)
        self.ft_values = np.concatenate([self.ft_values, ft_value[None, :]], axis=0)
        self.times = np.concatenate(
            [self.times, np.array([time], dtype=self.times.dtype)]
        )
        self.contact_mask = np.concatenate(
            [self.contact_mask, np.array([int(contact)], dtype=np.uint8)]
        )
        self.min_residual_indices = np.concatenate(
            [self.min_residual_indices, np.array([min_idx], dtype=np.int64)]
        )
        self.min_residuals = np.concatenate(
            [self.min_residuals, np.array([min_val], dtype=self.min_residuals.dtype)]
        )
        self.gt_contact_locations = np.concatenate(
            [self.gt_contact_locations, gt_se3[None, :, :]], axis=0
        )

    def add_batch(
        self,
        residuals: np.ndarray,  # (T, P)
        ft_values: np.ndarray,  # (T, 6)
        times: np.ndarray,  # (T,)
        contact_mask: np.ndarray,  # (T,)
        gt_contact_locations: np.ndarray = None,  # (T, 4, 4) or (T, 3)
    ) -> None:
        """Add a batch of timesteps."""
        residuals = np.asarray(residuals, dtype=self.residuals.dtype)
        ft_values = np.asarray(ft_values, dtype=self.ft_values.dtype)
        times = np.asarray(times, dtype=self.times.dtype)
        contact_mask = np.asarray(contact_mask, dtype=np.uint8)

        P = self.mesh_points.shape[0]
        T_new = residuals.shape[0]

        if residuals.shape[1] != P:
            raise ValueError(
                f"residuals shape {residuals.shape} incompatible with P={P}"
            )
        if ft_values.shape != (T_new, 6):
            raise ValueError(
                f"ft_values shape {ft_values.shape} != expected ({T_new}, 6)"
            )
        if times.shape != (T_new,):
            raise ValueError(f"times shape {times.shape} != expected ({T_new},)")
        if contact_mask.shape != (T_new,):
            raise ValueError(
                f"contact_mask shape {contact_mask.shape} != expected ({T_new},)"
            )

        # Handle gt_contact_locations - convert 3D points to SE3 if needed
        if gt_contact_locations is None:
            gt_se3_batch = np.tile(np.eye(4), (T_new, 1, 1)).astype(
                self.gt_contact_locations.dtype
            )
        else:
            gt_contact_locations = np.asarray(gt_contact_locations)
            if gt_contact_locations.shape == (T_new, 3):
                # Convert 3D points to SE3 with identity rotation
                gt_se3_batch = np.tile(np.eye(4), (T_new, 1, 1)).astype(
                    self.gt_contact_locations.dtype
                )
                gt_se3_batch[:, :3, 3] = gt_contact_locations
            elif gt_contact_locations.shape == (T_new, 4, 4):
                gt_se3_batch = gt_contact_locations.astype(
                    self.gt_contact_locations.dtype
                )
            else:
                raise ValueError(
                    f"gt_contact_locations shape {gt_contact_locations.shape} != expected ({T_new}, 3) or ({T_new}, 4, 4)"
                )

        # Compute min residual info for batch
        min_indices = np.argmin(residuals, axis=1)
        min_vals = residuals[np.arange(T_new), min_indices]

        self.residuals = np.concatenate([self.residuals, residuals], axis=0)
        self.ft_values = np.concatenate([self.ft_values, ft_values], axis=0)
        self.times = np.concatenate([self.times, times])
        self.contact_mask = np.concatenate([self.contact_mask, contact_mask])
        self.min_residual_indices = np.concatenate(
            [self.min_residual_indices, min_indices]
        )
        self.min_residuals = np.concatenate([self.min_residuals, min_vals])
        self.gt_contact_locations = np.concatenate(
            [self.gt_contact_locations, gt_se3_batch], axis=0
        )

    # ---------- Accessors ----------
    def get_contact_points(self) -> np.ndarray:
        """Return (T, 3) array of estimated contact points (min residual points)."""
        return self.mesh_points[self.min_residual_indices]

    def get_contact_normals(self) -> np.ndarray:
        """Return (T, 3) array of normals at estimated contact points."""
        return self.mesh_normals[self.min_residual_indices]

    def get_gt_contact_points(self) -> np.ndarray:
        """Return (T, 3) array of ground truth contact points (translation from SE3)."""
        return self.gt_contact_locations[:, :3, 3]

    def get_timestep(self, t: int) -> Dict:
        """Return dict for timestep t."""
        return {
            "residuals": self.residuals[t],  # (P,)
            "ft_value": self.ft_values[t],  # (6,)
            "time": float(self.times[t]),
            "contact": bool(self.contact_mask[t]),
            "min_residual_idx": int(self.min_residual_indices[t]),
            "min_residual": float(self.min_residuals[t]),
            "contact_point": self.mesh_points[self.min_residual_indices[t]],  # (3,)
            "contact_normal": self.mesh_normals[self.min_residual_indices[t]],  # (3,)
            "gt_contact_location": self.gt_contact_locations[t],  # (4, 4)
            "gt_contact_point": self.gt_contact_locations[t, :3, 3],  # (3,)
        }

    # ---------- I/O ----------
    def save(self, path: str) -> None:
        """Compressed, pickle-free save."""
        np.savez_compressed(
            path,
            mesh_points=self.mesh_points,
            mesh_normals=self.mesh_normals,
            residuals=self.residuals,
            ft_values=self.ft_values,
            times=self.times,
            contact_mask=self.contact_mask,
            min_residual_indices=self.min_residual_indices,
            min_residuals=self.min_residuals,
            gt_contact_locations=self.gt_contact_locations,
        )

    @classmethod
    def load(cls, path: str) -> "ContactResidualResult":
        """Pickle-free load."""
        data = np.load(path, allow_pickle=False)
        # Handle backward compatibility - older files may not have gt_contact_locations
        if "gt_contact_locations" in data:
            gt_contact_locations = data["gt_contact_locations"]
        else:
            # Create default identity SE3 matrices for each timestep
            T = data["times"].shape[0]
            gt_contact_locations = np.tile(np.eye(4), (T, 1, 1)).astype(np.float32)
        return cls(
            mesh_points=data["mesh_points"],
            mesh_normals=data["mesh_normals"],
            residuals=data["residuals"],
            ft_values=data["ft_values"],
            times=data["times"],
            contact_mask=data["contact_mask"],
            min_residual_indices=data["min_residual_indices"],
            min_residuals=data["min_residuals"],
            gt_contact_locations=gt_contact_locations,
        )

    # ---------- Convenience ----------
    @property
    def P(self) -> int:
        """Number of mesh points."""
        return self.mesh_points.shape[0]

    @property
    def T(self) -> int:
        """Number of timesteps."""
        return self.times.shape[0]

    @property
    def num_contacts(self) -> int:
        """Number of timesteps with contact."""
        return int(self.contact_mask.sum())

    def summary(self) -> str:
        if self.T == 0:
            return "ContactResidualResult: [empty]"
        t_min, t_max = float(self.times.min()), float(self.times.max())
        res_min, res_max = (
            float(self.min_residuals.min()),
            float(self.min_residuals.max()),
        )
        res_mean = (
            float(self.min_residuals[self.contact_mask > 0].mean())
            if self.num_contacts > 0
            else None
        )
        gt_contact_pts = self.get_gt_contact_points()
        has_gt = not np.allclose(gt_contact_pts, 0)
        return (
            "ContactResidualResult:\n"
            f"  Mesh points (P): {self.P}\n"
            f"  Timesteps (T): {self.T}\n"
            f"  Contact timesteps: {self.num_contacts}\n"
            f"  Time range: ({t_min:.4f}, {t_max:.4f})\n"
            f"  Min residual range: ({res_min:.6f}, {res_max:.6f})\n"
            f"  Mean min residual (contact only): {res_mean}\n"
            f"  GT contact locations: {'present' if has_gt else 'not set'}"
        )


@dataclass
class ContactLocationInferenceResult:
    """
    Hierarchical result for contact location inference: K measurements, N particles each.

    Fixed shapes:
      - poses:                 (K, N, 4, 4)
      - times:                 (K, N)
      - contact_locations_sim: (K, N, T, 3)
      - contact_locations_obs: (K, T, 3)
      - contact_mask:          (K, T)
      - unnormalised_log_pdfs: (K, N)
      - measurement_ids:       (K,)
      - probe_poses:           (K, 4, 4)

    Notes:
      * K is the number of measurements.
      * N is particles per measurement (fixed).
      * T is trajectory length (fixed).
    """

    poses: np.ndarray
    times: np.ndarray
    contact_locations_sim: np.ndarray
    contact_locations_obs: np.ndarray
    contact_mask: np.ndarray
    unnormalised_log_pdfs: np.ndarray
    measurement_ids: np.ndarray
    probe_poses: np.ndarray

    # ---------- Constructors ----------
    @classmethod
    def empty(
        cls,
        num_particles: int,
        traj_len: int,
        float_dtype=np.float32,
        id_dtype=np.int64,
    ) -> "ContactLocationInferenceResult":
        N = int(num_particles)
        T = int(traj_len)
        if N <= 0:
            raise ValueError("num_particles must be > 0")
        if T <= 0:
            raise ValueError("traj_len must be > 0")

        return cls(
            poses=np.empty((0, N, 4, 4), dtype=float_dtype),
            times=np.empty((0, N), dtype=float_dtype),
            contact_locations_sim=np.empty((0, N, T, 3), dtype=float_dtype),
            contact_locations_obs=np.empty((0, T, 3), dtype=float_dtype),
            contact_mask=np.empty((0, T), dtype=np.uint8),
            unnormalised_log_pdfs=np.empty((0, N), dtype=float_dtype),
            measurement_ids=np.empty((0,), dtype=id_dtype),
            probe_poses=np.empty((0, 4, 4), dtype=float_dtype),
        )

    # ---------- Append one measurement (with N particles) ----------
    def add_measurement(
        self,
        poses: np.ndarray,  # (N, 4, 4)
        times: np.ndarray,  # (N,)
        contact_locations_sim: np.ndarray,  # (N, T, 3)
        contact_locations_obs: np.ndarray,  # (T, 3)
        contact_mask: np.ndarray,  # (T,)
        unnormalised_log_pdfs: np.ndarray,  # (N,)
        measurement_id: int,
        probe_pose: np.ndarray,  # (4, 4)
    ) -> None:
        poses = np.asarray(poses)
        times = np.asarray(times)
        contact_locations_sim = np.asarray(contact_locations_sim)
        contact_locations_obs = np.asarray(contact_locations_obs)
        contact_mask = np.asarray(contact_mask, dtype=np.uint8)
        unnormalised_log_pdfs = np.asarray(unnormalised_log_pdfs)
        probe_pose = np.asarray(probe_pose)

        K_curr = self.poses.shape[0]
        N_expected = self.poses.shape[1] if K_curr > 0 else poses.shape[0]
        T_expected = (
            self.contact_locations_sim.shape[2]
            if K_curr > 0
            else contact_locations_sim.shape[1]
        )

        # Validate shapes
        if poses.shape != (N_expected, 4, 4):
            raise ValueError(
                f"poses shape {poses.shape} != expected {(N_expected, 4, 4)}"
            )
        if times.shape != (N_expected,):
            raise ValueError(f"times shape {times.shape} != expected {(N_expected,)}")
        if contact_locations_sim.shape != (N_expected, T_expected, 3):
            raise ValueError(
                f"contact_locations_sim shape {contact_locations_sim.shape} != expected {(N_expected, T_expected, 3)}"
            )
        if contact_locations_obs.shape != (T_expected, 3):
            raise ValueError(
                f"contact_locations_obs shape {contact_locations_obs.shape} != expected {(T_expected, 3)}"
            )
        if contact_mask.shape != (T_expected,):
            raise ValueError(
                f"contact_mask shape {contact_mask.shape} != expected {(T_expected,)}"
            )
        if unnormalised_log_pdfs.shape != (N_expected,):
            raise ValueError(
                f"unnormalised_log_pdfs shape {unnormalised_log_pdfs.shape} != expected {(N_expected,)}"
            )
        if probe_pose.shape != (4, 4):
            raise ValueError(f"probe_pose shape {probe_pose.shape} != expected (4, 4)")

        # Append along K
        self.poses = np.concatenate([self.poses, poses[None, ...]], axis=0)
        self.times = np.concatenate([self.times, times[None, ...]], axis=0)
        self.contact_locations_sim = np.concatenate(
            [self.contact_locations_sim, contact_locations_sim[None, ...]], axis=0
        )
        self.contact_locations_obs = np.concatenate(
            [self.contact_locations_obs, contact_locations_obs[None, ...]], axis=0
        )
        self.contact_mask = np.concatenate(
            [self.contact_mask, contact_mask[None, ...]], axis=0
        )
        self.unnormalised_log_pdfs = np.concatenate(
            [self.unnormalised_log_pdfs, unnormalised_log_pdfs[None, ...]], axis=0
        )
        self.measurement_ids = np.concatenate(
            [
                self.measurement_ids,
                np.array([measurement_id], dtype=self.measurement_ids.dtype),
            ],
            axis=0,
        )
        self.probe_poses = np.concatenate(
            [self.probe_poses, probe_pose[None, ...]], axis=0
        )

    # ---------- Accessors ----------
    def get_measurement(self, k: int) -> Dict:
        """Return dict for measurement k with arrays of shape (N, ...)."""
        return {
            "poses": self.poses[k],
            "times": self.times[k],
            "contact_locations_sim": self.contact_locations_sim[k],
            "contact_locations_obs": self.contact_locations_obs[k],
            "contact_mask": self.contact_mask[k],
            "unnormalised_log_pdfs": self.unnormalised_log_pdfs[k],
            "measurement_id": int(self.measurement_ids[k]),
            "probe_pose": self.probe_poses[k],
        }

    # ---------- I/O ----------
    def save(self, path: str) -> None:
        """Compressed, pickle-free save."""
        np.savez_compressed(
            path,
            poses=self.poses,
            times=self.times,
            contact_locations_sim=self.contact_locations_sim,
            contact_locations_obs=self.contact_locations_obs,
            contact_mask=self.contact_mask,
            unnormalised_log_pdfs=self.unnormalised_log_pdfs,
            measurement_ids=self.measurement_ids,
            probe_poses=self.probe_poses,
        )

    @classmethod
    def load(cls, path: str) -> "ContactLocationInferenceResult":
        """Pickle-free load (arrays must be numeric/fixed-shape)."""
        data = np.load(path, allow_pickle=False)
        return cls(
            poses=data["poses"],
            times=data["times"],
            contact_locations_sim=data["contact_locations_sim"],
            contact_locations_obs=data["contact_locations_obs"],
            contact_mask=data["contact_mask"],
            unnormalised_log_pdfs=data["unnormalised_log_pdfs"],
            measurement_ids=data["measurement_ids"],
            probe_poses=data["probe_poses"],
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
        return self.contact_locations_sim.shape[2] if self.K > 0 else 0

    def summary(self) -> str:
        if self.K == 0:
            return "ContactLocationInferenceResult: [empty]"
        lp_min = (
            float(self.unnormalised_log_pdfs.min())
            if self.unnormalised_log_pdfs.size
            else None
        )
        lp_max = (
            float(self.unnormalised_log_pdfs.max())
            if self.unnormalised_log_pdfs.size
            else None
        )
        return (
            "ContactLocationInferenceResult:\n"
            f"  Measurements (K): {self.K}\n"
            f"  Particles per measurement (N): {self.N}\n"
            f"  Trajectory length (T): {self.T}\n"
            f"  Pose shape: {self.poses.shape}\n"
            f"  Contact locations sim shape: {self.contact_locations_sim.shape}\n"
            f"  Contact locations obs shape: {self.contact_locations_obs.shape}\n"
            f"  Unnormalised log-pdf range: ({lp_min}, {lp_max})"
        )
