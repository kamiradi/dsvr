from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Iterable
import numpy as np
import os
from urllib.parse import quote as _q, unquote as _uq

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
