from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Iterable
import numpy as np
import os
from urllib.parse import quote as _q, unquote as _uq

DATASET_VERSION = "1.2"

@dataclass
class RoboticsDatasetV2:
    """
    Robotics dataset stored in a single ``.npz`` file.

    Streams: RGB images, depth images, joint states, **static SE(3) frames**, and
    **per-name SE(3) trajectories**, plus optional force/torque (F/T). Each stream
    has its own timestamps.

    Static SE(3): same as before (v1.1)
      - ``se3_names`` : (K,)
      - ``se3_mats``  : (K,4,4)

    NEW (v1.2): SE(3) trajectories stored PER NAME (no ragged concatenation)
      - In memory:  ``se3_traj`` is a dict: name -> (times(T,), mats(T,4,4))
      - On disk: two arrays per name:
          ``se3_traj/<encoded_name>/times`` and ``se3_traj/<encoded_name>/mats``
        (name is percent-encoded; decoded on load)

    F/T (if present):
      - ``ft`` : (T_ft,6), ``ft_ts`` : (T_ft,)

    Notes
    -----
    - ``.npz`` is not appendable; load → append in memory → save.
    - Timestamps are stored as float64 seconds (interpretation is up to you).
    """

    # --- RGB images ---
    images: Optional[np.ndarray] = None           # (T_img, H, W, C)
    image_ts: Optional[np.ndarray] = None         # (T_img,)

    # --- Depth images ---
    depth: Optional[np.ndarray] = None            # (T_depth, H, W)
    depth_ts: Optional[np.ndarray] = None         # (T_depth,)

    # --- Joint states ---
    joint_states: Optional[np.ndarray] = None     # (T_joint, DoF)
    joint_ts: Optional[np.ndarray] = None         # (T_joint,)

    # --- Static SE(3) frames (unchanged) ---
    se3_names: Optional[np.ndarray] = None        # (K,), unicode
    se3_mats: Optional[np.ndarray] = None         # (K,4,4)

    # --- NEW: Per-name SE(3) trajectories (in memory as a dict) ---
    # name -> (times(T,), mats(T,4,4))
    se3_traj: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    # --- Force/Torque ---
    has_ft: bool = False
    ft: Optional[np.ndarray] = None               # (T_ft,6)
    ft_ts: Optional[np.ndarray] = None            # (T_ft,)

    # --- optional arbitrary metadata (JSON string) ---
    metadata_json: Optional[str] = None

    # --- internal/schema ---
    version: str = field(default=DATASET_VERSION, init=False)

    # ----------------------- small helpers -----------------------
    @staticmethod
    def _as_float64_ts(ts: float | np.ndarray | Iterable[float]) -> np.ndarray:
        arr = np.asarray(ts, dtype=np.float64)
        return arr.reshape(1,) if arr.ndim == 0 else arr

    @staticmethod
    def _ensure_image_3d(img: np.ndarray) -> np.ndarray:
        img = np.asarray(img)
        if img.ndim == 2:
            img = img[..., None]
        if img.ndim != 3:
            raise ValueError(f"Image must be (H,W[,C]), got {img.shape}")
        return img

    @staticmethod
    def _ensure_images_4d(images: np.ndarray) -> np.ndarray:
        images = np.asarray(images)
        if images.ndim == 3:         # (H,W,C) or (H,W,1) -> (1,H,W,C)
            images = images[None, ...]
        elif images.ndim != 4:
            raise ValueError(f"Images must be (N,H,W,C) or (H,W,C), got {images.shape}")
        return images

    @staticmethod
    def _ensure_depth_2d(d: np.ndarray) -> np.ndarray:
        d = np.asarray(d)
        if d.ndim == 3 and d.shape[-1] == 1:
            d = d[..., 0]
        if d.ndim != 2:
            raise ValueError(f"Depth must be (H,W) or (H,W,1), got {d.shape}")
        return d

    @staticmethod
    def _ensure_depths_3d(D: np.ndarray) -> np.ndarray:
        D = np.asarray(D)
        if D.ndim == 2:
            D = D[None, ...]
        elif D.ndim == 4 and D.shape[-1] == 1:
            D = D[..., 0]
        if D.ndim != 3:
            raise ValueError(f"Depths must be (N,H,W) or (N,H,W,1), got {D.shape}")
        return D

    @staticmethod
    def _ensure_4x4(T: np.ndarray) -> np.ndarray:
        T = np.asarray(T, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"T must be (4,4), got {T.shape}")
        return T

    @staticmethod
    def _ensure_4x4_batch(Ts: np.ndarray) -> np.ndarray:
        Ts = np.asarray(Ts, dtype=np.float64)
        if Ts.ndim == 2 and Ts.shape == (4, 4):
            return Ts[None, ...]
        if Ts.ndim == 3 and Ts.shape[1:] == (4, 4):
            return Ts
        raise ValueError(f"Ts must be (N,4,4) or (4,4), got {Ts.shape}")

    @staticmethod
    def _concat(a: Optional[np.ndarray], b: np.ndarray, axis: int = 0) -> np.ndarray:
        return b if a is None else np.concatenate([a, b], axis=axis)

    @staticmethod
    def _require_same_shape(prefix: str, current: Tuple[int, ...], new: Tuple[int, ...]) -> None:
        if current != new:
            raise ValueError(f"{prefix} mismatch: existing {current} vs new {new}")

    # ----------------------- validation -----------------------
    def validate(self) -> None:
        def _req(a, name):
            if a is None:
                raise ValueError(f"Required field '{name}' is None.")

        def _check_stream(arr, ts, arr_name, ts_name, lead_dim=None):
            if arr is None and ts is None:
                return
            _req(arr, arr_name)
            _req(ts, ts_name)
            if ts.ndim != 1:
                raise ValueError(f"{ts_name} must be 1D, got {ts.shape}")
            if arr.shape[0] != ts.shape[0]:
                raise ValueError(
                    f"Length mismatch: {arr_name}.shape[0]={arr.shape[0]} vs {ts_name}.shape[0]={ts.shape[0]}"
                )
            if lead_dim is not None and arr.ndim < lead_dim:
                raise ValueError(f"{arr_name} must have at least {lead_dim} dims, got {arr.ndim}")

        # Normalize timestamp dtypes
        for ts_name in ["image_ts", "depth_ts", "joint_ts", "ft_ts"]:
            ts = getattr(self, ts_name)
            if ts is not None:
                setattr(self, ts_name, np.asarray(ts, dtype=np.float64))

        # Canonical shapes
        if self.images is not None and self.images.ndim != 4:
            raise ValueError(f"images must be (T,H,W,C), got {self.images.shape}")
        if self.depth is not None and self.depth.ndim != 3:
            raise ValueError(f"depth must be (T,H,W), got {self.depth.shape}")
        if self.joint_states is not None and self.joint_states.ndim != 2:
            raise ValueError(f"joint_states must be (T,DoF), got {self.joint_states.shape}")

        _check_stream(self.images, self.image_ts, "images", "image_ts", lead_dim=4)
        _check_stream(self.depth, self.depth_ts, "depth", "depth_ts", lead_dim=3)
        _check_stream(self.joint_states, self.joint_ts, "joint_states", "joint_ts", lead_dim=2)

        # Static SE(3)
        if self.se3_names is not None or self.se3_mats is not None:
            _req(self.se3_names, "se3_names")
            _req(self.se3_mats, "se3_mats")
            if self.se3_mats.ndim != 3 or self.se3_mats.shape[1:] != (4, 4):
                raise ValueError(f"se3_mats must be (K,4,4), got {self.se3_mats.shape}")
            if self.se3_names.shape[0] != self.se3_mats.shape[0]:
                raise ValueError("se3_names and se3_mats must have matching K.")
            self.se3_names = np.asarray(self.se3_names, dtype=np.dtype("U"))

        # Per-name SE(3) trajectories
        for name, (times, mats) in self.se3_traj.items():
            times = np.asarray(times, dtype=np.float64)
            mats = np.asarray(mats, dtype=np.float64)
            if times.ndim != 1:
                raise ValueError(f"se3_traj[{name}]: times must be 1D, got {times.shape}")
            if mats.ndim != 3 or mats.shape[1:] != (4, 4):
                raise ValueError(f"se3_traj[{name}]: mats must be (T,4,4), got {mats.shape}")
            if times.shape[0] != mats.shape[0]:
                raise ValueError(f"se3_traj[{name}]: length mismatch times({times.shape[0]}) vs mats({mats.shape[0]})")
            # write back normalized
            self.se3_traj[name] = (times, mats)

        # FT checks
        if self.has_ft:
            _req(self.ft, "ft")
            _req(self.ft_ts, "ft_ts")
            if self.ft.ndim != 2 or self.ft.shape[1] != 6:
                raise ValueError(f"ft must be (T,6), got {self.ft.shape}")
            if self.ft_ts.ndim != 1 or self.ft_ts.shape[0] != self.ft.shape[0]:
                raise ValueError("ft and ft_ts must have matching T.")
        else:
            self.ft = None
            self.ft_ts = None

        if self.metadata_json is not None and not isinstance(self.metadata_json, str):
            self.metadata_json = str(self.metadata_json)

    # ----------------------- save/load -----------------------
    @staticmethod
    def _traj_times_key(name: str) -> str:
        return f"se3_traj/{_q(name, safe='')}/times"

    @staticmethod
    def _traj_mats_key(name: str) -> str:
        return f"se3_traj/{_q(name, safe='')}/mats"

    def save(self, path: str, compressed: bool = True) -> None:
        """Write to a single .npz file (no pickles)."""
        self.validate()
        payload = {
            "version": np.array(self.version),
            "has_ft": np.array(self.has_ft),
        }
        # images
        if self.images is not None: payload["images"] = self.images
        if self.image_ts is not None: payload["image_ts"] = self.image_ts
        # depth
        if self.depth is not None: payload["depth"] = self.depth
        if self.depth_ts is not None: payload["depth_ts"] = self.depth_ts
        # joints
        if self.joint_states is not None: payload["joint_states"] = self.joint_states
        if self.joint_ts is not None: payload["joint_ts"] = self.joint_ts
        # static SE3
        if self.se3_names is not None: payload["se3_names"] = self.se3_names
        if self.se3_mats is not None: payload["se3_mats"] = self.se3_mats
        # per-name SE3 trajectories
        for name, (times, mats) in self.se3_traj.items():
            payload[self._traj_times_key(name)] = np.asarray(times, dtype=np.float64)
            payload[self._traj_mats_key(name)]  = np.asarray(mats, dtype=np.float64)
        # FT
        if self.has_ft and self.ft is not None: payload["ft"] = self.ft
        if self.has_ft and self.ft_ts is not None: payload["ft_ts"] = self.ft_ts
        # metadata
        if self.metadata_json is not None:
            payload["metadata_json"] = np.array(self.metadata_json)

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        saver = np.savez_compressed if compressed else np.savez
        saver(path, **payload)

    @classmethod
    def load(cls, path: str) -> "RoboticsDataset":
        with np.load(path, allow_pickle=False) as npz:
            files = set(npz.files)
            get = lambda k, default=None: npz[k] if k in files else default

            version = str(get("version", np.array("unknown")))
            has_ft = bool(get("has_ft", np.array(False)))

            # reconstruct per-name trajectories
            se3_traj: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            # scan keys of the form "se3_traj/<enc>/times" & ".../mats"
            for k in files:
                if not k.startswith("se3_traj/") or not k.endswith("/times"):
                    continue
                enc = k[len("se3_traj/"):-len("/times")]
                name = _uq(enc)
                tkey = k
                mkey = f"se3_traj/{enc}/mats"
                if mkey in files:
                    times = np.asarray(npz[tkey], dtype=np.float64)
                    mats  = np.asarray(npz[mkey], dtype=np.float64)
                    se3_traj[name] = (times, mats)

            ds = cls(
                images=get("images"),
                image_ts=get("image_ts"),
                depth=get("depth"),
                depth_ts=get("depth_ts"),
                joint_states=get("joint_states"),
                joint_ts=get("joint_ts"),
                se3_names=get("se3_names"),
                se3_mats=get("se3_mats"),
                se3_traj=se3_traj,
                has_ft=has_ft,
                ft=get("ft") if has_ft else None,
                ft_ts=get("ft_ts") if has_ft else None,
                metadata_json=str(get("metadata_json", np.array("None"))),
            )
            ds.version = version
            ds.validate()
            return ds

    # ----------------------- SE3 utilities (static) -----------------------
    def get_transform(self, name: str) -> np.ndarray:
        """Return a (4,4) static SE3 matrix by name."""
        if self.se3_names is None or self.se3_mats is None:
            raise KeyError("No static SE3 transforms stored.")
        idx = np.where(self.se3_names == name)[0]
        if len(idx) == 0:
            raise KeyError(f"SE3 '{name}' not found. Available: {list(self.se3_names)}")
        return self.se3_mats[idx[0]]

    def add_transform(self, name: str, T: np.ndarray) -> None:
        """Add/replace a static (4,4) SE3 matrix."""
        T = self._ensure_4x4(T)
        if self.se3_names is None:
            self.se3_names = np.array([name], dtype=np.dtype("U"))
            self.se3_mats = T[None, ...]
            return
        idx = np.where(self.se3_names == name)[0]
        if len(idx) == 0:
            self.se3_names = np.concatenate([self.se3_names, np.array([name], dtype=np.dtype("U"))])
            self.se3_mats = np.concatenate([self.se3_mats, T[None, ...]], axis=0)
        else:
            self.se3_mats[idx[0]] = T

    # ----------------------- SE3 trajectories (per name) -----------------------
    def add_se3_traj_stream(self, name: str) -> None:
        """Declare a named trajectory (no samples yet). No-op if it exists."""
        if name not in self.se3_traj:
            self.se3_traj[name] = (np.empty((0,), dtype=np.float64),
                                   np.empty((0, 4, 4), dtype=np.float64))

    def append_se3_traj(self, name: str, T: np.ndarray, ts: float | np.ndarray | Iterable[float]) -> None:
        """Append one (4,4) pose + timestamp to the named trajectory."""
        Ts = self._ensure_4x4_batch(T)
        ts = self._as_float64_ts(ts)
        if Ts.shape[0] != 1 or ts.shape[0] != 1:
            raise ValueError("append_se3_traj expects a single pose and a single timestamp.")
        self.append_se3_traj_batch(name, Ts, ts)

    def append_se3_traj_batch(self, name: str, Ts: np.ndarray, ts: np.ndarray | Iterable[float]) -> None:
        """Append many (N,4,4) poses and (N,) timestamps to the named trajectory."""
        Ts = self._ensure_4x4_batch(Ts)
        ts = self._as_float64_ts(ts)
        if ts.ndim != 1 or ts.shape[0] != Ts.shape[0]:
            raise ValueError(f"ts must be 1D with length N={Ts.shape[0]}, got {ts.shape}")
        if name not in self.se3_traj:
            self.add_se3_traj_stream(name)
        cur_t, cur_Ts = self.se3_traj[name]
        cur_t = np.asarray(cur_t, dtype=np.float64)
        cur_Ts = np.asarray(cur_Ts, dtype=np.float64)
        self.se3_traj[name] = (np.concatenate([cur_t, ts], axis=0),
                               np.concatenate([cur_Ts, Ts], axis=0))

    def get_se3_traj(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (times (T,), mats (T,4,4)) for a named trajectory."""
        if name not in self.se3_traj:
            raise KeyError(f"SE3 trajectory '{name}' not found. Available: {list(self.se3_traj.keys())}")
        return self.se3_traj[name]

    # ----------------------- append helpers (other streams) -----------------------
    def append_image(self, image: np.ndarray, ts: float | np.ndarray | Iterable[float]) -> None:
        img3 = self._ensure_image_3d(image)
        ts1 = self._as_float64_ts(ts)
        if ts1.shape[0] != 1:
            raise ValueError("append_image: expected a single timestamp.")
        if self.images is not None:
            self._require_same_shape("append_image (H,W,C)", tuple(self.images.shape[1:]), tuple(img3.shape))
        self.images = self._concat(self.images, img3[None, ...], axis=0)
        self.image_ts = self._concat(self.image_ts, ts1, axis=0)

    def append_depth(self, depth_img: np.ndarray, ts: float | np.ndarray | Iterable[float]) -> None:
        d2 = self._ensure_depth_2d(depth_img)
        ts1 = self._as_float64_ts(ts)
        if ts1.shape[0] != 1:
            raise ValueError("append_depth: expected a single timestamp.")
        if self.depth is not None:
            self._require_same_shape("append_depth (H,W)", tuple(self.depth.shape[1:]), tuple(d2.shape))
        self.depth = self._concat(self.depth, d2[None, ...], axis=0)
        self.depth_ts = self._concat(self.depth_ts, ts1, axis=0)

    def append_joint_state(self, q: np.ndarray, ts: float | np.ndarray | Iterable[float]) -> None:
        q = np.asarray(q)
        if q.ndim == 2 and q.shape[0] == 1:
            q = q[0]
        if q.ndim != 1:
            raise ValueError(f"append_joint_state: q must be 1D (DoF,), got {q.shape}")
        ts1 = self._as_float64_ts(ts)
        if ts1.shape[0] != 1:
            raise ValueError("append_joint_state: expected a single timestamp.")
        if self.joint_states is not None:
            self._require_same_shape("append_joint_state DoF", (self.joint_states.shape[1],), (q.shape[0],))
        self.joint_states = self._concat(self.joint_states, q[None, :], axis=0)
        self.joint_ts = self._concat(self.joint_ts, ts1, axis=0)

    def append_ft(self, sample: np.ndarray, ts: float | np.ndarray | Iterable[float]) -> None:
        s = np.asarray(sample, dtype=np.float64)
        if s.ndim == 2 and s.shape[0] == 1:
            s = s[0]
        if s.ndim != 1 or s.shape[0] != 6:
            raise ValueError(f"append_ft: sample must be 1D with 6 entries, got {s.shape}")
        ts1 = self._as_float64_ts(ts)
        if ts1.shape[0] != 1:
            raise ValueError("append_ft: expected a single timestamp.")
        if self.ft is not None:
            self._require_same_shape("append_ft length-6", (self.ft.shape[1],), (s.shape[0],))
        self.has_ft = True
        self.ft = self._concat(self.ft, s[None, :], axis=0)
        self.ft_ts = self._concat(self.ft_ts, ts1, axis=0)

    # ----------------------- batch append (other streams) -----------------------
    def append_images(self, images: np.ndarray, ts: np.ndarray | Iterable[float]) -> None:
        images4 = self._ensure_images_4d(images)
        ts = self._as_float64_ts(ts)
        if ts.ndim != 1 or ts.shape[0] != images4.shape[0]:
            raise ValueError(f"append_images: ts must be 1D of length N={images4.shape[0]}, got {ts.shape}")
        if self.images is not None:
            self._require_same_shape("append_images (H,W,C)", tuple(self.images.shape[1:]), tuple(images4.shape[1:]))
        self.images = self._concat(self.images, images4, axis=0)
        self.image_ts = self._concat(self.image_ts, ts, axis=0)

    def append_depths(self, depths: np.ndarray, ts: np.ndarray | Iterable[float]) -> None:
        D3 = self._ensure_depths_3d(depths)
        ts = self._as_float64_ts(ts)
        if ts.ndim != 1 or ts.shape[0] != D3.shape[0]:
            raise ValueError(f"append_depths: ts must be 1D of length N={D3.shape[0]}, got {ts.shape}")
        if self.depth is not None:
            self._require_same_shape("append_depths (H,W)", tuple(self.depth.shape[1:]), tuple(D3.shape[1:]))
        self.depth = self._concat(self.depth, D3, axis=0)
        self.depth_ts = self._concat(self.depth_ts, ts, axis=0)

    def append_joint_states(self, Q: np.ndarray, ts: np.ndarray | Iterable[float]) -> None:
        Q = np.asarray(Q)
        if Q.ndim != 2:
            raise ValueError(f"append_joint_states: Q must be (N,DoF), got {Q.shape}")
        ts = self._as_float64_ts(ts)
        if ts.ndim != 1 or ts.shape[0] != Q.shape[0]:
            raise ValueError(f"append_joint_states: ts must be 1D of length N={Q.shape[0]}, got {ts.shape}")
        if self.joint_states is not None:
            self._require_same_shape("append_joint_states DoF", (self.joint_states.shape[1],), (Q.shape[1],))
        self.joint_states = self._concat(self.joint_states, Q, axis=0)
        self.joint_ts = self._concat(self.joint_ts, ts, axis=0)

    def append_fts(self, F: np.ndarray, ts: np.ndarray | Iterable[float]) -> None:
        F = np.asarray(F, dtype=np.float64)
        if F.ndim != 2 or F.shape[1] != 6:
            raise ValueError(f"append_fts: F must be (N,6), got {F.shape}")
        ts = self._as_float64_ts(ts)
        if ts.ndim != 1 or ts.shape[0] != F.shape[0]:
            raise ValueError(f"append_fts: ts must be 1D of length N={F.shape[0]}, got {ts.shape}")
        if self.ft is not None:
            self._require_same_shape("append_fts length-6", (self.ft.shape[1],), (F.shape[1],))
        self.has_ft = True
        self.ft = self._concat(self.ft, F, axis=0)
        self.ft_ts = self._concat(self.ft_ts, ts, axis=0)

    # ----------------------- summary -----------------------
    def summary(self) -> Dict[str, Tuple]:
        """Quick shapes/lengths overview."""
        traj_info = {k: (v[0].shape, v[1].shape) for k, v in self.se3_traj.items()}
        return {
            "images": None if self.images is None else self.images.shape,
            "image_ts": None if self.image_ts is None else self.image_ts.shape,
            "depth": None if self.depth is None else self.depth.shape,
            "depth_ts": None if self.depth_ts is None else self.depth_ts.shape,
            "joint_states": None if self.joint_states is None else self.joint_states.shape,
            "joint_ts": None if self.joint_ts is None else self.joint_ts.shape,
            "se3_names": None if self.se3_names is None else self.se3_names.shape,
            "se3_mats": None if self.se3_mats is None else self.se3_mats.shape,
            "se3_traj": traj_info,  # dict: name -> ((T,), (T,4,4))
            "has_ft": self.has_ft,
            "ft": None if self.ft is None else self.ft.shape,
            "ft_ts": None if self.ft_ts is None else self.ft_ts.shape,
            "version": self.version,
        }
