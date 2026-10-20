"""Load trained 3DGS attributes without treating an RGB point cloud as a model."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.spatial.transform import Rotation


@dataclass
class Gaussians:
    xyz: np.ndarray
    scales: np.ndarray
    quats: np.ndarray  # w, x, y, z
    opacity: np.ndarray
    sh: np.ndarray  # N, (degree + 1)^2, RGB; Graphdeco real SH convention

    def validate(self):
        n = len(self.xyz)
        if not n or self.xyz.shape != (n, 3) or self.scales.shape != (n, 3):
            raise ValueError("Invalid Gaussian position/scale shape")
        if self.quats.shape != (n, 4) or self.opacity.shape != (n,):
            raise ValueError("Invalid Gaussian quaternion/opacity shape")
        if self.sh.ndim != 3 or self.sh.shape[0] != n or self.sh.shape[2] != 3 or self.sh.shape[1] not in (1, 4, 9, 16):
            raise ValueError("Only real SH degrees 0–3 are supported")
        for name in self.__dataclass_fields__:
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError(f"Non-finite Gaussian {name}")
        if (self.scales <= 0).any() or ((self.opacity < 0) | (self.opacity > 1)).any():
            raise ValueError("Scales must be positive and opacity must be in [0,1]")
        norms = np.linalg.norm(self.quats, axis=1, keepdims=True)
        if (norms < 1e-8).any():
            raise ValueError("Zero quaternion")
        self.quats = self.quats / norms
        return self

    def subset(self, selection):
        return Gaussians(**{k: v[selection] for k, v in vars(self).items()})


def load_gaussians(path):
    path = Path(path)
    if path.suffix.lower() == ".ply":
        from plyfile import PlyData
        v = PlyData.read(path)["vertex"].data
        required = ["x", "y", "z", "opacity"] + [f"f_dc_{i}" for i in range(3)] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
        missing = set(required) - set(v.dtype.names)
        if missing:
            raise ValueError(f"This is not a complete trained Gaussian PLY; missing {sorted(missing)}")
        def cols(names):
            return np.column_stack([v[n] for n in names]).astype(np.float32)
        rest = sorted([k for k in v.dtype.names if k.startswith("f_rest_")], key=lambda x: int(x[7:]))
        sh = cols([f"f_dc_{i}" for i in range(3)])[:, None, :]
        if rest:
            sh = np.concatenate([sh, cols(rest).reshape(len(v), 3, -1).transpose(0, 2, 1)], axis=1)
        result = Gaussians(cols(["x", "y", "z"]), np.exp(cols([f"scale_{i}" for i in range(3)])), cols([f"rot_{i}" for i in range(4)]), expit(v["opacity"]).astype(np.float32), sh)
    elif path.suffix.lower() in (".ckpt", ".pt", ".pth"):
        import torch
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint.get("pipeline", checkpoint.get("state_dict", checkpoint))
        def get(name):
            candidates = [k for k in state if k == name or k.endswith(".gauss_params." + name) or k.endswith(".gaussians." + name) or k == "gauss_params." + name]
            if len(candidates) != 1:
                raise ValueError(f"Cannot uniquely locate Splatfacto Gaussian parameter {name}: {candidates}")
            return state[candidates[0]].detach().float().cpu().numpy()
        dc, rest = get("features_dc"), get("features_rest")
        if dc.ndim == 2:
            dc = dc[:, None, :]
        result = Gaussians(get("means"), np.exp(get("scales")), get("quats"), expit(get("opacities")).reshape(-1), np.concatenate([dc, rest], axis=1))
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            result = Gaussians(**{k: data[k].astype(np.float32) for k in Gaussians.__dataclass_fields__})
    else:
        raise ValueError("Expected a trained Graphdeco/Nerfstudio PLY, Splatfacto checkpoint, or documented Gaussian NPZ")
    return result.validate()


def transform_gaussians(g, matrix):
    """Apply an explicitly supplied proper similarity transform, never fit one to future frames."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError("Expected a homogeneous 4x4 model-to-trajectory transform")
    linear = matrix[:3, :3]
    scale = np.linalg.norm(linear, axis=0).mean()
    rotation = linear / scale
    if scale <= 0 or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < 0.999:
        raise ValueError("Transform must contain uniform positive scale and a proper rotation")
    source_rot = Rotation.from_quat(g.quats[:, [1, 2, 3, 0]])
    q = (Rotation.from_matrix(rotation) * source_rot).as_quat()[:, [3, 0, 1, 2]]
    transformed = Gaussians((g.xyz @ linear.T + matrix[:3, 3]).astype(np.float32), (g.scales * scale).astype(np.float32), q.astype(np.float32), g.opacity, g.sh)
    return transformed.validate(), rotation.T.astype(np.float32)
