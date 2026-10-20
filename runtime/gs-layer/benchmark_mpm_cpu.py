"""Import bridge to the vendored EMPM solver; unused benchmark removed."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp
from scipy.spatial import ConvexHull

from gaussian_io import load_gaussians

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'empm/phys_sim/diff_simulator/warp_mpm'))
from mpm_data_structure import MPMStateStruct, MPMModelStruct
from mpm_solver_diff import MPMWARPDiff
