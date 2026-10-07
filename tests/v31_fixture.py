"""Small deterministic V31 source/support fixtures, without learned weights."""
from fractions import Fraction

import numpy as np

from adaptive_vcm.motion_support import build_motion_support


def action_source(task="ar"):
    count, size = (16, 128) if task == "ar" else (1, 320)
    yy, xx = np.indices((size, size))
    rgb = np.stack([np.stack(((xx * 7 + t * 3) % 256,
                             (yy * 9 + t * 5) % 256,
                             (xx + yy + t * 11) % 256), -1)
                    for t in range(count)]).astype(np.uint8)
    mask = np.zeros((size, size), np.float32)
    mask[8:24, 8:24] = 1
    support = build_motion_support(rgb, mask, task)
    source = {"task": task, "rgb": rgb, "codec": "h264", "padded": False,
              "source_fps": Fraction(30000, 1001) if task == "ar" else None,
              "duration": Fraction(16016, 15000) if task == "ar" else Fraction(1, 25),
              "source_transform": (1., 1., 0, 0, size, size),
              "original_shape": (size, size), "control_protection": mask.copy()}
    return source, support
