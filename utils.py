import math
import numpy as np 

def angle_wrap(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

def axis_angle_from_matrix(R: np.ndarray):
    """
    Extract (axis, angle) from a 3x3 rotation matrix R.
    Returns:
        axis (list of float) - e.g. [ax, ay, az]
        angle (float)        - in radians
    """
    # Based on standard formula: angle = arccos( (trace(R) - 1) / 2 )
    # Then axis is built from the off-diagonal terms.

    eps = 1e-7
    trace_val = R[0, 0] + R[1, 1] + R[2, 2]
    # Clamp the trace to [-1, 3]
    trace_val = max(-1.0, min(3.0, trace_val))

    angle = math.acos(max(-1.0, min(1.0, (trace_val - 1.0) / 2.0)))
    if abs(angle) < eps:
        # No rotation
        return [0.0, 0.0, 1.0], 0.0

    # Axis:
    rx = R[2, 1] - R[1, 2]
    ry = R[0, 2] - R[2, 0]
    rz = R[1, 0] - R[0, 1]
    axis_len = math.sqrt(rx*rx + ry*ry + rz*rz)
    if axis_len < eps:
        # Fallback axis
        return [0.0, 0.0, 1.0], angle

    inv = 1.0 / axis_len
    return [rx*inv, ry*inv, rz*inv], angle
