from dataclasses import dataclass
from geometry2d import Twist2dVelocity

inches_to_meters = 0.0254

@dataclass
class SE2Constraint:
    max_velocity: Twist2dVelocity
    max_acceleration: Twist2dVelocity
    max_jerk: Twist2dVelocity | None = None
    robot_radius: float | None = None

DASH_MOVEMENT_CONSTRAINT = SE2Constraint(
    max_velocity=Twist2dVelocity(vx=1.5, vy=1.5, w=360.0 * 1.5),
    max_acceleration=Twist2dVelocity(vx=1.5, vy=1.5, w=360.0 * 3.0),
    max_jerk=Twist2dVelocity(vx=1.5, vy=1.5, w=360.0 * 3.0),
    robot_radius=(11.0 / 2.0) * inches_to_meters
)



