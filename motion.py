from typing import List
from geometry2d import Transform2d, Vector2d, Twist2d, Twist2dVelocity
from dataclasses import dataclass
from constraints import SE2Constraint, DASH_MOVEMENT_CONSTRAINT



@dataclass
class SE2Trajectory:
    initial_pose: Transform2d
    final_pose: Transform2d
    trajectory: List[Transform2d]
    robot_constraints: SE2Constraint

    def __init__(self, initial_pose: Transform2d, final_pose: Transform2d, robot_constraints: SE2Constraint):
        self.initial_pose = initial_pose
        self.final_pose = final_pose
        self.robot_constraints = robot_constraints
        self.trajectory = self.construct_trajectory()

    def construct_trajectory(self) -> List[Transform2d]:
        pass        





