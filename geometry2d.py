from dataclasses import dataclass
import json

from numpy import cos, sin, pi, tan


@dataclass
class Vector2d:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __add__(self, other: "Vector2d") -> "Vector2d":
        if type(other) == Vector2d:
            return Vector2d(self.x + other.x, self.y + other.y)
        else:
            raise TypeError("Unsupported type for addition with Vector2d")

    def __sub__(self, other: "Vector2d") -> "Vector2d":
        if type(other) == Vector2d:
            return Vector2d(self.x - other.x, self.y - other.y)
        else:
            raise TypeError("Unsupported type for subtraction with Vector2d")

    def __mul__(self, other: float) -> "Vector2d":
        return Vector2d(self.x * other, self.y * other)

    def __truediv__(self, other: float) -> "Vector2d":
        return Vector2d(self.x / other, self.y / other)

    def norm(self) -> float:
        return (self.x**2 + self.y**2) ** 0.5

    def rotate(self, angle: float) -> "Vector2d":
        return Vector2d(self.x * cos(angle) - self.y * sin(angle), self.x * sin(angle) + self.y * cos(angle))


@dataclass
class Transform2d:
    x: float
    y: float
    theta: float

    def __mul__(self, other: "Transform2d") -> "Transform2d":
        return Transform2d(
            self.x + other.x * cos(self.theta) - other.y * sin(self.theta),
            self.y + other.x * sin(self.theta) + other.y * cos(self.theta),
            self.theta + other.theta,
        )

    def log(self) -> "Twist2d":
        theta = self.theta
        halfu = 0.5 * theta + snz(theta)
        v = halfu / tan(halfu)
        return Twist2d(v * self.x + halfu * self.y, -halfu * self.x + v * self.y, theta)

    @classmethod
    def from_json(cls, json_string) -> "Transform2d":
        data = json.loads(json_string)
        return cls(**data)


@dataclass
class Twist2d:
    dx: float
    dy: float
    dyaw: float

    @classmethod
    def from_velocity(cls, vx: float, vy: float, w: float, dt: float) -> "Twist2d":
        return cls(vx * dt, vy * dt, w * dt)
    
    @classmethod
    def from_twist2dvelocity(cls, twist2dvelocity: "Twist2dVelocity", dt: float) -> "Twist2d":
        return cls(twist2dvelocity.vx * dt, twist2dvelocity.vy * dt, twist2dvelocity.w * dt)

    def exp(self) -> Transform2d:
        angle = self.dyaw
        heading = self.dyaw
        u = angle + snz(angle)
        c = 1 - cos(u)
        s = sin(u)
        translation = Vector2d((s * self.dx - c * self.dy) / u, (c * self.dx + s * self.dy) / u)
        return Transform2d(translation.x, translation.y, heading)

    def to_velocity(self, dt: float) -> "Twist2dVelocity":
        return Twist2dVelocity(self.dx / dt, self.dy / dt, self.dyaw / dt)


@dataclass
class Twist2dVelocity:
    def __init__(self, vx: float, vy: float, w: float):
        self.vx = vx
        self.vy = vy
        self.w = w

    def to_twist2d(self, dt: float) -> Twist2d:
        return Twist2d(self.vx * dt, self.vy * dt, self.w * dt)

    def exp(self, dt: float) -> Transform2d:
        return self.to_twist2d(dt).exp()
    
    def __sub__(self, other: "Twist2dVelocity") -> "Twist2dVelocity":
        return Twist2dVelocity(self.vx - other.vx, self.vy - other.vy, self.w - other.w)
    
    def __add__(self, other: "Twist2dVelocity") -> "Twist2dVelocity":
        return Twist2dVelocity(self.vx + other.vx, self.vy + other.vy, self.w + other.w)
    def clone(self) -> "Twist2dVelocity":
        return Twist2dVelocity(self.vx, self.vy, self.w)

class AccelerationSE2:
    def __init__(self, ax: float, ay: float, a_yaw: float):
        self.ax = ax
        self.ay = ay
        self.a_yaw = a_yaw

    def to_twist2d(self, dt: float) -> Twist2d:
        return Twist2d(self.ax * dt**2, self.ay * dt**2, self.a_yaw * dt**2)

    def to_velocity(self, dt: float) -> Twist2dVelocity:
        return Twist2dVelocity(self.ax * dt, self.ay * dt, self.a_yaw * dt)


def wrap_angle(angle: float) -> float:
    while angle > pi:
        angle -= 2 * pi
    while angle < -pi:
        angle += 2 * pi
    return angle


def snz(x: float) -> float:

    EPS = 1e-9
    return EPS if x >= 0.0 else -EPS
