from dataclasses import dataclass, field

import numpy as np


@dataclass
class A:
    a: int
    b: float


@dataclass
class B(A):
    aa: int


a = A(0, 1.0)
print(dir(a))
print(A.__dataclass_fields__.keys())
print(B.__dataclass_fields__.keys())
