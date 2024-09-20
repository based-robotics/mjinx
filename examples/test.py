from dataclasses import dataclass, field

import numpy as np


# @dataclass
# class A:
#     a: int
#     b: np.ndarray


# print(dir(A))
# print(A.__dataclass_fields__.keys())


class A:
    a: int
    b: float
    __c: np.ndarray
    _d: str

    def __setattr__(self, name: str, value) -> None:
        super().__setattr__(name, value)
        print(f"Changing value of {name}")

    def lol(self):
        self.__c = 1


a = A()
a.a = 0
a.b = 1.0
a.lol()
a._d = "kek"
