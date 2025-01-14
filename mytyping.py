from typing import Any, Literal, TypeAlias , List
import numpy as np
from numpy.typing import NDArray

'''
TypeAlias ：类型别名
Literal ：字面量类型
'''
Number: TypeAlias = float | int
TeamName: TypeAlias = Literal["Red", "Blue"]
Observation: TypeAlias = dict[str, int]
Action: TypeAlias = list[Number] | NDArray[np.int_]
Reward: TypeAlias = dict[str, int]
Info: TypeAlias = dict[str, Any]
