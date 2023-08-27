import numpy as np
import pyvista as pv
from typing import Any, Dict, Type, Optional

from dynamics.sat_dynamics import satelliteDynamics
from dynamics.base_dynamics import baseDynamics
from dynamics.dynamic_object import dynamicObject
from util.util import *

class spacecraft(dynamicObject):

    def __init__(
            self,
            dynamics: Type[satelliteDynamics],
            mesh: Union[Dict[str, list[Any]],Type[pv.DataSet]],
            name: Optional[str] = None

    ):
        
        super().__init__(
            dynamics = dynamics,
            mesh = mesh,
            name = name
        )
