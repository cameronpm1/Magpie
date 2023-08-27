import numpy as np
import pyvista as pv
from typing import Any, Dict, Type, Optional, Union

from dynamics.quad_dynamics import quadcopterDynamics
from dynamics.base_dynamics import baseDynamics
from dynamics.dynamic_object import dynamicObject
from util.util import *

class quadcopter(dynamicObject):

    def __init__(
            self,
            dynamics: Type[quadcopterDynamics],
            mesh: Union[Dict[str, list[Any]],Type[pv.DataSet]],
            name: Optional[str] = None,
    ):
        
        super().__init__(
            dynamics = dynamics,
            mesh = mesh,
            name = name,
        )