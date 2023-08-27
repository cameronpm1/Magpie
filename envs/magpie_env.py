import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from typing import Any, Dict, Type, Optional, Union

from dynamics.base_dynamics import baseDynamics
from dynamics.sat_dynamics import satelliteDynamics
from dynamics.dynamic_object import dynamicObject
from dynamics.static_object import staticObject
from envs.gui import gui
from trajectory_planners.path_planner import pathPlanner
from envs.obstacle_avoidance_env import obstacleAvoidanceEnv

class magpieEnv(obstacleAvoidanceEnv):

    def __init__(
            self,
            main_object: Type[dynamicObject],
            path_planner: Type[pathPlanner],
            kwargs: Dict[str, Any],
            
    ):
        
        super.__init__(
            main_object=main_object,
            path_planner=path_planner
            kwargs=kwargs,
        )

    




    '''
    def plot() -> None:
        if not self.gui_on:
            print('Error: GUI set to off')
            exit()
        
        self.gui.plot_object(self.object.get_points(), self.object.get_lines())
        self.gui.show()
    '''
