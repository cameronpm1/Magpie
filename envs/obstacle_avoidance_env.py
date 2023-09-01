import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import cvxpy as cp
import control 
import copy
from typing import Any, Dict, Type, Optional, Union

from dynamics.base_dynamics import baseDynamics
from dynamics.sat_dynamics import satelliteDynamics
from dynamics.dynamic_object import dynamicObject
from dynamics.static_object import staticObject
from dynamics.quad_dynamics import quadcopterDynamics
from envs.gui import gui
from trajectory_planning.path_planner import pathPlanner


class obstacleAvoidanceEnv():

    def __init__(
            self,
            main_object: Type[dynamicObject],
            path_planner: Type[pathPlanner],
            control_method: str, #MPC
            kwargs: Dict[str, Any],
            point_cloud_radius: float = 5,
            path_point_tolerance: float = 0.1,
            point_cloud_size: float = 3000,
    ):
        
        self.main_object = main_object
        self.path_planner = path_planner
        self.point_cloud_radius = point_cloud_radius
        self.point_cloud_size = point_cloud_size
        self.path_point_tolerance = path_point_tolerance
        self.obstacles = []
        
        self.point_cloud_plot = None
        self.point_cloud = None
        self.dynamic_obstacles = False

        kwargs['dynamics'] = main_object.dynamics
        self.control_method = getattr(self,control_method)(**kwargs)
        self.control_method.update_state()

        self.time = 0
        self.current_path = None

    def step(self) -> None:
        if self.dynamic_obstacles:
            self.update_point_cloud()
        action = self.compute_next_action()
        self.main_object.dynamics.set_control(action)
        self.main_object.step()
        self.control_method.update_state()
        for obstacle in self.obstacles:
            if isinstance(obstacle,dynamicObject):
                obstacle.step()
        self.check_reached_path_point()
        if self.current_path.size == 0:
            self.get_new_path()

    def reset(self) -> None:
        self.main_object.dynamics.reset_state()
        self.update_point_cloud()
        self.get_new_path()

    def compute_next_action(self) -> list[float]:
        next_action = self.control_method.compute_action(goal=self.current_path[0])
        #print(next_action)
        #print(np.linalg.norm(self.main_object.dynamics.get_vel()))
        return next_action

    def check_reached_path_point(self) -> None:
        if np.linalg.norm(self.current_path[0][0:3]-self.main_object.dynamics.state[0:3]) < self.path_point_tolerance:
            self.current_path = self.current_path[1:]

    def get_new_path(self) -> None:
        self.update_point_cloud()
        if self.point_cloud is None:
            point_cloud = self.point_cloud
        else:
            point_cloud = self.generate_proximal_point_cloud()
        new_path = self.path_planner.compute_desired_path(state=self.main_object.dynamics.state, point_cloud=point_cloud)
        self.current_path = new_path[1:]

    def add_obstacle(
            self,
            obstacle: Union[Type[dynamicObject],Type[staticObject]],
    ) -> None:
        
        obstacle.update_points()
        self.obstacles.append(obstacle)
        if isinstance(obstacle, dynamicObject):
            self.dynamic_obstacles = True

    def remove_obstacle(
            self,
            obstacle_name: str,
    ) -> None:
        
        for i,obstacle in enumerate(self.obstacles):
            if obstacle.name == obstacle_name:
                self.obstacles.pop(i)
        self.update_point_cloud()
        self.get_new_path()


    def get_object_data(self) -> list[Dict[str, Any]]:
        objects = []
        objects.append(self.main_object.temp_mesh)
        objects = copy.deepcopy(self.main_object.temp_mesh)
        objects['goal'] = self.current_path[0]
        if self.point_cloud_plot is not None:
            objects['point cloud'] = self.point_cloud_plot
        objects['final goal'] = self.path_planner.goal[0:3]
        return objects
    
    def generate_proximal_point_cloud(self) -> list[list[float]]:

        location = self.main_object.dynamics.get_pos()

        cloud = []

        for point in self.point_cloud:
                new_point = point-location
                if np.linalg.norm(new_point) < self.point_cloud_radius:
                    cloud.append(new_point)

        if len(cloud) == 0:
            return None
        
        return cloud
    
    def update_point_cloud(self) -> None:
        self.point_cloud = None
        if len(self.obstacles) > 0:
            object_cloud_size = int(self.point_cloud_size/len(self.obstacles))
        for obstacle in self.obstacles:
            local_point_cloud = np.array(obstacle.point_cloud_from_mesh(n=object_cloud_size))
            if self.point_cloud is None:
                self.point_cloud = local_point_cloud
            else:
                self.point_cloud = np.concatenate((self.point_cloud,local_point_cloud))

        if self.point_cloud_size > 2000:
            idx = np.random.randint(self.point_cloud_size, size=2000)
            self.point_cloud_plot = self.point_cloud[idx,:]
        else:
            self.point_cloud_plot = self.point_cloud

    def collision_check(self):
        if self.point_cloud is None:
            return False
        
        location = self.main_object.dynamics.get_pos()

        for point in self.point_cloud:
            new_point = point-location
            if np.linalg.norm(new_point) < self.path_planner.algorithm.distance_tolerance:
                    return True
        return False
    
    def goal_check(self):
        if np.linalg.norm(self.path_planner.goal[0:3]-self.main_object.dynamics.state[0:3]) < self.path_point_tolerance:
            self.current_path = self.current_path[1:]

    def set_new_path(self, new_path: list[list[float]]) -> None:
        '''
        set path for testing purposes
        '''
        self.current_path = new_path
    
    class MPC():

        def __init__(
                self,
                dynamics: Type[baseDynamics], #only works w/ quadcopterDynamics satelliteDynamics not supported yet
                upper_state_bounds: list[float],
                lower_state_bounds: list[float],
                upper_control_bounds: Optional[list[float]] = None,
                lower_control_bounds: Optional[list[float]] = None,
                horizon: int = 20,
                valued_actions: int = 5,
        ):
        
            self.dynamics = dynamics
            self.horizon = horizon
            self.valued_actions = valued_actions

            self.state_bounds = [lower_state_bounds,upper_state_bounds]
            self.control_bounds = [lower_control_bounds,upper_control_bounds]

            self.initialize_optimization_parameters()
            self.initialize_discrete_matrices()

        def initialize_optimization_parameters(self) -> None:
            self.x = cp.Variable((self.dynamics.state.size, self.horizon+1))
            self.u = cp.Variable((self.dynamics.control.size, self.horizon))
            self.x_init = cp.Parameter(self.dynamics.state.size)

            self.x_init.value = self.dynamics.state

        def initialize_discrete_matrices(self) -> None:
            # Convert continuous time dynamics into discrete time
            sys = control.StateSpace(self.dynamics.A, self.dynamics.B, self.dynamics.C, self.dynamics.D)
            sys_discrete = control.c2d(sys, self.dynamics.timestep*self.dynamics.horizon, method='zoh')

            self.A = np.array(sys_discrete.A)
            self.B = np.array(sys_discrete.B)


        def update_state(self) -> None:
            self.x_init.value = self.dynamics.state

        def compute_action(
                self,
                goal: list[float],
        ) -> list[list[float]]:
        
            cost = 0
            constr = [self.x[:, 0] == self.x_init]

            for t in range(self.horizon):
                cost += cp.quad_form(goal - self.x[:, t], self.dynamics.Q) + cp.quad_form(self.u[:, t], self.dynamics.R)
                constr += [self.state_bounds[0] <= self.x[:, t], self.x[:, t] <= self.state_bounds[1]]
                if isinstance(self.dynamics,quadcopterDynamics):
                    constr += [self.x[:, t + 1] == self.A @ self.x[:, t] + self.B @ self.u[:, t]] 
                else:
                    constr += [self.x[:, t + 1] == self.A @ self.x[:, t] + self.B @ self.u[:, t]]

            cost += cp.quad_form(goal-self.x[:, self.horizon], self.dynamics.Q)  # End of trajectory error cost
            problem = cp.Problem(cp.Minimize(cost), constr)

            problem.solve(solver=cp.OSQP, warm_start=True)

            '''
            FOR FUTURE WORK, IF SOLUTION IS NONE 'is None' DOES NOT WORK
            MUST FIX, using if solution.size==1, but this is a bad fix
            solution data type is a numpy.ndarray but when set to None
            python does not recognize this...
            '''

            solution = np.transpose(np.array(self.u[:,0:self.valued_actions].value))
            if solution.size == 1:
                return solution
            elif solution is not None and isinstance(self.dynamics, quadcopterDynamics): #if quadcopter sim
                return np.array([s - np.array([self.dynamics.mass*self.dynamics.g, 0, 0, 0]) for s in solution]).squeeze()
            else:
                return solution.squeeze()
                
            

if __name__ == "__main__":

    xmin = np.array([-np.inf,  -np.inf,  -np.inf, -np.inf, -np.inf, -np.inf, -0.2, -0.2, -2*np.pi, -.25, -.25, -.25])
    xmax = np.array([np.inf,   np.inf,   np.inf,   np.inf,  np.inf, np.inf, 0.2,  0.2,   2*np.pi,  .25, .25,  .25])

    ymin = np.array([-20,-5,-5,-5])
    ymax = np.array([20,5,5,5])

    kwargs = {
        'upper_state_bounds' : xmax,
        'lower_state_bounds' : xmin,
        'horizon' : 20,
        'valued_actions' : 1,
    }