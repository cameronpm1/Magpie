import sys
sys.path.insert(1, 'c:/Users/Cameron Mehlman/Documents/Magpie')

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
from util.util import line_to_point
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
            goal_tolerance: float = 0.0001,
            time_tolerance: float = 300,
            perturbation_force: Optional[float] = None
    ):
        
        self.main_object = main_object
        self.path_planner = path_planner
        self.point_cloud_radius = point_cloud_radius
        self.point_cloud_size = point_cloud_size
        self.path_point_tolerance = path_point_tolerance
        self.goal_tolerance = goal_tolerance
        self.time_tolerance = time_tolerance
        self.obstacles = []
        self.perturbation_force = perturbation_force
        
        self.point_cloud_plot = None
        self.point_cloud = None
        self.dynamic_obstacles = False

        kwargs['dynamics'] = main_object.dynamics
        self.control_method = getattr(self,control_method)(**kwargs)
        self.control_method.update_state()

        self.time = 0
        self.done = False
        self.reward = 0
        self.current_path = None
        self.expected_time = None
        
        '''
        TO DO: the mpc controller has bad velocity control, as a result
        we cannot predict vehicle speed until it is already moving,
        Thus, we must recompute a trajectory after the first
        few timesteps
        '''
        self.first_goal = True #has first goal been reached?

    def step(self) -> tuple[list[float], list[float], bool, int]:
        self.collision_check()
        action = self.compute_next_action()
        self.main_object.dynamics.set_control(action)
        self.main_object.step()
        self.control_method.update_state()
        for obstacle in self.obstacles:
            if isinstance(obstacle,dynamicObject):
                if self.perturbation_force is not None:
                    prob = 0.1
                    a = np.random.random()
                    if a <= prob:
                        pforce = np.random.random((3,))*2*self.perturbation_force - self.perturbation_force
                        control = np.concatenate((pforce,np.zeros((obstacle.dynamics.control.size-3,))))
                        obstacle.dynamics.set_control(control)
                    else:
                        obstacle.dynamics.set_control(np.zeros((obstacle.dynamics.control.size,)))
                obstacle.step()
        if self.dynamic_obstacles:
            self.update_point_cloud(propagate=False)
        self.check_reached_path_point()
        if self.current_path.size == 0:
            self.first_goal = True
            self.get_new_path()
        self.time += 1
        return self.main_object.dynamics.state, action, self.done, self.reward
        
    def get_distance_to_goal(self) -> None:
        return np.linalg.norm(self.path_planner.goal[0:3]-self.main_object.dynamics.get_pos())
    
    def set_perturbation_force(
            self, 
            pforce : float,
    ) -> None:
        self.perturbation_force = pforce

    def reset(self) -> None:
        self.time = 0
        self.done = False
        self.reward = 0
        self.main_object.dynamics.reset_state()
        self.update_point_cloud()
        self.get_new_path()

    def compute_next_action(self) -> list[float]:
        next_action = self.control_method.compute_action(goal=self.current_path[0])
        return next_action
    
    def compute_schedule(
            self,
            speed: float,
    ) -> None:
        time = []
        time.append(np.linalg.norm(self.current_path[0][0:3]-self.main_object.dynamics.get_pos())/speed)
        for i in range(len(self.current_path)-1):
            time.append(time[-1] + np.linalg.norm(self.current_path[i+1][0:3]-self.current_path[i][0:3])/speed)
        self.expected_time = time


    def check_reached_path_point(self) -> None:
        if np.linalg.norm(self.current_path[0][0:3]-self.main_object.dynamics.state[0:3]) < self.path_point_tolerance:
            if self.first_goal:
                self.compute_schedule(self.main_object.dynamics.get_speed())
                self.first_goal = False
                self.get_new_path()
            #print('['+str(self.current_path[0][0])+','+str(self.current_path[0][1])+','+str(self.current_path[0][2])+'],')
            self.current_path = self.current_path[1:]
            if self.current_path.size > 0:
                self.compute_schedule(self.main_object.dynamics.get_speed())
                point_cloud = self.generate_proximal_point_cloud()
                self.path_planner.update_point_cloud(point_cloud=point_cloud)
                safe = self.path_planner.check_goal_safety(goals=self.current_path,state=self.main_object.dynamics.state[0:3])
                if not safe:
                    print('Obstacle has become too close to path, computing new path')
                    self.current_path = np.array([])

    def get_new_path(self) -> None:
        if self.goal_check():
            #print('Goal state achieved')
            self.done = True
            self.reward = 1
        self.update_point_cloud(propagate=True)
        if self.point_cloud is None:
            point_cloud = self.point_cloud
        else:
            point_cloud = self.generate_proximal_point_cloud()
        new_path = self.path_planner.compute_desired_path(state=self.main_object.dynamics.state, point_cloud=point_cloud)
        self.current_path = new_path

    def add_obstacle(
            self,
            obstacle: Union[Type[dynamicObject],Type[staticObject]],
    ) -> None:
        
        obstacle.update_points()
        self.obstacles.append(obstacle)
        if not self.dynamic_obstacles:
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
        '''
        package data for GUI
        return:
            main_object points and lines
            current goal point
            point cloud data ('if available')
            final goal point
        '''
        objects = {}
        objects['points'] = copy.deepcopy(self.main_object.temp_mesh['points'])
        objects['lines'] = copy.deepcopy(self.main_object.temp_mesh['lines'])
        objects['goal'] = self.current_path
        if self.point_cloud_plot is not None:
            objects['point cloud'] = self.point_cloud_plot
        objects['final goal'] = self.path_planner.goal[0:3]
        return objects
    
    def generate_proximal_point_cloud(self) -> list[list[float]]:
        '''
        return all points in env within point_cloud_radius 
        returns points location in relation to spacecraft
        '''

        if self.point_cloud is None:
            return None

        location = self.main_object.dynamics.get_pos()

        cloud = []

        for point in self.point_cloud:
                new_point = point-location
                if np.linalg.norm(new_point) < self.point_cloud_radius:
                    cloud.append(new_point)
 
        if len(cloud) == 0:
            return None
        
        return cloud
    
    def update_point_cloud(
            self,
            propagate: bool = True,
        ) -> None:
        self.point_cloud = None
        if len(self.obstacles) > 0:
            object_cloud_size = int(self.point_cloud_size/len(self.obstacles))
        for obstacle in self.obstacles:
            local_point_cloud = np.array(obstacle.point_cloud_from_mesh(n=object_cloud_size))
            if not self.first_goal and propagate:
                local_point_cloud = self.propagate_point_cloud(obstacle, local_point_cloud)
            if self.point_cloud is None:
                self.point_cloud = local_point_cloud
            else:
                self.point_cloud = np.concatenate((self.point_cloud,local_point_cloud))

        #dont plot point clouds greater than 2000 points
        self.point_cloud_plot = copy.deepcopy(self.point_cloud)
        '''
        if self.point_cloud_size > 2000:
            idx = np.random.randint(self.point_cloud_size, size=2000)
            self.point_cloud_plot = self.point_cloud[idx,:]
        else:
            self.point_cloud_plot = self.point_cloud
        '''

    def propagate_point_cloud(
            self,
            obstacle: Type[dynamicObject],
            point_cloud: list[list[float]],
    ) -> list[list[float]]:
        '''
            calculate the point cloud location at each timestep in the future
            for x timesteps and return combined point cloud data. ONLY based
            on obstacles current velocity

            propagate_timesteps: calculate for x timesteps
            obs_vel: velocity of the obstacle
            avg_std: the average of the stds of the obstacles point cloud dist.
            timestep: given obs_vel the amount of time it will take to travel 1 avg_std
        '''

        obs_pos = obstacle.dynamics.get_pos()
        obs_vel = obstacle.dynamics.get_vel()

        obs_speed = obstacle.dynamics.get_speed()

        new_cloud = point_cloud

        for i,goal in enumerate(self.current_path):
            rel_goal = goal[0:3]-obs_pos
            min_distance, dist_traveled = line_to_point(line = obs_vel, point = rel_goal)
            if dist_traveled < 0:
                continue
            else:
                if min_distance < self.path_planner.algorithm.min_distance:
                    time = dist_traveled/obs_speed
                    multiplier = 1/6 #how far to propogate (fraction of total distance between object and impact)
                    if abs(time-self.expected_time[i]) < self.time_tolerance:
                        print('Potential obstacle collision detected, propogating obstacle location')
                        std = 0
                        for point in point_cloud:
                            rel_point = point-obs_pos
                            std += (np.dot(rel_point,obs_vel)/np.linalg.norm(obs_vel))**2
                        std /= len(point_cloud)

                        vel_step = obs_vel/obs_speed * std
                        propogations = np.ceil((dist_traveled*multiplier + self.path_planner.algorithm.min_distance)/std)
                        forward_step = (obs_vel/obs_speed * dist_traveled*multiplier)
                        new_cloud += forward_step
                        for j in range(int(propogations)):
                            prop_obs = (point_cloud - obs_pos)*(j/propogations*3+1) + obs_pos + (j+1)*vel_step + forward_step
                            new_cloud = np.concatenate((new_cloud,prop_obs))
                        return new_cloud
        
        return new_cloud
        '''
        obs_vel = obstacle.dynamics.get_vel()
        propogated_cloud = []
        avg_std = np.average(np.std(point_cloud,axis=0))
        timestep = avg_std/np.linalg.norm(obs_vel)

        for point in point_cloud:
            for i in range(propagate_timesteps):
                propogated_cloud.append(obs_vel*timestep*i + point)
        return propogated_cloud
        '''


    def collision_check(self) -> None:
        if self.point_cloud is None:
            pass
        else:
            location = self.main_object.dynamics.get_pos()

            for point in self.point_cloud:
                new_point = point-location
                if np.linalg.norm(new_point) < self.path_planner.algorithm.histogram.distance_tol:
                    self.done = True
    
    def goal_check(self):
        if np.linalg.norm(self.path_planner.goal-self.main_object.dynamics.state) < self.goal_tolerance:
            return True
        return False

    def set_new_path(self, new_path: list[list[float]]) -> None:
        '''
        set path for testing purposes
        '''
        self.current_path = new_path

    class PID():
        def __init__(self) -> None:
            pass

        def compute_action(self) -> list[float]:
            '''
            return next action for satellite
            satellite control has 9 variables: Fx, Fy, Fz, Mx, My, Mz, Nx, Ny, Nz 
            (in that order)
            Nx,Ny,Nz are the acceleration of the momentum wheels
            there are 3 momentum wheels, each aligned w/ the respective axis
            of the satelite
            '''
            pass
    
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

            self.state_bounds = [np.array(lower_state_bounds),np.array(upper_state_bounds)]
            self.control_bounds = [np.array(lower_control_bounds),np.array(upper_control_bounds)]

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
    

    


