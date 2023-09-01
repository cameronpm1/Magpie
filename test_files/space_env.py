import sys
sys.path.insert(1, 'c:/Users/Cameron Mehlman/Documents/Magpie')

import numpy as np
import time
import pyvista as pv

from astropy import units as u

from envs.obstacle_avoidance_env import obstacleAvoidanceEnv
from dynamics.dynamic_object import dynamicObject
from dynamics.static_object import staticObject
from dynamics.quad_dynamics import quadcopterDynamics
from dynamics.sat_dynamics import satelliteDynamics
from trajectory_planning.path_planner import pathPlanner
from envs.gui import Renderer


#def run_env(env, renderer, n):

    

    

if __name__ == "__main__":

    radius = 2

    renderer = Renderer(
        xlim = [-20,20],
        ylim = [-20,20],
        zlim = [-10,10],
        vista = False,
    )
    time.sleep(1)

    dynamics = satelliteDynamics(
        timestep = 3,
        horizon = 2,
        pos = np.array([0, 0, 0]),
        initial_orbit = {
            'a' : 35786 << u.km,
            'ecc' : 0.0 << u.one,
            'inc' : 1.85 << u.deg,
            'raan' : 49.562 << u.deg,
            'argp' : 286.537 << u.deg,
            'nu' : 0 << u.deg,
        },
        initial_state_data = {'momentum_wheel_vel' : np.array([0, 0, 0])},
        spacecraft_data = {
            'J_sc' : np.array([1.7e4, 2.7e4, 2.7e4]),
            'alpha' : np.array([0.8, 0.8, 0.8]),
            'mass' : 4000,
        },
    )

    satellite = dynamicObject(
        dynamics = dynamics,
        mesh = {
            'points' : np.array([
                [1.0,0.5,0.5],[-1.0,0.5,0.5],
                [1.0,0.5,-0.5],[-1.0,0.5,-0.5],
                [1.0,-0.5,0.5],[-1.0,-0.5,0.5],
                [1.0,-0.5,-0.5],[-1.0,-0.5,-0.5],
            ]),
            'lines' : np.array([
                [0,1],[0,2],[1,3],[2,3],
                [4,5],[4,6],[5,7],[6,7],
                [0,4],[1,5],[2,6],[3,7]
            ])
        }
    )


    
    pathPlanner = pathPlanner(
        goal_state = [4,4,2,0,0,0,0,0,0,0,0,0,0,0,0],
        path_planning_algorithm='VFH',
        kwargs={'radius':4,'min_obstacle_distance':2,'iterations':1,'distance_tolerance':1.2},
        max_distance=0.1,
        interpolation_method='linear'
    )
    

    xmin = np.array([-np.inf,  -np.inf,  -np.inf, -np.inf, -np.inf, -np.inf, -0.2, -0.2, -2*np.pi, -.25, -.25, -.25, -1000, -1000, -1000])
    xmax = np.array([np.inf,   np.inf,   np.inf,   np.inf,  np.inf, np.inf, 0.2,  0.2,   2*np.pi,  .25, .25,  .25, 1000, 1000, 1000])

    kwargs = {
        'upper_state_bounds' : xmax,
        'lower_state_bounds' : xmin,
        'horizon' : 20,
        'valued_actions' : 1,
    }

    env = obstacleAvoidanceEnv(
        main_object=satellite,
        path_planner=pathPlanner,
        control_method='MPC',
        kwargs=kwargs,
        point_cloud_size=100,
        path_point_tolerance=0.1,
        point_cloud_radius=10,
    )

    ball1 = pv.read('stl_files/ball.stl')
    ball2 = pv.read('stl_files/ball.stl')
    ball1.points /= 1000
    ball2.points /= 300

    debris1_dyn = satelliteDynamics(
        timestep = 3,
        horizon = 2,
        pos = np.array([-10, -10, 0]),
        vel = np.array([0, 0.001, 0]),
        initial_orbit = {
            'a' : 35786 << u.km,
            'ecc' : 0.0 << u.one,
            'inc' : 1.85 << u.deg,
            'raan' : 49.562 << u.deg,
            'argp' : 286.537 << u.deg,
            'nu' : 0 << u.deg,
        },
        initial_state_data = {'momentum_wheel_vel' : np.array([0, 0, 0])},
        spacecraft_data = {
            'J_sc' : np.array([1.7e4, 2.7e4, 2.7e4]),
            'alpha' : np.array([0.8, 0.8, 0.8]),
            'mass' : 4000,
        },
    )

    debris2_dyn = satelliteDynamics(
        timestep = 3,
        horizon = 2,
        pos = np.array([5, 5, 3]),
        vel = np.array([-0.0001, -0.0005, 0]),
        initial_orbit = {
            'a' : 35786 << u.km,
            'ecc' : 0.0 << u.one,
            'inc' : 1.85 << u.deg,
            'raan' : 49.562 << u.deg,
            'argp' : 286.537 << u.deg,
            'nu' : 0 << u.deg,
        },
        initial_state_data = {'momentum_wheel_vel' : np.array([0, 0, 0])},
        spacecraft_data = {
            'J_sc' : np.array([1.7e4, 2.7e4, 2.7e4]),
            'alpha' : np.array([0.8, 0.8, 0.8]),
            'mass' : 4000,
        },
    )

    debris1 = dynamicObject(dynamics=debris1_dyn, mesh=ball1,name='debris1', pos=[-10, -10, 0])
    debris2 = dynamicObject(dynamics=debris2_dyn, mesh=ball2,name='debris1', pos=[5, 5, 3])

    #env.add_obstacle(obstacle=debris1)
    #env.add_obstacle(obstacle=debris2)

    env.reset()

    renderer.plot(env.get_object_data())

    time.sleep(1)

    for i in range(2000):
        env.step()
        renderer.plot(env.get_object_data())