import sys
sys.path.insert(1, 'c:/Users/Cameron Mehlman/Documents/Magpie')

import numpy as np
import time
import pyvista as pv

from envs.obstacle_avoidance_env import obstacleAvoidanceEnv
from dynamics.dynamic_object import dynamicObject
from dynamics.static_object import staticObject
from dynamics.quad_dynamics import quadcopterDynamics
from trajectory_planning.path_planner import pathPlanner
from envs.gui import Renderer


#def run_env(env, renderer, n):

    

    

if __name__ == "__main__":

    radius = 2

    renderer = Renderer(
        xlim = [-5,5],
        ylim = [-5,5],
        zlim = [0,10],
        vista = False,
    )
    time.sleep(1)

    dynamics = quadcopterDynamics(
        timestep = 0.01,
        horizon = 10,
        pos = np.array([0, 0, 0]),
        omega = np.array([0, 0, 0]),
        coef_data = {'kf' : 3.7102e-5, 'km' : 7.6933e-7},
        quad_data = {
            'l' : 0.243,
            'mass' : 1.587,
            'I' : np.array([0.0213, 0.02217, 0.0282]),
        },
    )

    quadcopter = dynamicObject(
        dynamics=dynamics,
        mesh = {
            'points' : np.array([
                [0.03,0.25,0.0],[-.03,0.25,0.0],
                [0.03,-0.25,0.0],[-.03,-0.25,0.0],
                [0.25,0.03,0.0],[0.25,-0.03,0.0],
                [-0.25,0.03,0.0],[-0.25,-0.03,0.0],
            ]),
            'lines' : np.array([
                [0,1],[1,3],[3,2],[2,0],[0,3],[1,2],
                [4,5],[5,7],[7,6],[6,4],[4,7],[5,6],
            ])
        }
    )


    
    pathPlanner = pathPlanner(
        goal_state = [0,10,10,0,0,0,0,0,0,0,0,0],
        path_planning_algorithm='VFH',
        kwargs={'radius':2,'min_obstacle_distance':1,'iterations':1,},
        max_distance=0.3,
        interpolation_method='linear'
    )
    

    '''
    Keep in mind, the path planner will pick trajectories at most 1/4
    the euclidian distance of the radius
    '''
    '''
    pathPlanner = pathPlanner(goal_state=[0.2,10.5,3,0,0,0,0,0,0,0,0,0],
                       path_planning_algorithm='VFH',
                       interpolation_method='linear',
                       kwargs={'radius':4,
                               'iterations':1,
                               'layers':1, 
                               'angle_sections':30,
                               'distance_tolerance': 0.2,
                               'probability_tolerance': 0.05,
                               'min_obstacle_distance': 1,
                               },
                        n = 30,
                       )
    '''

    xmin = np.array([-np.inf,  -np.inf,  -np.inf, -np.inf, -np.inf, -np.inf, -0.2, -0.2, -2*np.pi, -.25, -.25, -.25])
    xmax = np.array([np.inf,   np.inf,   np.inf,   np.inf,  np.inf, np.inf, 0.2,  0.2,   2*np.pi,  .25, .25,  .25])

    kwargs = {
        'upper_state_bounds' : xmax,
        'lower_state_bounds' : xmin,
        'horizon' : 20,
        'valued_actions' : 1,
    }

    env = obstacleAvoidanceEnv(
        main_object=quadcopter,
        path_planner=pathPlanner,
        control_method='MPC',
        kwargs=kwargs,
        point_cloud_size=10000,
        path_point_tolerance=0.25,
        point_cloud_radius=10,
    )


    obstacle_course_mesh = pv.read('stl_files/obstacle_course2.stl')
    obstacle_course_mesh.points *= 1000 #fix scaling issue w/ solidworks and STL exporting
    obstacle_course = staticObject(mesh=obstacle_course_mesh,name='obstacle_course')

    env.add_obstacle(obstacle=obstacle_course)

    renderer.plot(env.get_object_data())

    time.sleep(1)

    for i in range(2000):
        env.step()
        renderer.plot(env.get_object_data())