import sys
sys.path.insert(1, 'c:/Users/Cameron Mehlman/Documents/Magpie')

import numpy as np
import time

from dynamics.quad_dynamics import quadcopterDynamics
from dynamics.sat_dynamics import satelliteDynamics
from dynamics.dynamic_object import dynamicObject
from controllers.mpc import MPC
from envs.gui import Renderer
from envs.magpie_env import magpieEnv
from unsupported.quadcopter import quadcopter
from dynamics.static_object import staticObject
from envs.obstacle_avoidance_env import obstacleAvoidanceEnv

def test_quad_gui():
    renderer = Renderer(
        xlim = [-12,12],
        ylim = [0,20],
        zlim = [-12,12],
        vista = False,
    )
    time.sleep(1)

    '''
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
    '''

    dynamics = satelliteDynamics(
        timestep = 3,
        horizon = 5,
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
        dynamics=satelliteDynamics,
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

    '''
    drone = quadcopter(
        dynamics = dynamics,
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
            ]),
        },
        name = 'drone'
    )
    '''

    env = obstacleAvoidanceEnv(
        main_object=quadcopter,
    )

    xmin = np.array([-np.inf,  -np.inf,  -np.inf, -np.inf, -np.inf, -np.inf, -0.2, -0.2, -2*np.pi, -.25, -.25, -.25])
    xmax = np.array([np.inf,   np.inf,   np.inf,   np.inf,  np.inf, np.inf, 0.2,  0.2,   2*np.pi,  .25, .25,  .25])

    ymin = np.array([-1,-1,-1,-0.5,-0.5,-0.5,-100,-100,-100])
    ymax = np.array([1,1,1,0.5,0.5,0.5,100,100,100])

    mpc = MPC(
        dynamics=dynamics, 
        horizon = 30,
        lower_state_bounds=xmin, 
        upper_state_bounds=xmax, 
        lower_control_bounds=ymin,
        upper_control_bounds=ymax,
        valued_actions=1    
    )

    mpc.update_state()

    goals = [[0, 2, 2, 0,0,0,0,0,0,0,0,0],
             [0, 2, 1, 0,0,0,0,0,0,0,0,0],
             [1, 2, 1, 0,0,0,0,0,0,0,0,0],
             [2, 2, 1, 0,0,0,0,0,0,0,0,0],
             [3, 2, 1, 0,0,0,0,0,0,0,0,0]]

    j = 0
    goal = goals[j]
    plot_goal = 1
    

    renderer.plot(env.get_object_data())
    env.main_object.temp_mesh['goal'] = goal[0:3]
    
    

    for i in range(300):
        if plot_goal == 0:
            env.main_object.temp_mesh['goal'] = goal[0:3]
            plot_goal=1

        actions = mpc.compute_action(goal=goal)
        env.main_object.dynamics.set_control(actions,motor_velocity=False)
        env.step()
        mpc.update_state()
        #print(env.main_object.dynamics.state)
        renderer.plot(env.get_object_data())
        diff = np.linalg.norm(env.main_object.dynamics.state - goal)
        if diff < 0.1 and j < len(goals)-1:
            j += 1
            goal = goals[j]
            plot_goal = 0
            mpc.initialize_optimization_parameters()
            mpc.update_state()


if __name__ == "__main__":

    test_quad_gui()