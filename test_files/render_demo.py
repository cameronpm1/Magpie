import numpy as np
import time
import pyvista as pv

from astropy import units as u

from envs.gui import Renderer
from envs.magpie_env import magpieEnv
from dynamics.dynamic_object import dynamicObject
from dynamics.sat_dynamics import satelliteDynamics
from unsupported.spacecraft import spacecraft
from dynamics.quad_dynamics import quadcopterDynamics
from unsupported.quadcopter import quadcopter



def test_sat_gui():
    renderer = Renderer()
    time.sleep(1)

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

    satellite = spacecraft(
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

    env = magpieEnv(
        main_object=satellite,
    )

    for i in range(500):
        if i == 25:
            env.main_object.dynamics.set_control([0,0.1,0,0,0,0,0,3,0])
            print('start')
        if i == 26:
            env.main_object.dynamics.set_control([0,0,0,0,0,0,0,0,0.0])
        env.step()
        renderer.plot(env.get_object_data())

def test_quad_gui():
    renderer = Renderer(
        xlim = [-5,5],
        ylim = [-5,5],
        zlim = [0,20],
        vista = True,
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
            ])
        }
    )
    '''

    drone = quadcopter(
        dynamics = dynamics,
        mesh = pv.read('quadcopter.stl'),
    )

    env = magpieEnv(
        main_object=drone,
    )

    env.main_object.dynamics.set_control(np.array([340,340,340,340]))

    renderer.plot(env.get_object_data())
    

    for i in range(300):
        
        if i == 60:
            env.main_object.dynamics.set_control([305,305,305,305])
        #if i == 56:
        #    env.object.dynamics.set_control([350,300,350,300])
        if i == 100:
            env.main_object.dynamics.set_control([322,322,322,322])
        if i == 250:
            env.main_object.dynamics.set_control([304,304,300,300])
        if i == 252:
            env.main_object.dynamics.set_control([300,300,304,304])
        if i == 254:
            env.main_object.dynamics.set_control([325,325,325,325])
        env.step()
        print(i)
        renderer.plot(env.get_object_data())


if __name__ == "__main__":

    test_quad_gui()

    
