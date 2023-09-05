import numpy as np
import pyvista as pv

from astropy import units as u

from envs.obstacle_avoidance_env import obstacleAvoidanceEnv
from dynamics.dynamic_object import dynamicObject
from dynamics.static_object import staticObject
from dynamics.quad_dynamics import quadcopterDynamics
from dynamics.sat_dynamics import satelliteDynamics
from trajectory_planning.path_planner import pathPlanner
from envs.gui import Renderer

def correct_orbit_units(dynamics):
    orbit_params = {
        'a' : dynamics['a'] << u.km,
        'ecc' : dynamics['ecc'] << u.one,
        'inc' : dynamics['inc'] << u.deg,
        'raan' : dynamics['raan'] << u.deg,
        'argp' : dynamics['argp'] << u.deg,
        'nu' : dynamics['nu'] << u.deg,
    }
    return orbit_params

def make_env(cfg):
    
    orbit_params = correct_orbit_units(cfg['satellite']['dynamics']['initial_orbit'])

    dynamics = satelliteDynamics(
        timestep = cfg['satellite']['dynamics']['timestep'],
        horizon = cfg['satellite']['dynamics']['horizon'],
        pos = np.array(cfg['satellite']['dynamics']['pos']),
        initial_orbit = orbit_params,
        initial_state_data = cfg['satellite']['dynamics']['initial_state_data'],
        spacecraft_data = cfg['satellite']['dynamics']['spacecraft_data']
    )

    satellite = dynamicObject(
        dynamics = dynamics,
        mesh = {'points':np.array(cfg['satellite']['mesh']['points']),'lines':np.array(cfg['satellite']['mesh']['lines'])},
        name = cfg['satellite']['name'],
        pos = np.array(cfg['satellite']['dynamics']['pos']),
    )

    path_planner = pathPlanner(
        goal_state = cfg['path_planner']['goal_state'],
        path_planning_algorithm = cfg['path_planner']['path_planning_algorithm'],
        kwargs = cfg['path_planner']['kwargs'],
        max_distance = cfg['path_planner']['max_distance'],
        interpolation_method = cfg['path_planner']['interpolation_method'],
    )

    kwargs = {}

    for kwarg in cfg['env']['kwargs'].keys():
        kwargs[kwarg] = cfg['env']['kwargs'][kwarg]

    env = obstacleAvoidanceEnv(
        main_object = satellite,
        path_planner = path_planner,
        point_cloud_size = cfg['env']['point_cloud_size'],
        path_point_tolerance = cfg['env']['path_point_tolerance'],
        point_cloud_radius = cfg['env']['point_cloud_radius'],
        control_method = cfg['env']['control_method'],
        kwargs = kwargs,
    )

    '''
    for obstacle in cfg['obstacles']:
        stl = pv.read(cfg['obstacles'][obstacle]['stl'])
        stl.points *= cfg['obstacles'][obstacle]['stl_scale']

        obs_dynamics = satelliteDynamics(
            timestep = cfg['satellite']['dynamics']['timestep'],
            horizon = cfg['satellite']['dynamics']['horizon'],
            pos = np.array(cfg['obstacles'][obstacle]['pos']),
            vel = np.array(cfg['obstacles'][obstacle]['vel']),
            initial_orbit = orbit_params,
            initial_state_data = cfg['satellite']['dynamics']['initial_state_data'],
            spacecraft_data = cfg['satellite']['dynamics']['spacecraft_data']
        )

        temp_obstacle = dynamicObject(
            dynamics = obs_dynamics, 
            mesh = stl,
            name = cfg['obstacles'][obstacle]['name'], 
            pos = cfg['obstacles'][obstacle]['pos'])
        
        env.add_obstacle(obstacle=temp_obstacle)
    '''

    return env
    
    