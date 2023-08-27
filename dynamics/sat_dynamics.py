import numpy as np
import scipy
from typing import Any, Dict, Optional
from matplotlib import pyplot as plt

from astropy import units as u

from poliastro.bodies import Earth, Mars, Sun
from poliastro.twobody import Orbit

from dynamics.base_dynamics import baseDynamics
from util.util import *

'''
dynamics simulation for a satellite in orbit
dynamics equations come from:
https://www.sciencedirect.com/science/article/pii/S1270963817311756
units in m, s, kg
'''


class satelliteDynamics(baseDynamics):

    def __init__(
        self,
        timestep: int = 1,
        horizon: int = 10,
        pos: list[float] = np.array([0.0, 0.0, 0.0]), #initial position w/ respect to orbit
        vel: list[float] = np.array([0.0, 0.0, 0.0]), #initial velocity w/ respect to orbit
        quat: list[float] = np.array([1.0, 0.0, 0.0, 0.0]), #initial orientation of body to cf
        omega: list[float] = np.array([0.0, 0.0, 0.0]), #angular velocity vector
        cf: list[list[float]] = np.array([
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]), #coordinate frame of simulation in reference to LVLH frame
        initial_orbit: Optional[Dict[str, Any]] = None,
        initial_state_data: Optional[Dict[str, Any]] = None,
        spacecraft_data: Optional[Dict[str, Any]] = None,
    ):
        
        super().__init__(
            timestep = timestep,
            horizon = horizon,
            pos = pos,
            vel = vel,
            quat = quat,
            omega = omega,
            cf = cf,
        )

        if initial_orbit is None:
            print('Error: no orbit data presented')
            exit()
        elif initial_orbit.keys() >= {'pos0','vel0'}:
            self.orbit = Orbit.from_vectors(Earth, initial_orbit[pos0], initial_orbit[vel0])
        elif initial_orbit.keys() >= {'a', 'ecc', 'inc', 'raan', 'argp', 'nu'}:
            self.orbit = Orbit.from_classical(
                Earth, 
                initial_orbit['a'], 
                initial_orbit['ecc'], 
                initial_orbit['inc'], 
                initial_orbit['raan'], 
                initial_orbit['argp'], 
                initial_orbit['nu']
            )
        else:
            print('Error: insufficient orbit data presented')
            exit()

        self.mu = 3.986004418e14
        self.n = np.sqrt(self.mu/((self.orbit.a.value*1000)**3))
        self.mass = spacecraft_data['mass']

        self.initial_state_data = initial_state_data
        self.spacecraft_data = spacecraft_data
        self.state_matrix_discretized = None
        self.control_matrix_discretized = None


        self.initialize_state()
        self.initialize_control()
        self.initialize_state_matrix()
        self.initialize_control_matrix()


    def initialize_state(self) -> None:
        super().initialize_state()
        if self.initial_state_data.keys() >= {'momentum_wheel_vel'}:
            self.state = np.concatenate((self.state,self.initial_state_data['momentum_wheel_vel']), axis=None)
        else:
            self.state = np.concatenate(self.state,np.array([0, 0, 0]))
            print('No momentum wheel velocities in initial state data, setting to 0')
         

    def initialize_control(self) -> None:
        self.control = np.zeros((9,1))

    def initialize_state_matrix(self) -> None:
        super().initialize_state_matrix()
        self.state_matrix[0][3] = 1
        self.state_matrix[1][4] = 1
        self.state_matrix[2][5] = 1
        self.state_matrix[3][0] = 3*self.n**2
        self.state_matrix[3][4] = 2*self.n
        self.state_matrix[4][3] = -2*self.n
        self.state_matrix[5][2] = -(self.n**2)
        self.state_matrix[6][8] = self.n
        self.state_matrix[6][9] = 1
        self.state_matrix[7][10] = 1
        self.state_matrix[8][6] = -self.n
        self.state_matrix[8][11] = 1
        self.state_matrix[9][11] = -(self.spacecraft_data['J_sc'][1]-self.spacecraft_data['J_sc'][2])*self.n/self.spacecraft_data['J_sc'][0]
        self.state_matrix[9][14] = self.n*self.spacecraft_data['alpha'][2]/self.spacecraft_data['J_sc'][0]
        self.state_matrix[11][9] = -(self.spacecraft_data['J_sc'][0]-self.spacecraft_data['J_sc'][1])*self.n/self.spacecraft_data['J_sc'][2]
        self.state_matrix[11][12] = self.n*self.spacecraft_data['alpha'][0]/self.spacecraft_data['J_sc'][2]
        '''
        self.state_matrix_discretized = np.exp(self.state_matrix*self.timestep)
        '''

    def initialize_control_matrix(self) -> None:
        self.control_matrix = np.zeros((self.state.size,self.control.size))
        self.control_matrix[3][0] = 1/self.mass
        self.control_matrix[4][1] = 1/self.mass
        self.control_matrix[5][2] = 1/self.mass
        self.control_matrix[9][3] = -self.spacecraft_data['alpha'][0]/self.spacecraft_data['J_sc'][0]
        self.control_matrix[9][6] = 1/self.spacecraft_data['J_sc'][0]
        self.control_matrix[10][4] = -self.spacecraft_data['alpha'][1]/self.spacecraft_data['J_sc'][1]
        self.control_matrix[10][7] = 1/self.spacecraft_data['J_sc'][1]
        self.control_matrix[11][5] = -self.spacecraft_data['alpha'][2]/self.spacecraft_data['J_sc'][2]
        self.control_matrix[11][8] = 1/self.spacecraft_data['J_sc'][2]
        self.control_matrix[12][3] = 1
        self.control_matrix[13][4] = 1
        self.control_matrix[14][5] = 1
        '''
        numerator = (np.exp(self.state_matrix*self.timestep)-np.identity(self.state.size))                       
        control_matrix_integrated = np.nan_to_num( numerator / np.where(self.state_matrix==0, np.nan, self.state_matrix))
        self.control_matrix_discretized = np.dot(control_matrix_integrated, self.control_matrix) 
        self.state_matrix_discretized = np.nan_to_num(self.state_matrix_discretized)
        '''
           

    def set_control(self, control) -> None:
        for i in range(self.control.size):
            self.control[i] = control[i]

    def compute_derivatives(self, state, t) -> list[float]:
        '''
        new_state = np.dot(self.state_matrix_discretized,self.state) + np.dot(self.control_matrix_discretized,self.control) #calculate new state
        '''

        rotation_axis_matrix = np.array([
            [0, -self.omega[0], -self.omega[1], -self.omega[2]],
            [self.omega[0], 0, self.omega[2], -self.omega[1]],
            [self.omega[1], -self.omega[2], 0, self.omega[0]],
            [self.omega[2], self.omega[1], -self.omega[0], 0]
        ]) 
        new_quat = np.dot( (np.identity(4)+(0.5*self.timestep*rotation_axis_matrix)), self.quat) #calculate new quaternions
        self.quat = new_quat
        dxdt = np.matmul(self.state_matrix,state) + np.squeeze(np.matmul(self.control_matrix,self.control))
        return dxdt
    
    def forward_step(self) -> list[float]:
        timerange = np.arange(self.time, self.time+self.timestep*self.horizon, self.timestep)
        sol = scipy.integrate.odeint(
            self.compute_derivatives,
            self.state,
            timerange,
        )
        self.state = sol[-1]
        return sol
            
if __name__ == '__main__':
    sd1 = satelliteDynamics(
        timestep = 60,
        pos = np.array([0, 0, 0]),
        omega = np.array([0, 0, 0.1]),
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
    #print(np.matmul(sd1.state_matrix,sd1.state))
    #print(sd1.state)
    prop = sd1.forward_step()
    x = prop[:,0]
    y = prop[:,1]
    z = prop[:,2]
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(x, y, z)
    print(x,y,z)
    print(prop)
    plt.show()
    #print(sd1.forward_dynamics())