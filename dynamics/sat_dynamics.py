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
        euler: list[float] = np.array([0.0, 0.0, 0.0]), #initial euler angles
        #quat: list[float] = np.array([1.0, 0.0, 0.0, 0.0]), #initial orientation of body to cf
        omega: list[float] = np.array([0.0, 0.0, 0.0]), #angular velocity vector
        cf: list[list[float]] = np.array([
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]), #coordinate frame of simulation in reference to LVLH frame
        initial_orbit: Optional[Dict[str, Any]] = None,
        initial_state_data: Optional[Dict[str, Any]] = None,
        spacecraft_data: Optional[Dict[str, Any]] = None,
        max_control: list[float] = [0.3,0.3,0.3,0.1,0.1,0.1,1000,1000,1000],
    ):
        
        super().__init__(
            timestep=timestep,
            horizon=horizon,
            pos=pos,
            vel=vel,
            euler=euler,
            omega=omega,
            cf=cf,
        )

        if initial_orbit is None:
            print('Error: no orbit data presented')
            exit()
        elif initial_orbit.keys() >= {'pos0','vel0'}:
            self.orbit = Orbit.from_vectors(Earth, initial_orbit['pos0'], initial_orbit['vel0'])
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
        self.max_control = np.array(max_control)


        self.initialize_state()
        self.initialize_control()
        self.initialize_state_matrix()
        self.initialize_control_matrix()


    '''
    MPC SUPPORT FUNCTIONS

    A,B,C,D,Q,R are all solely to be used by the Linear-Quadratic MPC
    A and B are the control and state matrices for the simplified
    linear dyanmics of a quadcopter (from same paper). 

    convert_input transforms the motor command speeds to
    forces and torques using support fuctions from the 
    optomizer package used in the MPC

    MPC equations are presented as:

    dx/dt = Ax + Bu
    y = Cx + Du
    cost = xQx + uDu

    They assume ONLY small deviations in euler angles from 0 (~5deg)
    '''

    @property
    def A(self):
        # Linear state transition matrix
        A = np.zeros((self.state.size, self.state.size))
        A[0][3] = 1
        A[1][4] = 1
        A[2][5] = 1
        A[3][0] = 3*self.n**2
        A[3][4] = 2*self.n
        A[4][3] = -2*self.n
        A[5][2] = -(self.n**2)
        A[6][8] = self.n
        A[6][9] = 1
        A[7][10] = 1
        A[8][6] = -self.n
        A[8][11] = 1
        A[9][11] = -(self.spacecraft_data['J_sc'][1]-self.spacecraft_data['J_sc'][2])*self.n/self.spacecraft_data['J_sc'][0]
        A[9][14] = self.n*self.spacecraft_data['alpha'][2]/self.spacecraft_data['J_sc'][0]
        A[11][9] = -(self.spacecraft_data['J_sc'][0]-self.spacecraft_data['J_sc'][1])*self.n/self.spacecraft_data['J_sc'][2]
        A[11][12] = self.n*self.spacecraft_data['alpha'][0]/self.spacecraft_data['J_sc'][2]
        return A
    
    @property
    def B(self):
        # Control matrix
        B = np.zeros((self.state.size,self.control.size))
        B[3][0] = 1/self.mass
        B[4][1] = 1/self.mass
        B[5][2] = 1/self.mass
        B[9][3] = -self.spacecraft_data['alpha'][0]/self.spacecraft_data['J_sc'][0]
        B[9][6] = 1/self.spacecraft_data['J_sc'][0]
        B[10][4] = -self.spacecraft_data['alpha'][1]/self.spacecraft_data['J_sc'][1]
        B[10][7] = 1/self.spacecraft_data['J_sc'][1]
        B[11][5] = -self.spacecraft_data['alpha'][2]/self.spacecraft_data['J_sc'][2]
        B[11][8] = 1/self.spacecraft_data['J_sc'][2]
        B[12][3] = 1
        B[13][4] = 1
        B[14][5] = 1
        return B
    
    @property
    def C(self):
        C = np.eye(self.state.size)
        return C

    @property
    def D(self):
        D = np.zeros((self.state.size, self.control.size))
        return D
    
    @property
    def Q(self):
        # State cost
        Q = np.eye(self.state.size)
        Q[5, 5] = 5.  # z vel
        Q[3, 3] = 5.  # x vel
        Q[0, 0] = 10.  # x pos
        Q[1, 1] = 10.  # y pos
        Q[2, 2] = 10.  # z pos
        return Q

    @property
    def R(self):
        # Actuator cost
        R = np.eye(self.control.size)*.1
        R[6, 6] = 0.001
        R[7, 7] = 0.001
        R[8, 8] = 0.001
        return R

    '''
    END OF MPC SUPPORT FUNCTIONS
    '''


    def initialize_state(self) -> None:
        super().initialize_state()
        if self.initial_state_data.keys() >= {'momentum_wheel_vel'}:
            self.state = np.concatenate((self.state,self.initial_state_data['momentum_wheel_vel']), axis=None)
        else:
            self.state = np.concatenate(self.state,np.array([0, 0, 0]))
            print('No momentum wheel velocities in initial state data, setting to 0')
        
         
    def reset_state(self) -> None:
        super().reset_state()
        self.initialize_state()

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
        self.control_matrix[12][6] = 1
        self.control_matrix[13][7] = 1
        self.control_matrix[14][8] = 1
        '''
        numerator = (np.exp(self.state_matrix*self.timestep)-np.identity(self.state.size))                       
        control_matrix_integrated = np.nan_to_num( numerator / np.where(self.state_matrix==0, np.nan, self.state_matrix))
        self.control_matrix_discretized = np.dot(control_matrix_integrated, self.control_matrix) 
        self.state_matrix_discretized = np.nan_to_num(self.state_matrix_discretized)
        '''
           

    def set_control(self, control) -> None:
        control = np.clip(control, -self.max_control, self.max_control)
        for i in range(self.control.size):
            self.control[i] = control[i]

    def compute_derivatives(self, state, t) -> list[float]:

        '''
        rotation_axis_matrix = np.array([
            [0, -self.omega[0], -self.omega[1], -self.omega[2]],
            [self.omega[0], 0, self.omega[2], -self.omega[1]],
            [self.omega[1], -self.omega[2], 0, self.omega[0]],
            [self.omega[2], self.omega[1], -self.omega[0], 0]
        ]) 
        new_quat = np.dot( (np.identity(4)+(0.5*self.timestep*rotation_axis_matrix)), self.quat) #calculate new quaternions
        self.quat = new_quat
        '''
        dxdt = np.matmul(self.state_matrix,state) + np.squeeze(np.matmul(self.control_matrix,self.control))
        return dxdt
    
    def forward_step(self) -> list[float]:
        timerange = np.arange(self.time, self.time+(self.timestep*self.horizon), self.timestep)
        sol = scipy.integrate.odeint(
            self.compute_derivatives,
            self.state,
            timerange,
        )
        self.time += self.timestep*self.horizon
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