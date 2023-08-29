import numpy as np
import scipy
from typing import Any, Dict, Optional
from matplotlib import pyplot as plt
import cvxpy as cp

from dynamics.base_dynamics import baseDynamics
from util.util import *

'''
nonlinear dynamics simulation for a quadcopter
dynamics equations come from:
https://ieeexplore.ieee.org/abstract/document/7813499
constants for reference come from:
https://arxiv.org/ftp/arxiv/papers/2202/2202.07021.pdf
&
https://iopscience.iop.org/article/10.1088/1757-899X/1116/1/012095/pdf
for more accurate change in euler angles:
https://ieeexplore.ieee.org/abstract/document/8074380
units in m, s, kg
'''


class quadcopterDynamics(baseDynamics):

    def __init__(
        self,
        timestep: int = 1,
        horizon: int = 10,
        pos: list[float] = np.array([0.0, 0.0, 0.0]), #initial position w/ respect to orbit
        vel: list[float] = np.array([0.0, 0.0, 0.0]), #initial velocity w/ respect to orbit
        quat: list[float] = np.array([1.0, 0.0, 0.0, 0.0]), #initial orientation of body to cf
        omega: list[float] = np.array([0.0, 0.0, 0.0]), #angular velocity vector
        cf: list[list[float]] = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]), #coordinate frame of simulation in reference to LVLH frame
        coef_data: Optional[Dict[str, Any]] = None,
        quad_data: Optional[Dict[str, Any]] = None,
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

        if coef_data is None:
            print('Error: no coefficient data presented')
            exit()
        else:
            self.kf = coef_data['kf'] #thrust coefficient (Ns^2)
            self.km = coef_data['km'] #aerodynamic drag coefficient (Nms^2)

        if quad_data is None:
            print('Error: no quadcopter data presented')
            exit()
        else:
            self.l = quad_data['l'] #moment arm length
            self.mass = quad_data['mass'] #mass
            self.I = quad_data['I'] #moment of inertia array (kgm^2)
         
        self.g = -9.81 

        self.state_matrix_discretized = None
        self.control_matrix_discretized = None
        self.constant_matrix = None
        self.motor_constant_matrix = None

        self.initialize_state()
        self.initialize_control()
        self.initialize_constant()
        self.initialize_state_matrix()
        self.initialize_control_matrix()
        self.initialize_motor_constant_matrix()

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
        A[0, 3] = 1.
        A[1, 4] = 1.
        A[2, 5] = 1.
        A[3, 7] = -self.g
        A[4, 6] = self.g
        A[6, 9] = 1.
        A[7, 10] = 1.
        A[8, 11] = 1.
        return A
    
    @property
    def B(self):
        # Control matrix
        B = np.zeros((self.state.size, self.control.size))
        B[9, 1] = 1/self.I[0]
        B[10, 2] = 1/self.I[1]
        B[11, 3] = 1/self.I[2]
        B[5, 0] = 1/self.mass
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
        Q[6, 6] = 1
        Q[3, 3] = 100.  # z vel
        Q[4, 4] = 100.  # z vel
        Q[5, 5] = 100.  # z vel
        Q[0, 0] = 5.  # x pos
        Q[1, 1] = 5.  # y pos
        Q[2, 2] = 5.  # z pos
        return Q

    @property
    def R(self):
        # Actuator cost
        R = np.eye(self.control.size)*.001
        return R
    
    def convert_input(self, w):
        w = self.motor_constant_matrix@w**2
        return w

    
    '''
    END OF MPC SUPPORT FUNCTIONS
    '''

    def initialize_motor_constant_matrix(self) -> None:
        self.motor_constant_matrix = np.zeros((self.control.size,self.control.size))
        self.motor_constant_matrix[0,:] = self.kf
        self.motor_constant_matrix[1,:] = self.l*self.kf
        self.motor_constant_matrix[1][0] *= -1
        self.motor_constant_matrix[1][3] *= -1
        self.motor_constant_matrix[2,:] = self.l*self.kf
        self.motor_constant_matrix[1][2] *= -1
        self.motor_constant_matrix[1][3] *= -1
        self.motor_constant_matrix[2,:] = self.km
        self.motor_constant_matrix[1][1] *= -1
        self.motor_constant_matrix[1][3] *= -1

    def initialize_state(self) -> None:
        super().initialize_state()
         
    def initialize_constant(self) -> None:
        self.constant_matrix = np.zeros((self.state.size,))
        self.constant_matrix[5] = self.g

    def initialize_control(self) -> None:
        self.control = np.zeros((4,))

    def initialize_state_matrix(self) -> None:
        super().initialize_state_matrix()
        self.state_matrix[0][3] = 1
        self.state_matrix[1][4] = 1
        self.state_matrix[2][5] = 1
        self.state_matrix[6][9] = 1
        self.state_matrix[7][10] = 1
        self.state_matrix[8][11] = 1

    def initialize_control_matrix(self) -> None:
        self.control_matrix = np.zeros((self.state.size,self.control.size))
        self.control_matrix[3][0] = -1/self.mass
        self.control_matrix[4][0] = -1/self.mass
        self.control_matrix[5][0] = -1/self.mass
        self.control_matrix[9][1] = 1/self.I[0]
        self.control_matrix[10][2] = 1/self.I[1]
        self.control_matrix[11][3] = 1/self.I[2]

    def motor_vel_from_forces(
            self,
            F: list[float],
    ) -> Optional[list[float]]:
        if F.size != 4:
            return None

        A1 = F[0]/self.kf
        A2 = F[1]/(self.kf*self.l)
        A3 = F[2]/(self.kf*self.l)
        A4 = F[3]/self.km

        w1 = A1 - (A1+A2)/2 + (A4-A1+A2+A1+A3-A1)/4
        w2 = (A2+A1)/2 + (A3-A1)/2 - (A4-A1+A2+A1+A3-A1)/4
        w3 = (A3-A1)/-2 + (A4-A1+A2+A1+A3-A1)/4
        w4 = -(A4-A1+A2+A1+A3-A1)/4 

        w_sq = np.array([w1, w2, w3, w4])  

        if ((w_sq >= 0).sum() == w_sq.size):
            return np.array([w1, w2, w3, w4])
        else:
            return None

    def set_control(self, w, motor_velocity=False) -> None:
        #take motor velocities w1-w4, transform into system control
        # u1 = kf(w1 + w2 + w3 + w4)
        # u2 = lkf[(w2^2 + w3^2) - (w1^2+w4^2)]
        # u3 = lkf[(w1^2 + w2^2) - (w3^2+w4^2)]
        # u4 = km[(w1^2 + w3^2) - (w2^2+w4^2)]
        if w is None:
            pass
        if motor_velocity:
            w2 = np.square(w)

            #self.control = np.matmul(self.motor_constant_matrix,w2)

            self.control[0] = self.kf*(w2[0] + w2[1] + w2[2] + w2[3])
            self.control[1] = self.l*self.kf*( (w2[1]+w2[2]) - (w2[0]+w2[3]) )
            self.control[2] = self.l*self.kf*( (w2[0]+w2[1]) - (w2[2]+w2[3]) )
            self.control[3] = self.km*( (w2[0]+w2[2]) - (w2[1]+w2[3]) )
        else:
            if self.motor_vel_from_forces(w) is None:
                print('Error: infeasible motor thrust/torques input to dynamics, switching control to hover')
                self.control = np.array([15.46, 0, 0, 0])
            else:
                self.control = w


    def compute_derivatives(self, state, t) -> list[float]:
        '''
        new_state = np.dot(self.state_matrix_discretized,self.state) + np.dot(self.control_matrix_discretized,self.control) #calculate new state
        '''
        dcm_z = self.zyx_dcm_zaxis(state[6:9])/self.mass

        self.control_matrix[3][0] = dcm_z[0]
        self.control_matrix[4][0] = dcm_z[1]
        self.control_matrix[5][0] = dcm_z[2]

        self.state_matrix[9][7] = self.state[8]*(self.I[1] - self.I[2])/self.I[0]
        self.state_matrix[10][6] = self.state[8]*(self.I[2] - self.I[0])/self.I[1]
        self.state_matrix[11][6] = self.state[7]*(self.I[0] - self.I[1])/self.I[2]

        dxdt = np.matmul(self.state_matrix,state) + np.squeeze(np.matmul(self.control_matrix,self.control)) + self.constant_matrix

        # gravity check
        if state[2] < 0:
            dxdt[2] = 0
            dxdt[5] = 0

        return dxdt
    
    def forward_step(self) -> list[float]:
        '''
        forward step with Non-Linear dynamics of a quadcopter
        '''
        timerange = np.arange(self.time, self.time+self.timestep*(self.horizon+1), self.timestep)
        sol = scipy.integrate.odeint(
            self.compute_derivatives,
            self.state,
            timerange,
        )
        self.state = sol[-1]
        return sol
    
    def zyx_dcm_zaxis(self,E) -> list[float]:
        c1 = np.cos(E[0])
        s1 = np.sin(E[0])
        c2 = np.cos(E[1])
        s2 = np.sin(E[1])
        c3 = np.cos(E[2])
        s3 = np.sin(E[2])

        dcm_z = np.array([s3*s1+c3*c1*s2, c1*s3*s2-c3*s1, c2*c1])

        return dcm_z
    
if __name__ == '__main__':
    sd1 = quadcopterDynamics(
        timestep = 2,
        horizon = 2,
        pos = np.array([0, 0, 0]),
        omega = np.array([0, 0, 0]),
        coef_data = {'kf' : 3.7102e-5, 'km' : 7.6933e-7},
        quad_data = {
            'l' : 0.243,
            'mass' : 1.587,
            'I' : np.array([0.0213, 0.02217, 0.0282]),
        },
    )


    F = [10,1,1,1]

    print(sd1.motor_vel_from_forces(F))

    '''
    sd1.set_control(np.array([500,500,500,500]))
    prop = sd1.forward_step()
    x = prop[:,0]
    y = prop[:,1]
    z = prop[:,2]
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(x, y, z)
    #print(x,y,z)
    #print(prop)
    plt.show()
    #print(sd1.forward_dynamics())
    '''
