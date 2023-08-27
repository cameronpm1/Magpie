import numpy as np

from util.util import euler_from_quaternion



class baseDynamics:

    def __init__(
        self,
        timestep: float = 1.0,
        horizon: int = 10,
        pos: list[float] = np.array([0.0, 0.0, 0.0]), #initial position
        vel: list[float] = np.array([0.0, 0.0, 0.0]), #initial velocity
        quat: list[float] = np.array([1.0, 0.0, 0.0, 0.0]), #initial orientation of body to cf
        omega: list[float] = np.array([0.0, 0.0, 0.0]), #angular velocity vector
        cf: list[list[float]] = np.array([[1.0, 0.0, 0.0],
                                          [0.0, 1.0, 0.0],
                                          [0.0, 0.0, 1.0],
                                          ]) #coordinate frame of simulation
        
    ):
        self.timestep = timestep
        self.horizon = horizon
        self.pos = pos
        self.vel = vel
        self.euler = np.zeros((3,))
        self.quat = quat
        self.omega = omega
        self.cf = cf
        self.time = 0
        self.state = None
        self.control = None
        self.state_matrix = None
        self.control_matrix = None

    def initialize_state(self) -> None:
        euler = euler_from_quaternion(self.quat)
        self.euler = euler
        state1 = np.concatenate((self.pos,self.vel), axis=None)   
        state2 = np.concatenate((euler,self.omega), axis=None) 
        self.state = np.concatenate((state1,state2), axis=None)


    def initialize_state_matrix(self) -> None:
        self.state_matrix = np.zeros((self.state.size,self.state.size))

    def get_euler(self) -> list[float]:
        return self.state[6:9]
    
    def get_pos(self) -> list[float]:
        return self.state[0:3]
    
    def get_vel(self) -> list[float]:
        return self.state[3:6]
    
    def get_omega(self) -> list[float]:
        return self.state[9:12]

    def forward_dynamics(self):
        pass

    def step(self):
        pass

        
