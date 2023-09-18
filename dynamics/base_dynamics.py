import numpy as np

from util.util import euler_from_quaternion



class baseDynamics:

    def __init__(
        self,
        timestep: float = 1.0,
        horizon: int = 10,
        pos: list[float] = np.array([0.0, 0.0, 0.0]), #initial position
        vel: list[float] = np.array([0.0, 0.0, 0.0]), #initial velocity
        euler: list[float] = np.array([0.0, 0.0, 0.0]), #initial euler angles
        #quat: list[float] = np.array([1.0, 0.0, 0.0, 0.0]), #initial orientation of body to cf
        omega: list[float] = np.array([0.0, 0.0, 0.0]), #angular velocity vector
        cf: list[list[float]] = np.array([[1.0, 0.0, 0.0],
                                          [0.0, 1.0, 0.0],
                                          [0.0, 0.0, 1.0],
                                          ]) #coordinate frame of simulation
        
    ):
        self.timestep = timestep
        self.horizon = horizon
        self.pos = np.array(pos)
        self.vel = np.array(vel)
        self.euler = euler
        #self.quat = quat
        self.omega = np.array(omega)
        self.cf = cf
        self.time = 0
        self.state = None
        self.control = None
        self.state_matrix = None
        self.control_matrix = None

    def initialize_state(self) -> None:
        state1 = np.concatenate((self.pos,self.vel), axis=None)   
        state2 = np.concatenate((self.euler,self.omega), axis=None) 
        self.state = np.concatenate((state1,state2), axis=None)
        self.initial_state = self.state

    def reset_state(self) -> None:
        self.time = 0
        self.pos = self.initial_state[0:3]
        self.vel = self.initial_state[3:6]
        self.euler = self.initial_state[6:9]
        self.omega = self.initial_state[9:12]
        self.initialize_state()

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
    
    def get_speed(self) -> float:
        return np.linalg.norm(self.state[3:6])

    def forward_dynamics(self):
        pass

    def step(self):
        pass

        
