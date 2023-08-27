import numpy as np
from typing import Type, Optional
import cvxpy as cp
import control 

from dynamics.quad_dynamics import quadcopterDynamics

class MPC():

    def __init__(
            self,
            dynamics: Type[quadcopterDynamics],
            upper_state_bounds: list[float],
            lower_state_bounds: list[float],
            upper_control_bounds: Optional[list[float]] = None,
            lower_control_bounds: Optional[list[float]] = None,
            horizon: int = 20,
            valued_actions: int = 5,
    ):
        
        self.dynamics = dynamics
        self.horizon = horizon
        self.valued_actions = valued_actions

        self.state_bounds = [lower_state_bounds,upper_state_bounds]
        self.control_bounds = [lower_control_bounds,upper_control_bounds]

        self.initialize_optimization_parameters()
        self.initialize_discrete_matrices()

    def initialize_optimization_parameters(self) -> None:
        self.x = cp.Variable((self.dynamics.state.size, self.horizon+1))
        self.u = cp.Variable((self.dynamics.control.size, self.horizon))
        self.x_init = cp.Parameter(self.dynamics.state.size)

        self.x_init.value = self.dynamics.state

    def initialize_discrete_matrices(self) -> None:
        # Convert continuous time dynamics into discrete time
        sys = control.StateSpace(self.dynamics.A, self.dynamics.B, self.dynamics.C, self.dynamics.D)
        sys_discrete = control.c2d(sys, self.dynamics.timestep*self.dynamics.horizon, method='zoh')

        self.A = np.array(sys_discrete.A)
        self.B = np.array(sys_discrete.B)


    def update_state(self) -> None:
        self.x_init.value = self.dynamics.state

    def compute_action(
            self,
            goal: list[float],
    ) -> list[list[float]]:
        
        cost = 0
        constr = [self.x[:, 0] == self.x_init]

        for t in range(self.horizon):
            cost += cp.quad_form(goal - self.x[:, t], self.dynamics.Q) + cp.quad_form(self.u[:, t], self.dynamics.R)
            constr += [self.state_bounds[0] <= self.x[:, t], self.x[:, t] <= self.state_bounds[1]]
            if isinstance(self.dynamics,quadcopterDynamics):
                constr += [self.x[:, t + 1] == self.A @ self.x[:, t] + self.B @ self.u[:, t]] 
            else:
                constr += [self.x[:, t + 1] == self.A * self.x[:, t] + self.B * self.u[:, t]]

        cost += cp.quad_form(goal-self.x[:, self.horizon], self.dynamics.Q)  # End of trajectory error cost
        problem = cp.Problem(cp.Minimize(cost), constr)

        problem.solve(solver=cp.OSQP, warm_start=True)

        solution = np.transpose(np.array(self.u[:,0:self.valued_actions].value))
        if solution is not None and isinstance(self.dynamics, quadcopterDynamics):
            return np.array([s - np.array([self.dynamics.mass*self.dynamics.g, 0, 0, 0]) for s in solution]).squeeze()#solution #
        elif solution is not None:
            return solution.squeeze()
        else:
            return solution

if __name__ == '__main__':
    sd1 = quadcopterDynamics(
        timestep = 2,
        horizon = 1,
        pos = np.array([0, 0, 0]),
        omega = np.array([0, 0, 0]),
        coef_data = {'kf' : 3.7102e-5, 'km' : 7.6933e-7},
        quad_data = {
            'l' : 0.243,
            'mass' : 1.587,
            'I' : np.array([0.0213, 0.02217, 0.0282]),
        },
    )

    xmin = np.array([-np.inf,  -np.inf,  -np.inf, -np.inf, -np.inf, -np.inf, -0.2, -0.2, -2*np.pi, -.25, -.25, -.25])
    xmax = np.array([np.inf,   np.inf,   np.inf,   np.inf,  np.inf, np.inf, 0.2,  0.2,   2*np.pi,  .25, .25,  .25])

    ymin = np.array([-20,-5,-5,-5])
    ymax = np.array([20,5,5,5])

    mpc = MPC(dynamics=sd1, 
              lower_state_bounds=xmin, 
              upper_state_bounds=xmax,
              lower_control_bounds=ymin,
              upper_control_bounds=ymax,
              valued_actions=1,
    )

    goal = [0, 0, 10, 0,0,0,0,0,0,0,0,0]
    mpc.update_state()

    #actions = mpc.compute_action(goal=goal)
    #print(actions)

    for i in range(10):
        actions = mpc.compute_action(goal=goal)
        print(actions)
        sd1.set_control(actions,motor_velocity=False)
        sd1.forward_step()
        mpc.update_state()
        print(sd1.state)
    