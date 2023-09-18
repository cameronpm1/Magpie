import sys
sys.path.insert(1, 'c:/Users/Cameron Mehlman/Documents/Magpie')

import numpy as np
import pyvista as pv
from typing import Any, Dict, Type, Optional
import copy
import time
import matplotlib.pyplot as plt
from scipy import interpolate

from trajectory_planning.base_path_planner import basePathPlanner
from dynamics.static_object import staticObject


class pathPlanner(basePathPlanner):

    def __init__(
            self,
            goal_state: list[float],
            path_planning_algorithm: str, #VFH
            kwargs: Dict[str, Any],
            max_distance: float = 0.5,
            interpolation_method: str = 'linear',
            avg_speed: float = 0.5,
            n: int = 50,
    ):
        self.goal = goal_state
        self.avg_speed = avg_speed
        self.max_distance = max_distance
        self.interpolator = getattr(self,interpolation_method+'_interpolator')
        self.interpolation_method = interpolation_method
        self.n = n #for spline interpolation, determines how many points to compute

        super().__init__(
            path_planning_algorithm=path_planning_algorithm,
            kwargs=kwargs,
        )

    def set_goal_state(
            self,
            goal_state: list[float,]
    ) -> None:
        self.goal = goal_state

    def update_point_cloud(
            self,
            point_cloud: list[list[float]]
    ) -> None:
        if point_cloud is None:
            self.algorithm.reset_map()
        else:
            self.algorithm.input_points(points=point_cloud)

    def check_goal_safety(
            self,
            goals: list[list[float]],
            state: list[float] = None,
    ) -> bool:
        for goal in goals:
            goal = goal[0:3]
            safe = self.algorithm.check_goal_safety(goal - state)
            if not safe:
                return False
        return True


    def compute_desired_path(
            self,
            state: list[float],
            point_cloud: Optional[list[list[float]]] = None,
    ) -> list[float]:

        state_offset = np.zeros((state.size,))
        state_offset[0:3] = state[0:3]
        current_state = state

        if point_cloud is None:
            next_location = self.goal[0:3] - state[0:3]
            if np.linalg.norm(next_location) < 0.25*self.algorithm.radius:
                next_location = [current_state[0:3] - state[0:3], next_location]
            else:
                next_location = [current_state[0:3] - state[0:3], next_location/np.linalg.norm(next_location)*0.25*self.algorithm.radius]
        else:
            goal = self.goal-state_offset
            if np.linalg.norm(goal[0:3]) < self.algorithm.get_layer_size():
                next_location = [current_state - state_offset, goal]
            else:
                #t0 = time.time()
                next_location = self.algorithm.compute_next_point(points=point_cloud, goal=goal)
                #print(time.time() - t0)

        if self.interpolation_method == 'linear':
            path = np.array([])

            for i in range(len(next_location)):

                vel_vector = next_location[i][:] - current_state[0:3]
                vel_vector = vel_vector/np.linalg.norm(vel_vector) * self.avg_speed
        
                next_state = copy.deepcopy(state_offset)
                next_state[0:3] += next_location[i][:]
                next_state[3:6] = vel_vector
                new_path = self.interpolator([current_state,next_state])
                if i==0:
                    path = new_path
                else:
                    path = np.concatenate((path,new_path),axis=0)
                current_state = next_state
        elif self.interpolation_method == 'spline':
            path = self.spline_interpolator(next_location+state[0:len(next_location[0])],pts=self.n)
            if len(path) == 0:
                return []
            else:
                filler = np.zeros((len(path),len(state)-3))
                path = np.concatenate((path,filler),axis=1)
        return path

    
    def linear_interpolator(
            self,
            trajectory: list[list[float]],
    ) -> list[float]:
        
        start = np.array(trajectory[0][0:3])
        end = np.array(trajectory[-1][0:3])

        n = int(np.linalg.norm(start-end)//self.max_distance + 1)

        if n <= 1:
            return np.array(trajectory)
        else:
            return np.linspace(trajectory[0],trajectory[-1],n)
    
    def spline_interpolator(
            self,
            trajectory: list[list[float]],
            pts: int = 50
    ) -> list[float]:
        k = 2
        if k >= len(trajectory):
            k = len(trajectory)-1

        tck, u = interpolate.splprep([trajectory[:,0],trajectory[:,1], trajectory[:,2]],k=k, s=2)
        u_fine = np.linspace(0,1,pts)
        x, y, z = interpolate.splev(u_fine, tck)
        points = []

        for i,px in enumerate(x):
            points.append([px,y[i],z[i]])
            
        return points


if __name__ == "__main__":

    state = np.zeros(12,)
    obstacle_course_mesh = pv.read('stl_files/obstacle_course2.stl')
    obstacle_course_mesh.points *= 1000 #fix scaling issue w/ solidworks and STL exporting
    obstacle_course = staticObject(mesh=obstacle_course_mesh,name='obstacle_course')
    point_cloud = np.array(obstacle_course.point_cloud_from_mesh(n=5000))

    traj = pathPlanner(goal_state=[0.2,10.5,5,0,0,0,0,0,0,0,0,0],
                       path_planning_algorithm='VFH',
                       interpolation_method='spline',
                       kwargs={'radius':5,
                               'iterations':3,
                               'layers':8, 
                               'angle_sections':14,
                               'distance_tolerance': 0.2,
                               'probability_tolerance': 0.05,
                               'min_obstacle_distance': 2.0,
                               },
                        n = 30,
                       )
    points = traj.compute_desired_path(state=state, point_cloud=point_cloud)

    x = points[:,0]
    y = points[:,1]
    z = points[:,2]

    '''
    mx = m_points[:,0]
    my = m_points[:,1]
    mz = m_points[:,2]
    '''

    px = point_cloud[:,0]
    py = point_cloud[:,1]
    pz = point_cloud[:,2]

    ax = plt.figure().add_subplot(projection='3d')
    #ax.scatter(x, y, z)
    #ax.scatter(mx, my, mz, color='k')
    ax.scatter(px, py, pz)
    ax.plot3D(x,y,z, color='g')
    ax.scatter(0.2,10.5,5)

    plt.show()
    


