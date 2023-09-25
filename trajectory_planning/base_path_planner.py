import time
import numpy as np
from typing import Dict, Any, Type

from trajectory_planning.polar_histogram_3D import polarHistogram3D

class basePathPlanner():

    def __init__(
            self,
            path_planning_algorithm: str,
            kwargs: Dict[str, Any],
    ):
        self.algorithm = getattr(self,path_planning_algorithm)(**kwargs)

    class VFH():
        '''
        Vector Field Histogram Method in #D
        Original VFH publication:
        https://ieeexplore.ieee.org/document/846405
        '''

        def __init__(
                self,
                radius: float = 1,
                layers: int = 1,
                iterations: int = 1,
                angle_sections: float = 8,
                min_obstacle_distance: float = 1,
                probability_tolerance: float = 0.05,
                distance_tolerance: float = 0.2,
        ):

            self.histogram = polarHistogram3D(radius=radius, 
                                              layers=layers, 
                                              angle_sections=angle_sections,
                                              probability_tolerance=probability_tolerance,
                                              distance_tolerance=distance_tolerance,
                                              )
            self.min_distance = min_obstacle_distance
            self.iterations = iterations
            self.layers = layers
            self.radius = radius

        def input_points(
            self, 
            points: list[list[float]],
        ) -> None:
            self.histogram.input_points(points=points)

        def reset_map(self) -> None:
            self.histogram.reset_histogram()

        def get_layer_size(self) -> float:
            return self.histogram.layer_depth

        def compute_next_point(
                self,
                points: list[list[float]],
                goal: list[float],
        ) -> list[float]:
            
            
            off_set = [0,0,0]
            computed_points = [off_set]
            filler = np.zeros((goal.size-3,))

            past_bin = None
            past_bin2 = None
            done = False

            #t0 = time.time()
            for i in range(self.iterations):
                self.histogram.input_points(points=np.array(points)-off_set)
                #t0 = time.time()
                for j in range(self.layers):
                    #t0 = time.time()
                    candidates = self.histogram.sort_candidate_bins(
                                                                point=np.array(goal)-np.concatenate((off_set,filler)),
                                                                layer=j, 
                                                                previous=past_bin,
                                                                previous2=past_bin2,
                                                                )   
                    for i,candidate in enumerate(candidates):
                        if self.histogram.confirm_candidate_distance(min_distance=self.min_distance,
                                                                    bin=[candidate[1],candidate[2]],
                                                                    layer=j,
                                                                    past_bin=past_bin,
                                                                    goal=np.array(goal)-np.concatenate((off_set,filler)),
                                                                    ):
                            if self.layers > 1:
                                if j >= 1:
                                    past_bin2 = past_bin
                                past_bin = [int(candidate[1]),int(candidate[2])]
                            target, done = self.histogram.get_target_point_from_bin(bin=[candidate[1],candidate[2]],goal=goal[0:3],layer=j)
                            computed_points.append(target+off_set)
                            break
                    if done:
                        break
                    #print(time.time() - t0,'layer')
                #print(time.time() - t0)
                if self.iterations > 1:
                    off_set = computed_points[-1]
            #print(time.time() - t0)
            return np.array(computed_points)
        
        def check_goal_safety(
                self,
                goal: list[float],
        ) -> bool:
            safe = self.histogram.check_point_safety(min_distance=self.min_distance, point=goal)
            return safe