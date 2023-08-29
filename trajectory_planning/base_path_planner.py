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

        def compute_next_point(
                self,
                points: list[list[float]],
                goal: list[float],
        ) -> list[float]:
            
            
            off_set = [0,0,0]
            computed_points = []
            filler = np.zeros((9,))

            past_bin = None

            for i in range(self.iterations):
                self.histogram.input_points(points=np.array(points)-off_set)
                for j in range(self.layers):
                    candidates = self.histogram.sort_candidate_bins(point=np.array(goal)-np.concatenate((off_set,filler)),layer=j)   
                    for i,candidate in enumerate(candidates):
                        if self.histogram.confirm_candidate_distance(min_distance=self.min_distance,
                                                                    bin=[candidate[1],candidate[2]],
                                                                    layer=j,
                                                                    past_bin=past_bin
                                                                    ):
                            if self.layers > 1:
                                past_bin = [int(candidate[1]),int(candidate[2])]
                            computed_points.append(self.histogram.get_reference_point_from_bin(bin=[candidate[1],candidate[2]],layer=j)+off_set)
                            break
                if self.iterations > 1:
                    off_set = computed_points[-1]
            return np.array(computed_points)