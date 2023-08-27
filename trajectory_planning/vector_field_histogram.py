import numpy as np

from trajectory_planning.polar_histogram_3D import polarHistogram3D

class vectorFieldHistogram3D():

    def __init__(
            self,
            radius: float = 1,
            angle_sections: float = 36,
            min_obstacle_distance: float = 0.4,
    ):

        self.histogram = polarHistogram3D(radius=radius, angle_sections=angle_sections)

    def compute_next_point(
            self,
            points: list[list[float]],
            goal: list[float],
    ) -> list[float]:
            
        self.histogram.input_points(points=points)
        candidates = self.histogram.sort_candidate_bins(point=goal)

        for candidate in candidates:
            if self.histogram.confirm_candidate_distance(min_distance=self.min_distance,
                                                         bin=[candidate[1],candidate[2]]):
                return self.histogram.get_reference_point_from_bin([candidate[1],candidate[2]])
            
        print('VFH method cannot find viable point to go to')
        exit()




