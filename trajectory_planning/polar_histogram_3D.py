import numpy as np
from typing import Any, Dict, Optional
import matplotlib.pyplot as plt
import time

from util.util import gaussian_prob


class polarHistogram3D():

    def __init__(
            self,
            radius: float = 1,
            layers: int = 1,
            angle_sections: float = 36,
            probability_tolerance: float = 0.05,
            distance_tolerance: float = 0.2,
    ):
        self.points = None
        self.radius = radius
        self.layers = layers + 1 #always check 1 layer ahead
        #self.max_bin_arc = 2*np.pi*radius/angle_sections
        self.layer_depth = self.radius/self.layers
        self.probability_tol = probability_tolerance
        self.distance_tol = distance_tolerance

        self.sections = angle_sections
        self.range = 2*np.pi/self.sections
        self.histogram3D = np.zeros((self.sections,self.sections,self.layers,7)) #initialize the histogram
        self.refrerence_histogram3D = np.zeros((self.sections,self.sections,3)) #center points of each bin on unit ball
        self.initialize_reference_histogram3D()
        self.histogram_calc_storage = None

    def convert_cartesian_to_polar(
            self,
            point: list[float],
    ):
        theta1 = np.arctan2(point[1],point[0]) #angle between +x-axis and point vector
        theta2 = np.arctan2(point[2],point[0]) #angle between xy-plane and point vector (azimuth)

        #make sure angle is '+'
        if theta1 < 0:
            theta1 = 2*np.pi + theta1
        if theta2 < 0:
            theta2 = 2*np.pi + theta2

        dist = np.linalg.norm(point)

        return theta1,theta2,dist
    
    def convert_polar_to_bin(
            self,
            point: list[float],
    ) -> tuple[int,int,int]:
        theta = int(point[0]//self.range)
        phi = int(point[1]//self.range)
        layer = int(point[2]//self.layer_depth)

        return theta, phi, layer
    
    def convert_cartesian_to_bin(
            self,
            point: list[float],
    ) -> tuple[int,int,int]:
        theta1 = np.arctan2(point[1],point[0]) #angle between +x-axis and point vector
        theta2 = np.arctan2(point[2],point[0]) #angle between xy-plane and point vector (azimuth)

        #make sure angle is '+'
        if theta1 < 0:
            theta1 = 2*np.pi + theta1
        if theta2 < 0:
            theta2 = 2*np.pi + theta2

        dist = np.linalg.norm(point)

        theta = int(theta1//self.range)
        phi = int(theta2//self.range)
        layer = int(dist//self.layer_depth)

        return theta, phi, layer
    


    def get_reference_point_from_bin(
            self,
            bin: list[int],
            layer: int = 0,
    ) -> list[float]:

        return self.refrerence_histogram3D[int(bin[0])][int(bin[1])] * (self.layer_depth * (0.5+layer))
    
    def get_target_point_from_bin(
            self,
            bin: list[int],
            goal: list[float],
            layer: int = 0,
    ) -> tuple[list[float], bool]:
        '''
        check if goal is inside chosen bin, 
        if goal is inside bin and within range -> return goal
        if goal is inside bin and within range -> return goal vector w/ appropriate distance from center
        '''

        theta1,theta2,dist = self.convert_cartesian_to_polar(goal)
        if int(theta1) == int(bin[0]) and int(theta2) == int(bin[1]):
            if np.linalg.norm(goal) < (self.layer_depth * (0.5+layer)):
                return goal, True
            else:
                return goal/np.linalg.norm(goal) * (self.layer_depth * (0.5+layer)), False
        else:
            return self.refrerence_histogram3D[int(bin[0])][int(bin[1])] * (self.layer_depth * (0.5+layer)), False
    
    def get_bin_from_index(
            self,
            bin: list[int],
    ) -> list[float]:
        
        return self.histogram3D[int(bin[0])][int(bin[1])]
    
    def reset_histogram(self) -> None:
        self.histogram3D[:] = 0

    def input_points(
            self, 
            points: list[list[float]],
    ) -> None:
        #t0 = time.time()
        self.points = points
        self.histogram3D[:] = 0
        self.histogram_calc_storage = np.zeros((self.sections,self.sections,self.layers,3))
        a = 0

        for point in points:
            theta1,theta2,dist = self.convert_cartesian_to_polar(point)

            if dist > self.radius:
                next
            else:
                layer = int(dist//self.layer_depth)

                #bin_state = self.histogram3D[int(theta1//self.range)][int(theta2//self.range)][layer]
                bin_center = self.get_reference_point_from_bin(bin=[int(theta1//self.range),int(theta2//self.range)],layer=layer)
                self.histogram3D[int(theta1//self.range)][int(theta2//self.range)][layer][0:3] += point
                self.histogram3D[int(theta1//self.range)][int(theta2//self.range)][layer][3:6] += np.square(point) 
                self.histogram3D[int(theta1//self.range)][int(theta2//self.range)][layer][6] += 1
                self.histogram_calc_storage[int(theta1//self.range)][int(theta2//self.range)][layer] += point

                
                '''
                #only save the closest point to center in each bin
                if dist < self.histogram3D[int(theta1//self.range)][int(theta2//self.range)][layer][3] or self.histogram3D[int(theta1//self.range)][int(theta2//self.range)][layer][3] == 0:
                    self.histogram3D[int(theta1//self.range)][int(theta2//self.range)][layer] = [point[0],point[1],point[2],dist]
                '''
        '''
        Calculate the center of point cloud within each bin (average of x,y,z)
        and the standard dev. of the cloud within each bin (in x,y,z) to calculate
        a gaussian probability field of the chance there is an obstacle 
        '''
        for i,section1 in enumerate(self.histogram3D):
            for j,section2 in enumerate(section1):
                for k,layer in enumerate(section2):
                    if layer[6] == 0:
                        continue
                    else:
                        layer[0:3] /= layer[6]
                        layer[3:6] += np.multiply(-self.histogram_calc_storage[i][j][k]*2,layer[0:3]) + np.multiply(layer[6],np.square(layer[0:3]))
                        layer[3:6] /= layer[6]
                        layer[3:6] = np.sqrt(layer[3:6])
        #print(time.time() - t0)

    def initialize_reference_histogram3D(self) -> None:
        '''
        create a polar historgram that contains the xyz 
        coordinates of the centerpoint of each bin w distance of 1
        '''
        for i in range(self.sections):
            for j in range(self.sections):
                theta1 = i*self.range + self.range/2
                theta2 = j*self.range + self.range/2
                
                x = np.cos(theta2)*np.cos(theta1)
                y = np.cos(theta2)*np.sin(theta1)
                z = np.sin(theta2)

                self.refrerence_histogram3D[i][j] = [x,y,z]

    def sort_candidate_bins(
            self,
            point: list[float],
            layer: int = 0,
            previous: list[int] = None,
            previous2: list[int] = None,
    ) -> list[list[float]]:

        sorted_bins = []
        #theta1,theta2,layerp = self.convert_cartesian_to_polar(point)
        for i in range(self.sections):
            for j in range(self.sections):
                if (self.histogram3D[i][j][layer][0:3] == [0,0,0]).all():
                    if previous is None:
                        angle = np.arccos(np.dot(point[0:3],self.refrerence_histogram3D[i][j]) / (np.linalg.norm(point[0:3])*np.linalg.norm(self.refrerence_histogram3D[i][j])))
                        cost = angle
                    else:
                        previous_point, filler = self.get_target_point_from_bin(bin=previous,goal=point[0:3],layer=layer-1)
                        current_point, filler = self.get_target_point_from_bin(bin=[i,j],goal=point[0:3],layer=layer)
                        angle1 = np.arccos(np.clip(np.dot(point[0:3]-previous_point,current_point-previous_point) / (np.linalg.norm(point[0:3]-previous_point)*np.linalg.norm(current_point-previous_point)),-1,1))
                        cost = angle1
                        #if i == int(theta1) and j == int(theta2):
                        #    print(cost)
                        if previous2 is not None:
                            previous_point2, filler = self.get_target_point_from_bin(bin=previous2,goal=point[0:3],layer=layer-2)
                            angle2 = np.arccos(np.clip(np.dot(previous_point-previous_point2,current_point-previous_point) / (np.linalg.norm(previous_point-previous_point2)*np.linalg.norm(current_point-previous_point)),-1,1))
                            cost += angle2 * 0.2
                    sorted_bins.append([cost,i,j,layer])

        sorted_bins = np.array(sorted_bins)

        if sorted_bins.size == 0:
            return []
        else:
            return sorted_bins[sorted_bins[:, 0].argsort()]
        
    def check_obstacle_bins(
           self,
            point: list[float],
            bin: list[int],
            distance: float,
            layer: int = 0,
    ) -> bool:
        '''
        check obstacle bins in order of closest to furthest from point 
        if obstacle found but is further from distance, it is assumed
        all other obstacles will also be further from distance
        i.e. this WILL cause inaccurate min_distance if self.sections is low

        designed to be FAST, the slower version is sort_obstacle_bins()

        if * is the bin with the point of interest 
        first, we check:
        
                    + + +
                    + * +
                    + + +
        then, we check: 
                  
                  + + + + + 
                  +       +
                  +   *   +
                  +       +
                  + + + + +

        and so on...
        '''


        if all(b == 0 for b in self.histogram3D[:,:,layer,:].flatten()):
            return True
        else:
            theta = list(range(self.sections))
            phi = list(range(self.sections))

            theta.sort(key=lambda x: min(abs(bin[0]-x),abs(bin[0]-(x-self.sections))))
            phi.sort(key=lambda x: min(abs(bin[1]-x),abs(bin[1]-(x-self.sections))))
            theta.pop(0)
            phi.pop(0)

            row_flip = False
            column_flip = False
            last_pass = False

            iter = int(np.ceil((self.sections-1)/2))
            for k in range(iter):
                if last_pass:
                    return True
                if k < iter-1:
                    start = k*2
                    end = k*2 + 2
                else:
                    start = k*2
                    end = k*2 + 1
                #check columns of square
                low = min(phi[start],phi[end-1])
                high = max(phi[start],phi[end-1])
                if low==high:
                    column = list(range(0,self.sections))
                else:
                    if column_flip:
                        column = list(range(0,low+1))+list(range(high,self.sections))
                    else:
                        column = list(range(phi[start],phi[end-1]+1))
                    if low == 0 or high == self.sections-1:
                        column_flip = True
                for i in theta[start:end]:
                    for j in column:
                        if (self.histogram3D[i][j][layer][0:3] != [0,0,0]).any():
                            dist = np.linalg.norm(self.histogram3D[i][j][layer][0:3]-point)
                            if dist < distance:
                                return False
                            else:
                                last_pass = True
                #check columns of square
                low = min(theta[start],theta[end-1])
                high = max(theta[start],theta[end-1])
                if low==high:
                    row = list(range(0,self.sections))
                else:
                    if row_flip:
                        row = list(range(0,low))+list(range(high+1,self.sections))
                    else:
                        row = list(range(low+1,high))
                    if low == 0 or high == self.sections-1:
                        row_flip = True
                for j in phi[start:end]:
                    for i in row:
                        if (self.histogram3D[i][j][layer][0:3] != [0,0,0]).any():
                            dist = np.linalg.norm(self.histogram3D[i][j][layer][0:3]-point)
                            if dist < distance:
                                return False 
                            else:
                                last_pass = True
                            
            return True

    
    def sort_obstacle_bins(
            self,
            point: list[float],
            bin: list[int],
            distance: float,
            layer: int = 0,
    ) -> list[list[float]]:

        sorted_bins = []

        for i in range(self.sections):
            for j in range(self.sections):
                if (self.histogram3D[i][j][layer][0:3] != [0,0,0]).any():
                    dist = np.linalg.norm(self.histogram3D[i][j][layer][0:3]-point)
                    sorted_bins.append([dist,i,j,layer])
        
        sorted_bins = np.array(sorted_bins)
        if sorted_bins.size == 0:
            return []
        else:
            return sorted_bins#[sorted_bins[:, 0].argsort()]
        
    def check_point_safety(
            self,
            min_distance: float,
            point: list[float],           
    ) -> bool:
        #t0 = time.time()
        theta1,theta2,dist = self.convert_cartesian_to_polar(point)
        b1,b2,l = self.convert_polar_to_bin([theta1,theta2,dist])
        bin = [b1,b2,l]
        fs = 0.6 #factor of safety, point must be more than 10% closer to min_distance to change route

        if dist > self.radius:
            return True
        else:
            layer = int(dist//self.layer_depth)

        layers = range(self.layers)

        obstacle_bins = self.sort_obstacle_bins(point=point,bin=bin,distance=min_distance,layer=layers[0])

        for i in layers[1:]:
            temp_obstacle_bins = self.sort_obstacle_bins(point=point,bin=bin,distance=min_distance, layer=i)
            if len(obstacle_bins) == 0:
                obstacle_bins = temp_obstacle_bins
            else:
                if(len(temp_obstacle_bins)) == 0:
                    continue
                else:
                    obstacle_bins = np.vstack((obstacle_bins,temp_obstacle_bins))

        if len(obstacle_bins) > 0:
            obstacle_bins = obstacle_bins[obstacle_bins[:, 0].argsort()]

        for bad_bin in obstacle_bins:
            obstacle = self.histogram3D[int(bad_bin[1])][int(bad_bin[2])][int(bad_bin[3])][0:3]
            obstacle_std = self.histogram3D[int(bad_bin[1])][int(bad_bin[2])][int(bad_bin[3])][3:6]
            obstacle_probability = gaussian_prob(mu=obstacle, std=obstacle_std, x=point-obstacle)
            #if gaussian dist. is not in the same dimensions as goal (std dev of x,y, or z is0),
            #there is no probability of there being an obstacle, else, it is distance from 
            #center of the ellipsoide
            zeros = np.where(obstacle_probability == 0)[0]
            if zeros.size != 0:
                for zero in zeros:
                    if abs(point[zero]-obstacle[zero]) > 0:
                        fobstacle_probability = 0
                        break
            else:
                fobstacle_probability = min(obstacle_probability)
            if fobstacle_probability > self.probability_tol or np.linalg.norm(point-obstacle) < min_distance*fs:
                return False  
        #print(time.time() - t0)
        return True

    def confirm_candidate_distance(
            self,
            min_distance: float,
            bin: list[int],
            goal: list[float],
            layer: int = 0,
            past_bin: Optional[list[int]] = None,
    ) -> bool:
        '''
        Checks all obstacle bins and confirms that no obstacle
        is closer than min_distance to the centerline of the
        candidate bin.
        '''

        center_point, filler = self.get_target_point_from_bin(bin=bin,goal=goal[0:3],layer=layer)
        theta1,theta2,dist = self.convert_cartesian_to_polar(center_point)

        if dist < min_distance:
            layer0 = 0
        else:
            layer0 = int((dist-min_distance)//self.layer_depth)

        if (dist + min_distance) > self.radius:
            layerN = self.layers
        else:
            layerN = int(np.ceil((dist+min_distance)/self.layer_depth))
            
        layers = range(layer0,layerN)

        b1,b2,l = self.convert_polar_to_bin([theta1,theta2,dist])

        for i in layers:
            safe = self.check_obstacle_bins(point=center_point,bin=[b1,b2,l],distance=min_distance, layer=i)
            if not safe:
                return False
        '''
        for bad_bin in obstacle_bins:
            #Calculate probability of an obstacle being at the new goal
            if bad_bin[0] < min_distance:
                return False
            else:
                obstacle = self.histogram3D[int(bad_bin[1])][int(bad_bin[2])][int(bad_bin[3])][0:3]
                obstacle_std = self.histogram3D[int(bad_bin[1])][int(bad_bin[2])][int(bad_bin[3])][3:6]
                obstacle_probability = gaussian_prob(mu=obstacle, std=obstacle_std, x=center_point-obstacle)
                #if gaussian dist. is not in the same dimensions as goal (std dev of x,y, or z is0),
                #there is no probability of there being an obstacle, else, it is distance from 
                #center of the ellipsoide
                zeros = np.where(obstacle_probability == 0)[0]
                if zeros.size != 0:
                    for zero in zeros:
                        if abs(center_point[zero]-obstacle[zero]) > 0:
                            fobstacle_probability = 0
                            break
                else:
                    fobstacle_probability = min(obstacle_probability)
                    
                if fobstacle_probability > self.probability_tol:
                    return False    
        '''
        '''
        Check if path from previous chose ben to current bin intersects a bin with obstacles
        ONLY necissary when running algorithm w/ more than 1 layer
        '''
        #if past_bin is not None:
        #    past_bin_center, filler = self.get_target_point_from_bin(bin=past_bin,goal=goal[0:3],layer=layer-1)
        #    n = int(np.linalg.norm(past_bin_center-center_point)//self.distance_tol + 1)
        #    check_positions = np.linspace(past_bin_center,center_point,n)
        #    for position in check_positions:
        #        theta1,theta2,dist = self.convert_cartesian_to_polar(position)
        #        layer = int(dist//self.layer_depth)
        #        if (self.histogram3D[int(theta1//self.range)][int(theta2//self.range)][layer][0:3] != [0,0,0]).any():
        #            return False
        return True
    

        #bin_min_distance = min_distance + self.max_bin_arc
        '''
        obstacle_bins = self.sort_obstacle_bins(point=self.refrerence_histogram3D[int(bin[0])][int(bin[1])], layer=layer)
        center_line = self.refrerence_histogram3D[int(bin[0])][int(bin[1])] - off_set

        for bad_bin in obstacle_bins:
            if bad_bin[0] > np.pi/2:
                break
            else:
                obstacle = self.histogram3D[int(bad_bin[1])][int(bad_bin[2])][layer][0:3] - off_set
                distance = np.linalg.norm(np.cross(center_line,-obstacle))/np.linalg.norm(center_line)
                if distance < min_distance:
                    return False
                
        return True
        '''

        


if __name__ == '__main__':
    histogram = polarHistogram3D()

    t0 = time.time()
    for i in range(100):
        bins = histogram.sort_bins(goal=[1,0,0])
    t1 = time.time()
    print(bins)
    print(t1-t0)
    '''
    test = histogram.refrerence_histogram3D

    x = []
    y = []
    z = []

    for i in range(histogram.sections):
        for j in range(histogram.sections):
            x.append([test[i][j][0]])
            y.append([test[i][j][1]])
            z.append([test[i][j][2]])

    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(x, y, z)
    plt.show()
    '''