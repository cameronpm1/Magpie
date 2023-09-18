import numpy as np
from typing import Union

def euler_from_quaternion(quaternion):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        w = quaternion[0]
        x = quaternion[1]
        y = quaternion[2]
        z = quaternion[3]

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = np.arcsin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)
     
        return np.array([roll_x, pitch_y, yaw_z]) # in radians

def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

def dcm_zyx(E):
      '''
      Input: 3 euler angles
      Output: zyx Direction Cosine Matrix

      If the euler angles are used to describe the change from fixed
      coordinate system to body coordinate system, this will transform
      coordinates in the body from to the fixed frame

      For transforming coordinates from the fixed frame to body frame,
      use the transpose of this DCM
      '''
      
      c1 = np.cos(E[0])
      s1 = np.sin(E[0])
      c2 = np.cos(E[1])
      s2 = np.sin(E[1])
      c3 = np.cos(E[2])
      s3 = np.sin(E[2])

      dcm = (
            [[c3*c2, c3*s2*s1-c1*s3, s3*s1+c3*c1*s2],
             [c2*s3, c3*c1+s3*s2*s1, c1*s3*s2-c3*s1],
             [-s2, c2*s1, c2*c1]]
            )
      
      return dcm

def triangle_area(v):
      '''
      Computes the area of a triangle given 3 3-dimensional coordinates
      of the triangles vertices

      v = [ [v_11, v_12, v_13], ... ,[v_i1,v_i2,v_i3] ]
      '''
      areas = []

      for vert in v:
            area = 0.5*np.linalg.norm(np.cross(vert[1]-vert[0],vert[2]-vert[0]))
            areas.append(area)

      return np.array(areas)

def gaussian_prob(
            x: list[float],
            mu: list[float],
            std: list[float],
) -> list[float]:
      '''
      given an array of means, standard deviations, and values x
      calculate the gaussian probability of values x
      '''
      prob = prob = np.zeros((len(x),))
      if np.linalg.norm(std) == 0 and np.linalg.norm(mu-x) > 0:
            return prob
      else: 
            for i in range(len(x)):
                  if std[i] == 0:
                        continue
                  else:
                        exponent = np.around(-0.5*np.square(np.divide((x[i]-mu[i]),std[i])), decimals=8)
                        prob[i] = np.divide(1,std[i]*np.sqrt(np.pi*2))*np.exp(exponent)
      return prob

def line_to_point(
      line: list[float],
      point: list[float],
) -> tuple[float, float]:
      '''
      given a line AB (represented as a vector [x,y,z]), and a point (P)
      in cartesian coordinates, returns the shortest distance between
      the line and point (line CP), as well as the length of AC.
      IMPORANT: assumes that the line starts at the origin 
          P*
          /|
         / | 
        /  |
      A-----------B
          C    
      '''

      CP = np.linalg.norm(np.cross(line,point)) / np.linalg.norm(line)
      AC = np.dot(line,point) / np.linalg.norm(line)

      return CP, AC