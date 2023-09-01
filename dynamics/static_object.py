import numpy as np
import pyvista as pv
import copy
import matplotlib.pyplot as plt
from typing import Any, Dict, Type, Optional, Union


from util.util import *

class staticObject():

    def __init__(
            self,
            mesh: Union[Dict[str, list[Any]],Type[pv.DataSet]], #a list of vertices and lines connecting them, or VistMesh
            name: Optional[str] = None,
            pos: list[float] = [0,0,0],
    ):
        if mesh is None:
            print('Error: no mesh data given')
            exit()
        elif isinstance(mesh,pv.DataSet):
            self.vista = True
        else:
            self.vista = False

        self.name = name
        self.mesh = mesh
        self.initial_pos = pos

        self.temp_mesh = copy.deepcopy(mesh)
        self.point_cloud = None

    def update_points(self) -> None:
        if self.vista:
            self.temp_mesh.points += self.initial_pos
        else:
            self.temp_mesh['points'] += self.initial_pos

    def point_cloud_from_mesh(self, n) -> list[float]:
        '''
        takes a vista mesh, and returns a point cloud of n points
        method from:
        https://medium.com/@daviddelaiglesiacastro/3d-point-cloud-generation-from-3d-triangular-mesh-bbb602ecf238
        '''

        if not self.vista:
            print('Error: cannot generate point cloud data from simplified mesh, use pyvista, using corner points')
            return self.temp_mesh['points']

        #calculate face areas and sample faces to generate points on
        faces = self.temp_mesh.faces.reshape(-1, 4)[:, 1:]
        face_vertices = np.take(self.temp_mesh.points,faces,axis=0)

        areas = triangle_area(face_vertices)
        weighted_areas = areas/np.sum(areas)

        weighted_random_faces = np.random.choice(range(len(areas)), size=n, p=weighted_areas)
        sample_faces = np.take(face_vertices,weighted_random_faces,axis=0)

        #generate n random points in barycentric coordinates
        u = np.random.rand(n,1)
        v = np.random.rand(n,1)
        problem = u+v>1

        u[problem] = 1 - u[problem]
        v[problem] = 1 - v[problem]

        w = 1 - (u+v)

        b_coord = np.transpose(np.array([u,v,w])).squeeze()

        #convert barycentric to global coordinates
        point_cloud = []

        for i in range(n):
            point = (sample_faces[i][0]*b_coord[i][0] + sample_faces[i][1]*b_coord[i][1] + sample_faces[i][2]*b_coord[i][2])
            point_cloud.append(point)

        return np.array(point_cloud)
    

if __name__ == '__main__':
    mesh = pv.read('obstacle_course2.stl')
    mesh.points *= 1000
    obj = staticObject(mesh=mesh)
    pc = obj.point_cloud_from_mesh(10000)
    x = pc[:,0]
    y = pc[:,1]
    z = pc[:,2]
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(x, y, z, s=1)
    plt.show()

        


