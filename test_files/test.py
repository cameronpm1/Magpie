import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

mesh = pv.read('ball.stl')
#mesh.plot()
print(mesh.faces.reshape(-1, 4)[:, 1:])
print(mesh.points)
x = mesh.points[:,0]
y = mesh.points[:,1]
z = mesh.points[:,2]
ax = plt.figure().add_subplot(projection='3d')
ax.plot(x, y, z)

plt.show()