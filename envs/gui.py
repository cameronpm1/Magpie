import numpy as np
import pyvista as pv
import threading
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
from typing import Any, Dict, Type, Optional


from dynamics.base_dynamics import baseDynamics
from dynamics.sat_dynamics import satelliteDynamics
from dynamics.dynamic_object import dynamicObject

def threaded(fn):
    """Call a function using a thread."""
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper

class gui(object):

    '''
    DOES NOT WORK IF self.vista = True
    '''

    def __init__(
            self,
            rate: int = 50,
            xlim: list[float] = [-5,5],
            ylim: list[float] = [-5,5],
            zlim: list[float] = [-5,5],
            vista: bool = False,
    ):
        self.rate = rate
        self.xlim = xlim
        self.ylim = zlim
        self.zlim = ylim
        self.vista = vista


    def call_back(self, misc=None):
        if self.pipe.poll():
            command = self.pipe.recv()
            if command is None:
                self.terminate()
                return False
            else:
                self.plot(command)
        return True

    def terminate(self):
        plt.close('all')        

    def __call__(self, pipe):
        print('starting plotter...')
        self._fig = plt.figure(figsize=(10, 10))
        self._ax1 = self._fig.add_subplot(1, 1, 1, projection='3d')
        
        self.pipe = pipe
        timer = self._fig.canvas.new_timer(interval=1)
        timer.add_callback(self.call_back)
        timer.start()
        plt.show()

        print('...done')
        

    def plot_object(self, object1) -> None:
        points = object1['points']
        lines = object1['lines']
        self._ax1.clear()
        for line in object1['lines']:
            self._ax1.plot([points[line[0]][0],points[line[1]][0]],
                            [points[line[0]][1],points[line[1]][1]],
                            [points[line[0]][2],points[line[1]][2]], color="k")
        if 'goal' in object1.keys():
            self._ax1.plot(object1['goal'][:,0],object1['goal'][:,1],object1['goal'][:,2])
        if 'point cloud' in object1.keys():
            #remove point cloud data outside of axis limits
            for i,point in reversed(list(enumerate(object1['point cloud']))):
                if (point[0] < self.xlim[0] or point[0] > self.xlim[1]) or (point[1] < self.ylim[0] or point[1] > self.ylim[1]) or (point[2] < self.zlim[0] or point[2] > self.zlim[1]):
                    object1['point cloud']=np.delete(object1['point cloud'], i, 0) 
            self._ax1.scatter(object1['point cloud'][:][:,0],object1['point cloud'][:][:,1],object1['point cloud'][:][:,2], color='r',s=8)
        if 'final goal' in object1.keys():
            self._ax1.scatter(object1['final goal'][0],object1['final goal'][1],object1['final goal'][2], color='g', s=40)
        self._ax1.set_xticks(np.linspace(self.xlim[0],self.xlim[1],10))
        self._ax1.set_yticks(np.linspace(self.ylim[0],self.ylim[1],10))
        self._ax1.set_zticks(np.linspace(self.zlim[0],self.zlim[1],10))
            

    def plot(self, objects) -> None:
        self._ax1.clear()
        self.plot_object(objects)

        self._fig.canvas.draw()

class Renderer:
    """ send data to gui and invoke plotting """

    def __init__(
        self,
        xlim: list[float] = [-5,5],
        ylim: list[float] = [-5,5],
        zlim: list[float] = [-5,5],
        vista: bool = False,
    ):
        self.vista = vista
        if not self.vista:
            self.plot_pipe, plotter_pipe = mp.Pipe()
            self.plotter = gui(xlim=xlim,ylim=ylim,zlim=zlim,vista=vista)
            self.plot_process = mp.Process(
                target=self.plotter, args=(plotter_pipe,), daemon=True)
            self.plot_process.start()
        else:
            self.plotter = pv.Plotter()
            self.plotter.show(interactive_update=True)
            self.plotter.set_position([-10,0,10])
            self.plotter.fly_to([0,0,0])


    def plot(self, data):
        if not self.vista:
            send = self.plot_pipe.send
            if data is not None:
                send(data)
            else:
                send(None)
        else:
            self.plotter.clear_actors()
            for o in data:
                actor = self.plotter.add_mesh(o, color='black', style='wireframe', line_width=1)
                
            self.plotter.update()



    