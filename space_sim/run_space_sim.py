import sys
sys.path.insert(1, 'c:/Users/Cameron Mehlman/Documents/Magpie')

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import time

from space_sim.make_env import make_env
from envs.gui import Renderer

def simulate_space_env(render=False,verbose=False,pforce=0) -> None:

    @hydra.main(version_base=None, config_path="conf", config_name="config")
    def simulate(cfg : DictConfig):

        env = make_env(cfg)
        env.reset()
        env.set_perturbation_force(pforce)

        if render:
            renderer = Renderer(
                xlim = [-30,30],
                ylim = [-30,30],
                zlim = [-30,30],
                vista = False,
            )
            renderer.plot(env.get_object_data())

        time.sleep(10)
        timesteps = 5000
        timestep = 0
        done = False

        for i in range(timesteps):
            state,action,done,rew = env.step()
            if i%15 == 0:
                if verbose:
                    print('distance to goal:', env.get_distance_to_goal())
            if done:
                timestep = i
                break
            if render:
                renderer.plot(env.get_object_data())

        if done and rew>0:
            print('Simulation successful, goal reached in', timestep, 'timesteps')
        else:
            if i < timesteps-1:
                print('Simulation unsuccessful, ended early at timestep', timestep)
            else:
                print('Simulation unsuccesful, goal never reached')

    simulate()

if __name__ == "__main__":

    '''
    pforces = [5.0,10,15,20,25]
    iter = 3
    for p in pforces:
        print(p)
        for i in range(iter):
            simulate_space_env(render=False,verbose=False,pforce=p)
    '''
    simulate_space_env(render=False,verbose=False)