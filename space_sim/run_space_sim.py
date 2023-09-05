import sys
sys.path.insert(1, 'c:/Users/Cameron Mehlman/Documents/Magpie')

import hydra
from omegaconf import DictConfig, OmegaConf
import time

from space_sim.make_env import make_env
from envs.gui import Renderer

@hydra.main(version_base=None, config_path="conf", config_name="config")
def simulate_space_env(cfg : DictConfig) -> None:
    env = make_env(cfg)

    renderer = Renderer(
        xlim = [-5,5],
        ylim = [-5,5],
        zlim = [-5,5],
        vista = False,
    )
    
    env.reset()
    renderer.plot(env.get_object_data())

    time.sleep(1)

    for i in range(2000):
        env.step()
        renderer.plot(env.get_object_data())
    
    '''
    for i in range(2000):
        env.step()
    '''

if __name__ == "__main__":
    simulate_space_env()