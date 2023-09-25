# Magpie
Python environment for simulation autonomous path planning on a satellite or UAV <br>

Instalation:<br>
  &emsp;-use a conda environment with the included requirements.txt file<br>
  &emsp;-in certain files such as enjoy_obstacle_avoidance_env.py you much change<br> &emsp; the path to the correct file location on your computer (line 2 of any file with this issue)<br>
  &emsp;-when running on linux or mac, always make sure the terminal file location is in the <br> &emsp; Magpie folder and not a subfolder, also run the code in the terminal as follows: <br> &emsp; python -m space_sim.run_space_sim <br>
  &emsp;-you may get an error in numpy.core.methods mentioning numpy.bool having been depreciated, <br> &emsp; in line 74 of vtkmodules.util.numpy_support.py change numpy.bool to numpy.bool_<br>

