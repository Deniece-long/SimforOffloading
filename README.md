This is the code for unpublished papers. Note:(The title of the paper will be attached afterwards)
# Installation
_____
- Python 3.8
- SimPy, You can install SimPy easily via pip <http://pypi.python.org/pypi/pip>_:
  ```sh
   $ pip install -U simpy
  ```
 - tensoflow 1 version
 
# launch
____
The startup program simply runs the main function directly, then adjusts the number of edge servers and users in the participating environment by changing the global variable parameters,such as simulation time `simeTime`, the processing speed of edge server `rho`,  Edge server cache pool `buffer`, the cache pool indicates that the server reads unloaded tasks from the queue pool. The implement of offloading algorithm by changing different `name` in Offloading_Strategy.py file.

# Code structure
______
- `./dataset` :contains edgeResources-melbCBD.csv and users-melbdbd-generated.csv files.from [EUA datasets](https://github.com/PhuLai/eua-dataset).
- `./results`: Dueling DQN algorithm results
- `./userName`: Recording the results of different reinforcement learning algorithms on different edge servers with different users
- `./usermove`: Recording the latitude and longitude of random movements of users
- `Offloading_Strategy.py`: repsents the offloading strategy of user
- `RL_DDQN.py`:represents the Double Deep-Q Network algorithm
- `RL_DQN.py`:represents the Deep Q network alogrithm
- `RL_Dueling DQN.py`: represents the Dueling Deep Q network algorithm
- `RL_PRDQN.py`: represents the Deep Q network with the priority experience replay
- `main.py`: task offloading main function with fault tolerance
- `sysMpdel.py`:Includes task generation, user movement, edge server resources, edge server fault recovery.
