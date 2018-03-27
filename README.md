# Deep Q-Network
Keras implementation of the algorithm from the DeepMind deep Q-learning nature paper 
(https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf). 
Pre-trained models for the OpenAI Gym CartPole-v1 and MountainCar-v0 environments included.
Includes a command line interface to train, run, and generate GIFs of an agent.
## Installation
In order to generate GIFs of your agent running you must have ffmpeg installed.
To install it type the following into a Python shell:
```python
import imageio
imageio.plugins.ffmpeg.download()
```
## Trained Agents
### CartPole-v1
![](gifs/CartPole-v1.gif?raw=true)
### MountainCar-v0
![](gifs/MountainCar-v0.gif?raw=true)