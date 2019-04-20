# P2 - Continuous Control  

This report details the algorithm and implementation for the Continuous Control project of the Deep Reinforcement Learning Nanodegree on Udacity.

The environment (Version 2) was solved using the DDPG algorithm which involves the use of four neural networks, two each for both actor and critic. 
Architecture of Neural Networks :
  1. Actor : The actor target and local networks had four layers, out of which two were hidden with 400 and 300 units respectively. 
  2. Critic : The critic target and local networks also had four layers, again out of which two were hidden with 400 and 300 units respectively. 

Exponential Linear Unit was used instead of ReLU or Leaky ReLU based on the both personal experience and advise from other students on the student hub, for both actor and critic networks. The function used as activation in output layer of actor network was tanh owing to the range of action space being -1 to 1, whereas the activation function used in output layer of critic network was linear acting as value function values meant to act as critic for the agent. 
