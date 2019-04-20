# P2 - Continuous Control  

This report details the algorithm and implementation for the Continuous Control project of the Deep Reinforcement Learning Nanodegree on Udacity.

The environment (Version 2) was solved using the DDPG algorithm (suitable for environments with continuous action spaces) which involves the use of four neural networks, two each for both actor and critic. 
Architecture of Neural Networks :
  1. Actor : The actor target and local networks had four layers, out of which two were hidden with 400 and 300 units respectively. 
  2. Critic : The critic target and local networks also had four layers, again out of which two were hidden with 400 and 300 units respectively. 

Exponential Linear Unit was used instead of ReLU or Leaky ReLU based on the both personal experience and advise from other students on the student hub, for both actor and critic networks. The function used as activation in output layer of actor network was tanh owing to the range of action space being -1 to 1, whereas the activation function used in output layer of critic network was linear acting as value function values meant to act as critic for the agent. 

![Actor network Code](https://github.com/championballer/RL/raw/master/P2_Continuous_Actions/Photos/ActorNetwork.png)
![Critic network Code](https://github.com/championballer/RL/raw/master/P2_Continuous_Actions/Photos/CriticNetwork.png)

The code for the agent system (generalising for >1 agents, distributed learning) comprises of agent class implementation that involves initialisation of the four networks, replay buffer, and OUNoise. The target and local networks for both actor and critic are initialised with same weights (Indicated to give good performance on Student Hub). The class also includes act method that gives the array of actions for the different states observed by different agents of the environment by passing the states through the actor network. This method is used to gain experience on the basis of current policy enacted by the actor network. We also have the step function which stores the current sets of states and actions along with their rewards for all 20 agents one by one in the replay buffer. This increases the amount of exploration done by the agents improving the time taken for the training of the model to happen. This method is decoupled from the update part unlike as implemented in the ddpg pendulmn notebook provided by Udacity. To be able to update without needing to explore, another method was defined. Next is the reset function which resets the noise object employed by the agent system per episode. Then we have the learn method which involves, obtaining the next actions on the basis of the next states in the sampled experiences from the replay buffer, from the actor network. Using these actions, Q_targets are estimated and are compared with Q_expected for the current set of states and actions to calculate the loss function using the mean square error function on which the optimisation of the critic is done. Next the actor network is optimised using the predictions of actions for the current states, and passing them through the critic network. That is followed by soft updates of both target networks.

![Agent Code 1](https://github.com/championballer/RL/raw/master/P2_Continuous_Actions/Photos/Agent1.png)
![Agent Code 2](https://github.com/championballer/RL/raw/master/P2_Continuous_Actions/Photos/Agent2.png)
![Agent Code 3](https://github.com/championballer/RL/raw/master/P2_Continuous_Actions/Photos/Agent3.png)
![Agent Code 4](https://github.com/championballer/RL/raw/master/P2_Continuous_Actions/Photos/Agent4.png)


The OUNoise class is same as that employed in the ddpg pendlumn with the difference that it is adapted for 20 different agents, plus standard normal distribution based noise is added (Student Hub tip). This is followed by the replay buffer class which is exactly same as that of the one in ddpg pendulmn. 