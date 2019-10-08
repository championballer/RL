# Project 1: Navigation
<p align="center">
<img src ="https://github.com/championballer/RL/raw/master/P1_Navigation/Navigation.gif">
<br>
<b> Solved Environment <b>
</p>

This report details the algorithm and implementation for the Navigation project of the Deep Reinforcement Learning Nanodegree on Udacity.

The environment was solved using the double DQN algorithm which involves the use of two networks namely target and local to map states to action values to improve reward using fixed targets methadology, while operating on a memory buffer to be able to sample different uncorrelated episodes to remove the bias that can get introduced in the agent's behaviour due to the correlation. The neural network takes place of the Q table which was handy in discrete state space in mapping states to action values. Double DQN is used here; which is an improvement over the normal DQN where the local network was used to select actions for future states, and those actions were used in the target network to estimate rewards. This is done to avoid over estimation. The rewards are high only if both networks of the agent agree on the actions being used. Else the rewards will be less. In the long run, the algorithm prevents the agent from propogating incidental high rewards that may be obtained by chance and do not reflect long term returns. 

On implementation side of things, Double DQN turned out to be more stable than classic DQN. The architecture for the neural networks is that comprising of two hidden layers containing 128 and 64 units each. The configuration was selected after experimenting with many architectures. With 256 units in two layers, the performance got a little slow, whereas with 64 units in two layers, the learning period took longer initially. Then the currrent architecture was tried and it yielded satisfactory results, and hence it was used henceforth. It may be possible that the performance on other configurations might have been affected by other hyper parameters. So keeping that in mind is very important. The activation function employed on each hidden layer is Exponential Linear Unit or ELU which was observed to give minutely better and more stable performance than ReLU. The output layer of the networks was normal output function to obtain action values corresponding to each possible action in the action space. 

<p align="center">
<img src ="https://github.com/championballer/P1_Navigation/raw/master/Images/SS1.png">
</p>

Note : The construction function of the Qnetwork takes in an array indicating the sizes of the hidden layers and that are used to create the network architecture. In this case they were, [128,64]. 

For the agent, local and target networks are initialised using the mentioned hidden layers sizes, while Adam optimiser is loaded for the optimisation process. Replay buffer has been picked from the code used in the DQN exercise. The local and targert networks are initialised with the same weights, as it was indicated on student hub to be giving relatively better results. This is done using a hard update function written after the soft update function. 

The act function is decoupled from the update function, for the obvious reasons of allowing updates to be independent of the experience gain, and also the requirement of being able to modify the update variable in the main notebook and not having to modify the agent code for the same. The act function chooses the random function from numpy to see if the random value is more or less than the current value of epsilon, if it is more than the max action value based action is selected, else a random action is sampled.

For the learn function of the agent, the local network is used to find the best possible actions for the next states and these actions are used to find the td target for the local network to optimise its weights for, hence implementing the double q learning principle to reduce over estimation. The mean square error loss function is used for optimisation purposes. Also as a means to increase stability of the network that gradients have been clipped for each learning step. After which the soft update takes place between target and local network with tau factor.

<p align="center">
<img src ="https://github.com/championballer/P1_Navigation/raw/master/Images/SS2.png">
<img src ="https://github.com/championballer/P1_Navigation/raw/master/Images/SS3.png">
<img src ="https://github.com/championballer/P1_Navigation/raw/master/Images/SS4.png">
<img src ="https://github.com/championballer/P1_Navigation/raw/master/Images/SS5.png">
<img src ="https://github.com/championballer/P1_Navigation/raw/master/Images/SS6.png">
</p>


The main notebook involves, the implementation of the main DQN function where the number of max episodes, and number of max time steps to be taken in any episodes are given. Based on that states and actions are added to memory at each step and at each step where the buffer size is more than the size of the batch, update of the network takes place depending on the value of the update variable, which in the final solution is 4. Epsilon greedy based learning is used in the solution where the decay rate is fixed at 0.995 (learned from fellow student in the student hub), while the minimum epsilon value was fixed at 0.01. 

The other hyper parameters take the value as indicated in the image below, and have been picked from the DQN exercise notebook itself.

<p align="center">
<img src ="https://github.com/championballer/P1_Navigation/raw/master/Images/SS7.png">
</p>

After every 100 episodes, the average score of the last 100 epsiodes, was printed till the average score of the last 100 epsiodes doesn't reach the value of 13, which in our case happened in 715 episodes. The plot for the same is as shown below :

<p align="center">
<img src ="https://github.com/championballer/P1_Navigation/raw/master/Images/SS8.png">
</p>

## Future Possible Improvements

1. As mentioned by Udacity, priority q targets as well as Dueling DQN would be implemented to see how the performance of the network improves.

2. Different other architectural structures for the neural nets  will be tried to check if the efficiency of the network can be increase further.

3. Batch normalisation will be tried with ReLU to see if it outperforms ELU performance wise.

4. Update frequency will be played with to further fine tune performance of the agent. 
