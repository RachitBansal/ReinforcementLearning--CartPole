# Q_Learning is a way to measure the reward that we would get when taking a particular action 'a' in a state 's'. 
# It is not only a measurment of the immediate reward but a summation of the entire future reward we would get from consequent actions as well. 
# Q(s,a) = r + Y*max(Q(s',a')); where, r is the immediate reward
# Using Mean Squared Loss
# Input will have a state matrix, the output matrix from the Neural Network would be a matrix of how good each action is

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random

class Agent:
	def __init__(self,state_size,action_size):
		self.state_size = state_size
		self.action_size = action_size
		self.memory = deque(maxlen=2000)
		# Memory is needed because we do not have any training data, so in this memory we will store the past exp[eriences of the agent
		# deque is a double ended queue, a data structure which can be edited from both sides 
		self.gamma = 0.95 #Discount Factor
		# Exploration vs Exploitation Tradeoff
		# Exploration : Good in the beginning, helps us to try various things and compare the rewards
		# Exploitation : Sample good experiences from the past, good in the end, using the knowledge of the past
		self.epsilon = 1.0 # 100% exploration in the beginning
		self.epsilon_decay = 0.995
		self.epsilon_min = 0.01 # 
		self.learning_rate = 0.001
		self.model = self._create_model()

	def _create_model(self):
		# Building a MLP to learn Q-Reward 
		model = Sequential()
		model.add(Dense(24,input_dim=self.state_size,activation='relu'))
		model.add(Dense(24,activation='relu'))
		model.add(Dense(self.action_size,activation='linear'))
		model.compile(loss='mse',optimizer=Adam(lr=0.001))
		return model

	# We need data to train the neural network, thus we use Replay Buffer Technique to generate data on the fly and use it for training
	# Thus, we predict the reward values, store it in the memory along with other parameters, use it to train our model and this process of training and appending goes on
	# The double sided queue in our case will store a maximum amount of data, we'll continuosly remove the initial values 

	def remember(self,state,action,reward,next_state,done):
		self.memory.append((state,action,reward,next_state,done))

	def act(self,state):
		# Sampling according to the Epsilon Greedy Method
		if np.random.rand()<=self.epsilon:
			# Return Random Action
			return random.randrange(self.action_size)
#		else:
			# Return Prediction from MLP
		return np.argmax(self.model.predict(state)[0])

	def train(self,batch_size=32):
		# Training using 'Buffer Replay'
		# We make training update after every tuple is passed 
		minibatch = random.sample(self.memory,batch_size)
		for experience in minibatch:
			state,action,reward,next_state,done = experience
			# x,y : state, expected reward 
			if not done:
				# Game not over, using Bellmam's equation
				target = reward + self.gamma*np.max(self.model.predict(next_state)[0])
			else:
				target = reward

			target_F = self.model.predict(state)
			target_F[0][action] = target

			self.model.fit(state,target_F,epochs=1,verbose=0)

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def load(self,name):
		self.model.load_weights(name)

	def save(self,name):
		self.model.save_weights(name)





  		