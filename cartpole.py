import gym 
env = gym.make("CartPole-v1")
for e in range(20):
	observation = env.reset()
	for t in range(50):
		env.render()
		action = env.action_space.sample()
		observation,reward,done,info = env.step(action)
		if done:
			print("Game Episode: {}/{} Score: {}".format(e+1,20,t))
			break
env.close()
print("All 20 episodes done")

# Q_Learning is a way to measure the reward that we would get when taking a particular action 'a' in a state 's'. 
# It is not only a measurment of the immediate reward but a summation of the entire future reward we would get from consequent actions as well. 
# Q(s,a) = r + Y*max(Q(s',a')); where, r is the immediate reward
# Using Mean Squared Loss
# Input will have a state matrix, the output matrix from the Neural Network would be a matrix of how good each action is

