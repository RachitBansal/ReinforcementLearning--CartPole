import qlearning 
#import cartpole
import gym 
import numpy as np

env = gym.make("CartPole-v1")

n_episodes = 500
output_dir = "ReinforcementLearning/weights/"

state_size=4
action_size=2

# agent = qlearning.Agent
agent = qlearning.Agent(state_size=4,action_size=2)

done=False

for e in range(n_episodes):
	state = env.reset()
	state = np.reshape(state,[1,state_size])
	batch_size = 32

	for time in range(500):
		env.render()
		action = agent.act(state)
		next_state,reward,done,other_info = env.step(action)
		reward = reward if not done else -10
		next_state = np.reshape(next_state,[1,state_size])
		agent.remember(state,action,reward,next_state,done) # One experience for the agent
		state = next_state

		if done:
			print("Game Episode: {}/{} Score: {} Exploration Rate: {:.2}".format(e+1,n_episodes,time,agent.epsilon))
			break

	if len(agent.memory)>batch_size:
		agent.train(batch_size)

#	if e%50 == 0:
#		agent.save(output_dir+"weights_"+'{:04d}'.format(e)+".hdf5")


print("Deep Q-Learner Model Trained!")
env.close()
# print("All 20 episodes done")