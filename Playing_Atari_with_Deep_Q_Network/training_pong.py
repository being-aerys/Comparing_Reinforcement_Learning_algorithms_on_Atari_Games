#This work was done as a self-practise with references from Deep Reinforcement Learning Hands-On by Maxim Lapan.

import Wrappers_for_hacks_for_better_performance
import DQN_model
import collections, numpy as np
import torch, argparse, time
from tensorboardX import SummaryWriter

'''CONSTANTS'''
DISCOUNT_FACTOR = 0.9
DEFAULT_ENVIRONMENT = "PongNoFrameskip-v4"
DEFAULT_REWARD_TO_BEAT = 20
REPLAY_BUFFER_SIZE = 10000
INITIAL_EPSILON_FOR_EXPLORATION = 1
FINAL_EPSILON = 0.02
LAST_FRAME_TO_DECAY_EPSILON = 10 ** 5
LEARNING_RATE = 1e-4
REPLAY_SIZE_BEFORE_FIRST_MODEL_UPDATE = 10000
UPDATE_TARGET_NETWORK_FREQUENCY = 1000
BATCH_SIZE = 32


'''Data structure for each transition'''
Experience_data_structure = collections.namedtuple("Experience", field_names=["state", "action", "reward", "done", "next_state"])

'''Replay Buffer Class definition'''
class Experience_Replay_Buffer:

    def __init__(self, size_of_buffer):
        self.replay_buffer = collections.deque(maxlen=size_of_buffer) #automatically removes???
        #deque data structure is a double-ended queue and supports insertion and deletion from both the ends
        # Append and pop operations with a complexity of O(1) vs an array with O(n)

    def __len__(self):
        return len(self.replay_buffer)

    def append(self, sample_experience):
        self.replay_buffer.append(sample_experience)

    def sample_an_experience(self, batch_size_to_sample):
        #generate random integers between 0 and buffersize-1
        random_indices = np.random.choice(len(self.replay_buffer),batch_size_to_sample,replace=False) #setting replace == False makes sure you do not sample the same element twice in a batch
        states, actions, next_states, rewards, dones = zip(*[self.replay_buffer[idx] for idx in random_indices])
        #point is changing the data types below?
        # print(np.array(states))
        # print(np.array(actions))
        # print(np.array(rewards, dtype=np.float32))
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(next_states), np.array(dones, dtype=np.float)

'''DQN agent class definition'''
class Agent:

    def __init__(self, agent_env, replay_buffer):
        self.env = agent_env
        self.replay_buffer = replay_buffer
        self.total_reward = 0
        self.state = self.env.reset()


    def step(self, network, epsilon = 0, device = "cpu"):

        episode_reward_holder = None

        #Do Epsilon-Greedy exploration
        if np.random.random() < epsilon: #epsilon
            action_to_perform = self.env.action_space.sample()
        else: #greedy
            state_array = np.array([self.state], copy= False) # Not making a new copy of the object but rather passing a reference
            state_tensor = torch.tensor(state_array).to(device)
            q_values_returned_by_the_network = network(state_tensor.float())
            _, action_to_choose = torch.max(q_values_returned_by_the_network, dim= 1 ) # returns the maximum value of each row of the input tensor in the given dimension dim
            action_to_perform = int(action_to_choose.item()) #getting only the data part of the cuda variable

        next_state, reward, is_done, _ = self.env.step(action_to_perform)
        self.total_reward += reward

        #Store this experience in the buffer
        experience = Experience_data_structure(self.state, action_to_perform, reward, is_done, next_state)
        self.replay_buffer.append(experience)

        self.state = next_state
        if is_done == True:
            episode_reward_holder = self.total_reward #we return the episode reward only if the episode is over, else we return None
            self.env.reset()
            self.total_reward = 0

        return  episode_reward_holder

    def calculate_error(self, batch_of_samples, updating_network, target_network, device = "cpu"):
        states, actions, rewards, dones, next_states = batch_of_samples

        #convert all to tensors since we want to do faster matrix operations using a GPU
        #move them to gpus
        states_tensor = torch.tensor(states).to(device)
        actions_tensor = torch.tensor(actions).to(device)
        next_states_tensor = torch.tensor(next_states).to(device)
        rewards_tensor = torch.tensor(rewards).to(device)
        dones_tensor = torch.ByteTensor(dones).to(device) #why bytestensor??????

        #states_tensor is a batch of samples
        #we can do parallel processing
        q_values_returned = updating_network(states_tensor).gather(1,actions_tensor.unsqueeze(-1)).squeeze(-1)
        next_states_q_values = target_network(next_states_tensor.float()).max(1)[0]
        next_states_q_values[dones_tensor] = 0.0
        next_states_q_values = next_states_q_values.detach() #see this detach() in the book

        biased_q_values = rewards_tensor + DISCOUNT_FACTOR * next_states_q_values

        loss_func = torch.nn.MSELoss()
        loss = loss_func(q_values_returned, biased_q_values)
        return loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--cuda", default= False, action="store_true", help = "Enable GPU computations")
    parser.add_argument("--environment", default= DEFAULT_ENVIRONMENT, help= "Name of the environment")
    parser.add_argument("--reward_to_beat", default=DEFAULT_REWARD_TO_BEAT)

    args = parser.parse_args()

    '''Check for cuda devices'''
    print("Current cuda device is ", torch.cuda.current_device())
    print("Total cuda-supporting devices count is ", torch.cuda.device_count())
    print("Current cuda device name is ", torch.cuda.get_device_name(torch.cuda.current_device()))
    device = torch.device("cuda" if args.cuda else "cpu")

    '''Get the game environment and wrap it up with different wrappers'''
    env = Wrappers_for_hacks_for_better_performance.make_env(args.environment)

    '''Create both UPDATING and TARGET networks'''
    updating_DQN_network = DQN_model.DQN_Model(env.observation_space.shape, env.action_space.n).float().to(device)
    target_DQN_network = DQN_model.DQN_Model(env.observation_space.shape, env.action_space.n).float().to(device)
    print(target_DQN_network)
    print(updating_DQN_network)

    summary_writer = SummaryWriter(comment="-"+args.environment)


    exp_replay_buffer = Experience_Replay_Buffer(REPLAY_BUFFER_SIZE)

    dqn_agent = Agent(env, exp_replay_buffer)


    epsilon = INITIAL_EPSILON_FOR_EXPLORATION

    optimizer = torch.optim.SGD(updating_DQN_network.parameters(), lr = LEARNING_RATE)

    total_rewards = []
    frame_count = 0
    ts_frame = 0
    ts = time.time()
    current_best_mean_reward = None

    '''Run the training loop until the DQN agent converges.'''
    while True:

        frame_count += 1

        #update the episilon-greedy parameter
        epsilon = max(FINAL_EPSILON, INITIAL_EPSILON_FOR_EXPLORATION - frame_count / LAST_FRAME_TO_DECAY_EPSILON) #wont go below final epsilon

        #get the new reward accoding to the updated epsilon
        reward = dqn_agent.step(updating_DQN_network, epsilon, device)

        #If the game ends
        if reward is not None:

            total_rewards.append(reward)
            speed = (frame_count - ts_frame)/(time.time() - ts)
            ts_frame = frame_count
            ts = time.time()

            #Take the mean total reward of the last 100 episodes, will be used to check for convergence
            mean_reward = np.mean(total_rewards[-100:])

            print("Current Episode ends in frame no: %d ,%d games played., mean reward of last 100 episodes: %.3f, Epsilon used: %.2f, Processing speed: %.2f frames per sec" % (frame_count, len(total_rewards), mean_reward, epsilon, speed))

            #Add values to the SummaryWriter
            summary_writer.add_scalar("epsilon", epsilon, frame_count)
            summary_writer.add_scalar("speed", speed, frame_count)
            summary_writer.add_scalar("reward_100", mean_reward, frame_count)
            summary_writer.add_scalar("reward", reward, frame_count)

            #Save model until the env is solved
            if current_best_mean_reward is None or current_best_mean_reward < mean_reward:
                torch.save(updating_DQN_network.state_dict(), args.environment + "-best.pth")
                if current_best_mean_reward is not None:
                    print("Best mean reward updated from %.3f to %.3f" % (current_best_mean_reward, mean_reward))
                current_best_mean_reward = mean_reward
            if mean_reward > args.reward_to_beat:
                print("Solved the Atari game in %d frames." % frame_count)
                break

        #If not enough samples in the replay buffer, do not update the updating network
        if len(exp_replay_buffer) < REPLAY_SIZE_BEFORE_FIRST_MODEL_UPDATE:
            #Do not try to update the DQN network , skip the update code in the current iteration and go to next iteration
            continue

        #Update target network at certain intervals to stablilize learning
        if frame_count % UPDATE_TARGET_NETWORK_FREQUENCY == 0:
            target_DQN_network.load_state_dict(updating_DQN_network.state_dict())

        #Update the updating network
        optimizer.zero_grad()
        batch = exp_replay_buffer.sample_an_experience(BATCH_SIZE)
        loss_t = dqn_agent.calculate_error(batch, updating_DQN_network, target_DQN_network, device=device)
        loss_t.backward()
        optimizer.step()

    summary_writer.close()








        #check that to_device() thing




