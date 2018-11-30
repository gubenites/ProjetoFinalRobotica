import gym
from colour import Color
from funcs import downsample,remove_color,remove_background,preprocess_observations,sigmoid,relu,apply_neural_nets,sum_probability,choose_action,discount_with_rewards,compute_gradient,update_weights
import cv2
import numpy as np
import os.path
env = gym.make('MsPacman-v0')

points = 0

highscore = 0
flat_list = []
color_list = []

episode_number = 0
batch_size = 5
gamma = 0.99
decay_rate = 0.99
num_hidden_layer_neurons = 500
input_dimensions = 80 * 85
learning_rate = 1e-4

reward_sum = 0
running_reward = None
prev_processed_observation = None


if os.path.isfile('weights'):
    with open('weights','rb') as data:
        weights = {}
        weights['1'] = np.load(data)
    with open('weights2','rb') as data2:
        weights['2'] = np.load(data2)

else:
    weights = {
        '1' :np.random.randn(num_hidden_layer_neurons,input_dimensions) / np.sqrt(input_dimensions),
        '2' :np.random.randn(num_hidden_layer_neurons,4) / np.sqrt(num_hidden_layer_neurons)
    }

expectation_g_squared = {}
g_dict = {}
for layer_name in weights.keys():
    expectation_g_squared[layer_name] = np.zeros_like(weights[layer_name])
    g_dict[layer_name] = np.zeros_like(weights[layer_name])

observation = env.reset()
episodes_hidden_layer_values,episode_observations,episode_gradient_log_ps,episode_rewards = [],[],[],[]
while  True:

    env.render('human')
    processed_observations, prev_processed_observation = preprocess_observations(observation, prev_processed_observation, input_dimensions)
    hidden_layer_values, move_probability = apply_neural_nets(processed_observations, weights)

    episode_observations.append(processed_observations)
    episodes_hidden_layer_values.append(hidden_layer_values)

    action = choose_action(move_probability)

    observation, reward, done, info = env.step(action)
    reward_sum += reward
    episode_rewards.append(reward)

    if action == 1:
        fake_label = np.array([1,0,0,0])
    if action == 2:
        fake_label = np.array([0,1,0,0])
    if action == 3:
        fake_label = np.array([0,0,1,0])
    if action == 4:
        fake_label = np.array([0,0,0,1])

    if done:
        episode_number+=1

        loss_function_gradient = fake_label - move_probability
        episode_gradient_log_ps.append(loss_function_gradient)

        episodes_hidden_layer_values = np.vstack(episodes_hidden_layer_values)
        episode_observations = np.vstack(episode_observations)
        episode_gradient_log_ps = np.vstack(episode_gradient_log_ps)
        episode_rewards = np.vstack(episode_rewards)

        episode_gradient_log_ps_discounted = discount_with_rewards(episode_gradient_log_ps,episode_rewards,gamma)

        gradient = compute_gradient(
            episode_gradient_log_ps_discounted,
            episodes_hidden_layer_values,
            episode_observations,
            weights
        )
        if episode_number % batch_size == 0:
            update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate)
            with open('weights','wb') as data:
                np.save(data,weights['1'])
            with open('weights2','wb') as data2:
                np.save(data2, weights['2'])
        episodes_hidden_layer_values,episode_observations,episode_gradient_log_ps,episode_rewards = [],[],[],[]
        observation = env.reset()
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print ('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
        reward_sum = 0
        prev_processed_observations = None

env.close()  # https://github.com/openai/gym/issues/893
