import numpy as np
def downsample(image):
    # Take only alternate pixels - basically halves the resolution of the image (which is fine for us)
    return image[::2, ::2, :]

def remove_color(image):
    """Convert all color (RGB is the third dimension in the image)"""
    return image[:, :, 0]

def remove_background(image):
    image[image == 144] = 0
    image[image == 109] = 0
    return image

def preprocess_observations(input_observation,prev_processed_observation,input_dimensions):
    processed_observations = input_observation[0:170,0:160]
    processed_observations = downsample(processed_observations)
    processed_observations = remove_color(processed_observations)
    processed_observations = remove_background(processed_observations)
    processed_observations[processed_observations != 0] = 1
    processed_observations = processed_observations.astype(np.float).ravel()

    if prev_processed_observation is not None:
        input_observation = processed_observations - prev_processed_observation
    else:
        input_observation = np.zeros(input_dimensions)
    prev_processed_observation = processed_observations
    return input_observation, prev_processed_observation

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def relu(vector):
    vector[vector < 0] = 0
    return vector

def apply_neural_nets(observation_matrix, weights):
    hidden_layer_values = np.dot(weights['1'], observation_matrix)
    hidden_layer_values = relu(hidden_layer_values)
    output_layer_values = np.dot(hidden_layer_values, weights['2'])
    output_layer_values = sigmoid(output_layer_values)
    return hidden_layer_values, output_layer_values

def sum_probability(probability):
    soma = 0
    for item in probability:
        soma+=item
    probability = probability/soma
    return probability
def choose_action(probability):
    probability = sum_probability(probability)
    random_value = np.random.uniform()
    list_values = probability.tolist()
    soma1 = list_values[0] + list_values[1]
    soma2 = soma1 + list_values[2]
    if random_value < list_values[0]:
        return 1
    if random_value>=list_values[0] and random_value<soma1:
        return 2
    if random_value>=soma1 and random_value<soma2:
        return 3
    else:
        return 4
def discount_rewards(rewards, gamma):
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, rewards.size)):
        if rewards[t] != 0:
            running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards
def discount_with_rewards(gradient_log_p, episode_rewards, gamma):
    discounted_episode_rewards = discount_rewards(episode_rewards, gamma)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_episode_rewards -= np.mean(discounted_episode_rewards)
    discounted_episode_rewards /= np.std(discounted_episode_rewards)
    return gradient_log_p * discounted_episode_rewards

def compute_gradient(gradient_log_p, hidden_layer_values, observation_values, weights):
    delta_L = gradient_log_p
    dC_dw2 = np.dot(hidden_layer_values.T, delta_L) #.ravel()

    delta_l2 = np.dot(delta_L, weights['2'].T)
    delta_l2 = relu(delta_l2)
    dC_dw1 = np.dot(delta_l2.T, observation_values)
    return {
        '1': dC_dw1,
        '2': dC_dw2
    }
def update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate):
    epsilon = 1e-5
    for layer_name in weights.keys():
        g = g_dict[layer_name]
        expectation_g_squared[layer_name] = decay_rate * expectation_g_squared[layer_name] + (1 - decay_rate) * g**2
        weights[layer_name] += (learning_rate * g)/(np.sqrt(expectation_g_squared[layer_name] + epsilon))
        g_dict[layer_name] = np.zeros_like(weights[layer_name]) # reset batch gradient buffer
