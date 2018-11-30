import gym
env = gym.make('MsPacman-v0')
highscore = 0
for i_episode in range(20): # run 20 episodes
  observation = env.reset()
  points = 0 # keep track of the reward each episode
  while True: # run until episode is done
    env.render()
    action = 4 # if angle if positive, move right. if angle is negative, move left
    observation, reward, done, info = env.step(action)
    points += reward
    if done:
      if points > highscore: # record high score
        highscore = points
        break

#1 Cima
#2 Direita
#3 Esquerda
#4 Baixo
