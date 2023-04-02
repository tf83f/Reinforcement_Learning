#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pettingzoo.classic import connect_four_v3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym 
from collections import deque
import random

class connect_four_v3(gym.Env):
    def __init__(self):
        self.board = np.zeros((6,7))
        self.current_player = 1
        self.action_space = gym.spaces.Discrete(7)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(6, 7), dtype=np.float32)
        
    def step(self, action):
        if self.board[0][action] != 0:
            return self.last(), -10, True, {}
        for row in range(5,-1,-1):
            if self.board[row][action] == 0:
                self.board[row][action] = self.current_player
                break
        reward = self._check_winner()
        termination = (reward != 0)
        self.current_player = -self.current_player
        return self.last(), reward, termination, {}
    
    def reset(self):
        self.board = np.zeros((6,7))
        self.current_player = 1
        return self.last()
    
    def render(self):
        print(self.board)
    
    def last(self):
        return np.copy(self.board)
    
    def _check_winner(self):
        for row in range(6):
            for col in range(4):
                if self.board[row][col] != 0 and \
                   self.board[row][col] == self.board[row][col+1] and \
                   self.board[row][col] == self.board[row][col+2] and \
                   self.board[row][col] == self.board[row][col+3]:
                    return 1 if self.current_player == 1 else -1
                
        for row in range(3):
            for col in range(7):
                if self.board[row][col] != 0 and \
                   self.board[row][col] == self.board[row+1][col] and \
                   self.board[row][col] == self.board[row+2][col] and \
                   self.board[row][col] == self.board[row+3][col]:
                    return 1 if self.current_player == 1 else -1
        
        for row in range(3):
            for col in range(4):
                if self.board[row][col] != 0 and \
                   self.board[row][col] == self.board[row+1][col+1] and \
                   self.board[row][col] == self.board[row+2][col+2] and \
                   self.board[row][col] == self.board[row+3][col+3]:
                    return 1 if self.current_player == 1 else -1
        
        for row in range(3, 6):
            for col in range(4):
                if self.board[row][col] != 0 and \
                   self.board[row][col] == self.board[row-1][col+1] and \
                   self.board[row][col] == self.board[row-2][col+2] and \
                   self.board[row][col] == self.board[row-3][col+3]:
                    return 1 if self.current_player == 1 else -1
        
        return 0

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Define the first fully connected layer with input size 6*7 and output size 128
        self.fc1 = nn.Linear(6*7, 128)
        
        # Define the second fully connected layer with input size 128 and output size 128
        self.fc2 = nn.Linear(128, 128)
        
        # Define the third fully connected layer with input size 128 and output size 7
        self.fc3 = nn.Linear(128, 7)

    def forward(self, x):
    
        # Reshape the input tensor to have shape (-1, 6*7)
        x = x.view(-1, 6*7)
        
        # Apply ReLU activation function to the output of the first layer
        x = torch.relu(self.fc1(x))
        
        # Apply ReLU activation function to the output of the second layer
        x = torch.relu(self.fc2(x))
        
        # Compute the output of the third layer without applying an activation function
        x = self.fc3(x)
        
        # Return the output tensor
        return x

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Define the first convolutional layer with input size 1 and output size 16, kernel size 3x3, stride 1x1, and padding 1x1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        
        # Define the second convolutional layer with input size 16 and output size 32, kernel size 3x3, stride 1x1, and padding 1x1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # Define the first fully connected layer with input size 32*6*7 and output size 256
        self.fc1 = nn.Linear(32 * 6 * 7, 256)
        
        # Define the second fully connected layer with input size 256 and output size 7
        self.fc2 = nn.Linear(256, 7)

    def forward(self, x):
    
        # Flatten the input tensor to have shape (-1, 1*6*7)
        x = nn.Flatten()(x)
        
        # Reshape the input tensor to have shape (-1, 1, 6, 7)
        x = x.view(-1, 1, 6, 7)
        
        # Apply ReLU activation function to the output of the first convolutional layer
        x = torch.relu(self.conv1(x))
        
        # Apply ReLU activation function to the output of the second convolutional layer
        x = torch.relu(self.conv2(x))
        
        # Flatten the output tensor to have shape (-1, 32*6*7)
        x = x.view(-1, 32 * 6 * 7)
        
        # Apply ReLU activation function to the output of the first fully connected layer
        x = torch.relu(self.fc1(x))
        
        # Compute the output of the second fully connected layer without applying an activation function
        x = self.fc2(x)
        
        # Return the output tensor
        return x
    
class Player:
    def __init__(self, NN):
        # Initialize the player's hyperparameters
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.999  # Exploration rate decay
        self.gamma = 0.99  # Discount factor
        self.lr = 1e-4  # Learning rate for the optimizer
        self.batch_size = 24  # Batch size for replay
        self.memory = deque(maxlen=10000)  # Replay memory
        self.model = NN  # Neural network to approximate the Q-function
        self.target_model = NN  # Target network to stabilize learning
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)  # Adam optimizer
        self.loss_fn = nn.MSELoss()  # Mean squared error loss function

    def get_action(self, state):
    
        # Choose an action based on the epsilon-greedy policy
        if np.random.rand() <= self.epsilon:
        
            # Take a random action
            return np.random.choice(7)
        else:
        
            # Choose an action based on the Q-values predicted by the neural network
            state = torch.FloatTensor(state)
            q_values = self.model(state)
            return q_values.argmax().item()

    def remember(self, state, action, reward, observation, termination):
    
        # Store the transition in replay memory
        self.memory.append((state, action, reward, observation, termination))

    def replay(self):
    
        # If there are not enough experiences in the memory buffer, return without updating the model
        if len(self.memory) < self.batch_size:
            return
            
        # Randomly sample a batch of experiences from the memory buffer
        batch = random.sample(self.memory, self.batch_size)
        
        # Separate the different components of the batch into separate arrays
        states, actions, rewards, observations, terminations = zip(*batch)
        
        # Convert the arrays into tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        observations = torch.FloatTensor(observations)
        terminations = torch.FloatTensor(terminations)
        
        # Get the Q-values of the current model for the states in the batch
        q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Get the maximum Q-values of the target model for the observations in the batch
        next_q_values = self.target_model(observations).max(1)[0].detach()
        
        # Calculate the target Q-values as a combination of the observed rewards and the discounted future Q-values
        target_q_values = rewards + (1 - terminations) * self.gamma * next_q_values
        
        # Calculate the loss between the predicted Q-values and the target Q-values
        loss = self.loss_fn(q_values, target_q_values.unsqueeze(1))
        
        # Zero out the gradients in the optimizer
        self.optimizer.zero_grad()
        
        # Calculate the gradients of the loss with respect to the model parameters
        loss.backward()
        
        # Update the model parameters using the optimizer
        self.optimizer.step()

    def update_target_model(self):
    
        # Update the target network by copying the parameters from the main network
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
    
        # Decay the exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)

def training_player(Player, env, num_episodes):
    print("Training Deep Q-Learning player against random player during : ", str(num_episodes), " episodes")
    print("     ")
    
    # Set up environment and initial state
    env = env
    state = env.reset()
    
    # Train for the given number of episodes
    for episode in range(num_episodes):
    
        # Reset the environment and set termination to False
        state = env.reset()
        termination = False
        
        # Play until the game is over
        while not termination:
        
            # Get action from the Player's policy
            action = Player.get_action(state)
            
            # Take the action and observe
            observation, reward, termination, _ = env.step(action)
            
            # Store the current
            Player.remember(state, action, reward, observation, termination)
            
            # Update the current state to the next state
            state = observation
            
            # Train the player's Q-network using a batch of samples from memory
            Player.replay()
            
        # Update the target Q-network
        Player.update_target_model()
        
        # Decay the epsilon-greedy exploration rate
        Player.decay_epsilon()
        
        # Print the number of wins every 100 episodes
        if episode % 100 == 0:
            wins = 0
            for _ in range(100):
                state = env.reset()
                termination = False
                while not termination:
                
                    # With a probability of 5%, select a random action from the environment's action space
                    if np.random.rand() <= 0.05:
                        action = env.action_space.sample()
                        
                    # Otherwise, select an action using the Player's policy
                    else:
                        action = Player.get_action(state)
                        
                    # Take the action and observe the next state, reward, and termination signal
                    state, reward, termination, _ = env.step(action)
                    
                # If the Player won the game, increment the win counter
                if reward == 1:
                    wins += 1
                    
            # Print the episode number and the number of wins
            print("Episode: %d, Wins: %d" % (episode, wins))
    print("     ")
    print("End of training")
    print("     ")
    print("_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-")
    print("     ")
    
    # Return the trained Player
    return Player

def simulate_game(num_games, Player1, Player2, env):

    # print the number of games to be played
    print("Running ", str(num_games), " games : ")
    print("     ")
    
    # initialize the wins for both players to zero
    Player1_wins = 0
    Player2_wins = 0
    
    # loop through each game
    for i in range(num_games):
        # create a new instance of the Connect Four environment
        env = connect_four_v3()
        
        # reset the environment and get the initial state
        state = env.reset()
        termination = False
        
        # play the game until it is over
        while not termination:
            # get the current state
            state = env.last()
            
            # determine which player's turn it is and select an action using the corresponding player's policy
            if env.current_player == 1:
                action = Player1.get_action(state)
            else:
                action = Player2.get_action(state)
            
            # take the selected action and observe the resulting state, reward, and termination flag
            observation, reward, termination, _ = env.step(action)
  
        # determine which player won the game and update their respective win counts
        winner = 1 if env.current_player == -1 else -1
        if winner == 1:
            Player1_wins += 1
        else:
            Player2_wins += 1
    
    # calculate the win percentages for each player and print the results
    Player1_win_percent = Player1_wins / num_games * 100
    Player2_win_percent = Player2_wins / num_games * 100
    print(f"Player 1 wins: {Player1_win_percent:.2f}%")
    print(f"Player 2 wins: {Player2_win_percent:.2f}%")

# create a player instance with MLP neural network architecture and train for 5000 episodes
Player_MLP = training_player(Player(MLP()), connect_four_v3(), 5000)

# create another player instance with CNN neural network architecture and train for 5000 episodes
Player_CNN = training_player(Player(CNN()), connect_four_v3(), 5000)

# simulate 1000 games between the two trained players using the connect_four_v3 environment
simulate_game(1000, Player_MLP, Player_CNN, connect_four_v3())

# simulate 1000 games between the two trained players using the connect_four_v3 environment
simulate_game(1000, Player_CNN, Player_MLP, connect_four_v3())

# simulate 1000 games between the two trained players using the connect_four_v3 environment
simulate_game(1000, Player_CNN, Player_CNN, connect_four_v3())

# simulate 1000 games between the two trained players using the connect_four_v3 environment
simulate_game(1000, Player_MLP, Player_MLP, connect_four_v3())










