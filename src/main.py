'''
Code inspired by: https://github.com/python-engineer/snake-ai-pytorch/blob/main/agent.py
'''


import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from environment import Environment


MAX_MEMORY = 100000
BATCH_SIZE = 100
LR = 0.001


class Linear_QNetwork(nn.Module):

    def __init__(self, lr, gamma, input_size, hidden_size, output_size):
        super().__init__()
        # Hyper parameters
        self.lr = lr
        self.gamma = gamma

        # 1 input layer, 2 hidden layers, 1 output layer
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

        # We use the Adam optimization algorithm
        self.optimizer = optim.Adam(self.parameters(), lr=LR)

        # We use the Mean Squared Error loss function
        self.loss = nn.MSELoss()

    def forward(self, state):
        '''
        Forward propagation fucntion that passes the input through the network and produces an output.
        '''

        # We use the relu activasion fucntion
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        actions = self.linear3(x)
        
        return actions

    def train_step(self, old_state, new_state, action, reward, done):
        '''
        Function that trains the network.
        '''

        # Convert to tensor with shape (n, x). A tensor varaible is used by the pytorch library
        old_state   = torch.tensor(old_state, dtype=torch.float)
        new_state   = torch.tensor(new_state, dtype=torch.float)
        action      = torch.tensor(action, dtype=torch.long)
        reward      = torch.tensor(reward, dtype=torch.float)

        # If dimension is of length 1 (x), only true when training short memory
        if len(old_state.shape) == 1:
            # Convert to shape (1, x)
            old_state   = torch.unsqueeze(old_state, 0)
            new_state   = torch.unsqueeze(new_state, 0)
            action      = torch.unsqueeze(action, 0)
            reward      = torch.unsqueeze(reward, 0)
            done = (done, ) # define a tuple with 1 value

        # Predicted the Q values with old_state
        old_Q = self.forward(old_state)

        # Clone is possible with a pytorch tensor, we only want the shape
        target = old_Q.clone()
        
        # Calculate the new Q value using the Bellman equation
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.forward(new_state[idx])) # Bellman equation

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # Empty the gradiant, pytorch spesific syntax
        self.optimizer.zero_grad()

        # Calculate the loss function on the new and old Q values
        loss = self.loss(target, old_Q)

        # Backward propagation to update the gradiant of the network
        loss.backward()

        self.optimizer.step()


class Agent():

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.mem_cntr = 0
        
        self.model = Linear_QNetwork(LR, self.gamma, 11, 256, 3)

        # Memory is a matrix
        self.old_state_memory   = np.zeros((MAX_MEMORY, 11), dtype=np.int32)
        self.new_state_memory   = np.zeros((MAX_MEMORY, 11), dtype=np.int32)
        self.action_memory      = np.zeros((MAX_MEMORY, 3 ), dtype=np.int32)
        self.reward_memory      = np.zeros( MAX_MEMORY,      dtype=np.int32)
        self.done_memory        = np.zeros( MAX_MEMORY,      dtype=np.bool_)

    def remember(self, old_state, new_state, action, reward, done):
        '''
        Function that populates the memory matrix with the given inputs.
        '''

        # This will wrap around, using modulo, and pop left
        index = self.mem_cntr % MAX_MEMORY

        # Populate the memory
        self.old_state_memory[index] = old_state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.done_memory[index] = done

        self.mem_cntr += 1

    def train_long_memory(self):
        '''
        Function that trains the network with x amount of batches, equal to the BATCH_SIZE variable.
        '''

        # If memory is not filled enough to train, return
        if self.mem_cntr < BATCH_SIZE:
            return

        # Calculate the last filled memory slot
        max_mem = min(self.mem_cntr, MAX_MEMORY)
        # Pick random memories to learn from, will be an array of random memory indicies
        batch = np.random.choice(max_mem, BATCH_SIZE, replace=False)
        
        # Matrix of memories to learn from
        old_states  = self.old_state_memory[batch]
        new_states  = self.new_state_memory[batch]
        actions     = self.action_memory[batch]
        rewards     = self.reward_memory[batch]
        dones       = self.done_memory[batch]

        self.model.train_step(old_states, new_states, actions, rewards, dones)

    def train_short_memory(self, old_state, new_state, action, reward, done):
        '''
        Function that trains the network with one batch.
        '''

        self.model.train_step(old_state, new_state, action, reward, done)
    
    def get_action(self, state):
        '''
        Function that chooses the next action, chooses either at random or via the network.
        '''

        action = [0,0,0]
        self.epsilon = 80 - self.n_games
        if self.epsilon > random.randint(0, 200): # Exploration
            move = random.randint(0, 2)
            action[move] = 1
        else: # Exploitation
            state = torch.tensor(state, dtype=torch.float)
            prediction = self.model.forward(state)
            move = torch.argmax(prediction).item()
            action[move] = 1

        return action


def train():

    record = 0
    agent = Agent()
    env = Environment(width=600, height=400, size=10, speed=60)

    while True:
        # Get old state
        old_state = env.get_state()

        # Get action
        action = agent.get_action(old_state)

        # Perform action
        done, score, reward = env.step(action)

        # Get new state
        new_state = env.get_state()

        # Train short memory
        agent.train_short_memory(old_state, new_state, action, reward, done)

        # Save results from current iteration
        agent.remember(old_state, new_state, action, reward, done)

        if done:
            # Train long memory
            env.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score

            print('Game', agent.n_games, 'Score', score, 'Record:', record)


if __name__ == '__main__':
    train()
