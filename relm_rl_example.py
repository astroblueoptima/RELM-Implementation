
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Data definition
x_train = torch.tensor(np.linspace(0, 10, 1000)[:, np.newaxis], dtype=torch.float32)
y_train = 2 * x_train

# Neural Network model definition
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Model parameters
input_dim = 1
hidden_dim = 50
output_dim = 1
model = SimpleNN(input_dim, hidden_dim, output_dim)

# Primary Training
epochs = 1000
lr = 0.01
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()

# Reinforcement Learning Environment setup
class SimpleRLEnv:
    # ... [Full Environment Definition as before]

def loss_to_state(loss):
    if loss > 1e5:
        loss = 1e5
    return int(loss * 1000) % n_states

# Initialize the RL environment
env = SimpleRLEnv(model, x_train, y_train, criterion)

# Q-learning agent setup and training loop for RL optimization
class QLearningAgent:
    # ... [Full QLearningAgent Definition as before]

n_actions = 2
n_states = 1000
agent = QLearningAgent(n_actions, n_states)
lr_changes_rl = []

for episode in range(n_episodes):
    state = loss_to_state(env.reset()[0])
    total_reward = 0
    for _ in range(10):
        action = agent.choose_action(state)
        next_state, reward, lr = env.step(action)
        next_state = loss_to_state(next_state[0])
        agent.learn(state, action, reward, next_state)
        state = next_state
        total_reward += reward
    lr_changes_rl.append(lr)

# Results
loss_after_rl = criterion(model(x_train), y_train).item()
