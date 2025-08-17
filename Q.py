import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Define discrete actions: 0=Hold, 1=Buy, 2=Sell
DISCRETE_ACTIONS = [
    np.array([0.0]),  # Hold
    np.array([0.5]),  # Buy
    np.array([1.0])   # Sell
]
NUM_ACTIONS = len(DISCRETE_ACTIONS)

class QNetwork(nn.Module):
    def __init__(self, input_shape, action_space):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape[0], 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, action_space)
        )

    def forward(self, x):
        return self.model(x)

def create_q_model(input_shape, action_space):
    return QNetwork(input_shape, action_space)

class DQNAgent:
    def __init__(self, input_shape, action_space, gamma=0.95, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995, learning_rate=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_shape = input_shape
        self.action_space = action_space
        self.memory = deque(maxlen=10000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = create_q_model(input_shape, action_space).to(self.device)
        self.target_model = create_q_model(input_shape, action_space).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        state = torch.FloatTensor(state).to(self.device)
        if np.random.rand() < self.epsilon:
            action_idx = random.randrange(self.action_space)
        else:
            self.model.eval()
            with torch.no_grad():
                q_values = self.model(state).cpu().numpy()[0]
            action_idx = np.argmax(q_values)
        print(f"Action chosen: {action_idx} (Epsilon: {self.epsilon:.3f})")
        return action_idx, DISCRETE_ACTIONS[action_idx]

    def remember(self, state, action_idx, reward, next_state, done):
        self.memory.append((state, action_idx, reward, next_state, done))
        print(f"Memory size: {len(self.memory)}")

    def replay(self, batch_size):
        if len(self.memory) < batch_size // 2:
            print(f"Replay skipped: Insufficient memory ({len(self.memory)} < {batch_size // 2})")
            return
        minibatch = random.sample(self.memory, min(batch_size, len(self.memory)))
        states = np.array([m[0] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])
        if states.ndim == 3:
            states = states.reshape(states.shape[0], states.shape[-1])
        if next_states.ndim == 3:
            next_states = next_states.reshape(next_states.shape[0], next_states.shape[-1])

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)

        self.model.train()
        targets = self.model(states)
        next_q = self.target_model(next_states).detach()

        for i, (s, a, r, s_next, done) in enumerate(minibatch):
            target = r
            if not done:
                target += self.gamma * torch.max(next_q[i]).item()
            targets[i][a] = target

        self.optimizer.zero_grad()
        loss = self.loss_fn(targets, self.model(states))
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            print(f"Epsilon updated to {self.epsilon:.3f}")

    def get_valuable_memories(self, n=3, profit=0):
        if not self.memory:
            return []
        sorted_memories = sorted(
            self.memory,
            key=lambda x: abs(x[2]) + abs(profit) * 0.1,
            reverse=True
        )
        return sorted_memories[:n]

    def save_model(self, path="dqn_model_final.pt"):
        torch.save(self.model.state_dict(), path)