from collections import deque
import random

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def append(self, snippet):
        self.memory.append(snippet)

    def last(self):
        return [self.memory[-1]]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)