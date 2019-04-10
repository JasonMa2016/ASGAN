# does deque really keep

from collections import deque
import random
import torch

memory = deque()

# add to memory
G_result = torch.randn(21,3*28*28)
for i in range(G_result.shape[0]):
    memory.append(G_result[i].detach())

print('deque elts on cuda?', memory[0].is_cuda)

samples = random.sample(memory, int(mini_batch/2)+1)
samples = torch.stack(samples)
print('samples on cuda?', samples.is_cuda)
print(G_result.shape, samples.shape)