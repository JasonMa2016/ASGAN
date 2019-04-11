# does deque really keep

from collections import deque
import random
import torch

memory = deque(maxlen=12)

# add to memory
G_result = torch.randn(21,3*28*28).cuda()
for i in range(G_result.shape[0]):
    memory.append(G_result[i].detach())

print('deque elts on cuda?', memory[0].is_cuda)
print('deque elts require grad?', memory[0].requires_grad)

samples = random.sample(memory, int(21/2)+1)
samples = torch.stack(samples)
print('samples on cuda?', samples.is_cuda)
print(G_result.shape, samples.shape)
