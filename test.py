import torch
import time

# Create a large random tensor
data = torch.randn(10000, 10000)

# Perform matrix multiplication on CPU
start = time.time()
result = torch.matmul(data, data)
print("CPU Time:", time.time() - start)

# Perform matrix multiplication on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)  # Move data to GPU
start = time.time()
result = torch.matmul(data, data)
print("GPU Time:", time.time() - start)
