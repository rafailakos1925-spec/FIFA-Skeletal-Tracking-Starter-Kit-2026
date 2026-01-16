import torch
import time

# Μέγεθος πίνακα (δοκίμασε 1024, 2048, 4096 αν έχεις καλή GPU)
N = 2048
iters = 100

#################################
# CPU
#################################
x_cpu = torch.rand(N, N)
y_cpu = torch.rand(N, N)

# Ζέσταμα
_ = x_cpu @ y_cpu

start = time.perf_counter()
for _ in range(iters):
    z_cpu = x_cpu @ y_cpu
end = time.perf_counter()

cpu_time_ms = (end - start) * 1000 / iters

#################################
# GPU
#################################
x_gpu = x_cpu.cuda()
y_gpu = y_cpu.cuda()

# CUDA events
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# Ζέσταμα
_ = x_gpu @ y_gpu
torch.cuda.synchronize()

start_event.record()
for _ in range(iters):
    z_gpu = x_gpu @ y_gpu
end_event.record()

torch.cuda.synchronize()
gpu_time_ms = start_event.elapsed_time(end_event) / iters

#################################
# Αποτελέσματα
#################################
print(f"CPU χρόνος ανά πράξη: {cpu_time_ms:.3f} ms")
print(f"GPU χρόνος ανά πράξη: {gpu_time_ms:.3f} ms")
print(f"Speedup GPU: {cpu_time_ms / gpu_time_ms:.2f}x")
