#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Martin Timms
# Created Date: 15th December 2022
# License: BSD-3-Clause License
# Organisation:
# Project: https://github.com/Electro-resonance/PyTorchTensorBenchmark
# Description: Simple Benchmark test of subsequent multiplying of large tensors
# requiring enough capacity to reach 100% processing capacity on GPUs and CPUs.
# =============================================================================

import os
from prettytable import PrettyTable as pt
from prettytable import SINGLE_BORDER
import psutil
import random
from si_prefix import si_format
import time
import torch

print("Torch version:",torch.__version__)
print("Cuda devices:",torch.cuda.device_count())
print("Cuda initialised:",torch.cuda.is_initialized())
print("Cuda GPU generic data type:",torch.get_autocast_gpu_dtype())
print("Cuda available:",torch.cuda.is_available())
if(torch.cuda.is_available()):
    print('Current Cuda device:', torch.cuda.current_device())
    print('GPU properties:',torch.cuda.get_device_properties(0))

try:
    # Attempt to use CUDA
    cuda_tensor = torch.zeros(1).cuda()
    print("CUDA Tensor successfully created.")
except AssertionError as e:
    # Handle the case where CUDA is not available
    print("CUDA is not available:", e)

#For Cuda GPU tests, requires CUDA to be installed and python needs to be installed from conda
# Ensure to install Cuda for your GPU and OS.
# For example: https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_network

#Check GPU details from command window:
# nvcc --version
# nvidia-smi

# Install PyTorch precompiled with Cuda support, for example using Anaconda:
# ..\..\Scripts\conda.exe update -n base -c defaults conda
# ..\..\Scripts\conda.exe install prettytable psutil pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia


#Pytorch Documentation: https://pytorch.org/docs/stable/tensors.html

def benchmark_tensor_addition(iterations, tensor_size, tensor_type, debug=False):
    print("Cuda devices:",torch.cuda.device_count())

    if(tensor_type.is_floating_point):
        max_value = torch.finfo(tensor_type).max
    else:
        max_value = torch.iinfo(tensor_type).max

    type_measure_dict={
        torch.float16 : 'Flop',
        torch.float32: 'Flop',
        torch.float64: 'Flop',
        torch.uint8: 'op',
        torch.int8: 'op',
        torch.int16: 'op',
        torch.int32 : 'op',
    }

    if torch.cuda.is_available():
        device='cuda:0'
        gpu=True
    else:
        device = 'cpu'
        gpu=False

    shape = (tensor_size,)
    accumulator_tensor = torch.zeros(shape, dtype=tensor_type,device=device)
    if(tensor_type.is_floating_point):
        multiplier=random.random()*1000
        incrementor_tensor = torch.rand(size=shape, dtype=tensor_type, device=device)
        incrementor_tensor *=multiplier
    else:
        incrementor_tensor= torch.randint(max_value,size=shape, dtype=tensor_type,device=device)

    print("Start Accumulator Value: ",accumulator_tensor,
          ' Device: ',accumulator_tensor.device,
          ' Shape:',accumulator_tensor.shape)
    print("Increment: ",incrementor_tensor,
          ' Device: ',incrementor_tensor.device,
          ' Shape:',incrementor_tensor.shape)
    print()
    print('---->')
    print('Running',iterations,'iterations across a tensor of size',tensor_size,'words of type',tensor_type,'on the',device )
    if(gpu):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        start=time.time()
    for iteration in range(1,iterations+1):
        accumulator_tensor+=incrementor_tensor

        if(debug):
            print('Accumulator: ',accumulator_tensor,
                  ' Device: ',accumulator_tensor.device,
                  ' Shape:',accumulator_tensor.shape,
                  ' Tensor Iteration:', iteration,
                  ' Type:', accumulator_tensor.dtype)
        else:
            print('Iteration: ',iteration, end="\r")

    if(gpu):
        end.record()
        print('\nWaiting for the GPU to complete operations')
        gpu_memory_usage=torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024
        # Wait cuda completion
        torch.cuda.synchronize()
        print('<----')
        print()
    else:
        end=time.time()
        gpu_memory_usage=0
        print('<----')
        print()
    pid = os.getpid()
    python_process = psutil.Process(pid)
    cpu_memory_usage = python_process.memory_info()[0] / 1024 / 1024 / 1024

    #Following waits for the result before printing
    print("Final Accumulator Value: ",accumulator_tensor,
          '\nDevice: ',accumulator_tensor.device,
          ' Shape:',accumulator_tensor.shape,
          ' Tensor Addition Iterations:',iterations,
          ' Type:',accumulator_tensor.dtype)

    if(gpu):
        completion_secs=start.elapsed_time(end)/1000
    else:
        completion_secs = end - start

    ops_second=(iterations*tensor_size)/completion_secs
    ops_second_string=(si_format(ops_second,3)+type_measure_dict[tensor_type]+'s').strip()
    print('\nCompletion time:', round(completion_secs, 3), ' seconds')
    return [completion_secs, ops_second, ops_second_string, tensor_type, iterations, tensor_size, gpu_memory_usage, cpu_memory_usage]

if __name__ == '__main__':
    results=[]
    torchtypes=[ torch.float64,
                 torch.float32,
                 torch.int32,
                 torch.int16,
                 torch.int8,
                 torch.uint8]

    for torchtype in torchtypes:
        print('-' * 80)
        if(torchtype==torch.float64):
            tensor_size = 250000000 #smaller set so that GPU does not run out of memory
        else:
            tensor_size = 500000000
        #Run a benchmark test for each numeric type
        results.append(benchmark_tensor_addition(iterations=500,
                                                 tensor_size=tensor_size,
                                                 tensor_type=torchtype,
                                                 debug=False))

    print('-'*80)
    print('Benchmark Summary')
    print('-' * 80)
    table_fields=['Data Type','Iterations','Tensor Size','Completion Time','Operations / sec','GPU Mem. Usage','CPU Mem. Usage']
    tb = pt(field_names=table_fields, align='l', width=15, max_table_width=120)
    tb.set_style(SINGLE_BORDER)
    for result in results:
        table_row = [result[3],
                     str(result[4]),
                     str(result[5]),
                     str(round(result[0],3))+' seconds',
                     result[2],
                     "%0.3fGB" % result[6],
                     "%0.3fGB" % result[7]]
        tb.add_row(table_row)
    print(tb)

