# PyTorch Tensor Benchmarking
Benchmarking of tensor computation with very large tensors (half a billion words in size). The benchmark load is made
by creating a two tensors, one an accumulator and one containing random numbers. The PyTorch code is then called to 
sequentially add the tensor containing random numbers to the accumulator tensor and then repeating the process 500 
times for each data type being tested. Doing so for a RTX 3060 GPU or I7 processor presents a near 100% load.

### Example result on an RTX 3060 GPU running at 2.0 GHz 
```
┌───────────────┬────────────┬─────────────┬─────────────────┬──────────────────┬────────────────┬────────────────┐
│ Data Type     │ Iterations │ Tensor Size │ Completion Time │ Operations / sec │ GPU Mem. Usage │ CPU Mem. Usage │
├───────────────┼────────────┼─────────────┼─────────────────┼──────────────────┼────────────────┼────────────────┤
│ torch.float64 │ 500        │ 250000000   │ 9.498 seconds   │ 13.161 GFlops    │ 3.727GB        │ 1.241GB        │
│ torch.float32 │ 500        │ 500000000   │ 9.488 seconds   │ 26.348 GFlops    │ 3.727GB        │ 1.241GB        │
│ torch.int32   │ 500        │ 500000000   │ 9.452 seconds   │ 26.450 Gops      │ 3.727GB        │ 1.241GB        │
│ torch.int16   │ 500        │ 500000000   │ 4.777 seconds   │ 52.339 Gops      │ 1.863GB        │ 1.241GB        │
│ torch.int8    │ 500        │ 500000000   │ 2.371 seconds   │ 105.430 Gops     │ 0.931GB        │ 1.241GB        │
│ torch.uint8   │ 500        │ 500000000   │ 2.369 seconds   │ 105.531 Gops     │ 0.931GB        │ 1.241GB        │
└───────────────┴────────────┴─────────────┴─────────────────┴──────────────────┴────────────────┴────────────────┘
```

### Example result on an I7-12700H running at 3.68 GHz
```
┌───────────────┬────────────┬─────────────┬─────────────────┬──────────────────┬────────────────┬────────────────┐
│ Data Type     │ Iterations │ Tensor Size │ Completion Time │ Operations / sec │ GPU Mem. Usage │ CPU Mem. Usage │
├───────────────┼────────────┼─────────────┼─────────────────┼──────────────────┼────────────────┼────────────────┤
│ torch.float64 │ 500        │ 250000000   │ 66.775 seconds  │ 1.872 GFlops     │ 0.000GB        │ 3.832GB        │
│ torch.float32 │ 500        │ 500000000   │ 66.337 seconds  │ 3.769 GFlops     │ 0.000GB        │ 3.832GB        │
│ torch.int32   │ 500        │ 500000000   │ 66.148 seconds  │ 3.779 Gops       │ 0.000GB        │ 3.833GB        │
│ torch.int16   │ 500        │ 500000000   │ 33.331 seconds  │ 7.500 Gops       │ 0.000GB        │ 1.970GB        │
│ torch.int8    │ 500        │ 500000000   │ 22.933 seconds  │ 10.901 Gops      │ 0.000GB        │ 1.038GB        │
│ torch.uint8   │ 500        │ 500000000   │ 21.306 seconds  │ 11.734 Gops      │ 0.000GB        │ 1.038GB        │
└───────────────┴────────────┴─────────────┴─────────────────┴──────────────────┴────────────────┴────────────────┘
```


## Acknowledgements:
PyTorch for the Tensor language:
https://pytorch.org/

## Installing

PyTorch is easiest to run under an anaconda install and will also require a Nvidia Cuda driver:
https://anaconda.org/pytorch/pytorch
https://developer.nvidia.com/cuda-downloads

The install command for a specific setup can be generated from the first table in the install guide:
https://pytorch.org/

# Useful Links:

###### PyTorch Tensor Documentation
* https://pytorch.org/docs/stable/tensors.html

###### PyTorch
* https://github.com/pytorch/pytorch
* https://pytorch.org/get-started/locally/

# License
Software Licensed under BSD-3-Clause License

# GPU and Processor Usage

Note that this software can when run use the full computing capacity of the CPU and GPU. The user should determine if 
the software is suitable for running on their system and they should ensure due consideration to usage, especially 
with regards to ensuring that GPU, CPU and PC thermal management is enabled and configured as per the manufacturers 
recommendations.

# Disclaimer

[THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.]
