# Week 5: Real-World Projects

## Module Overview

Apply everything you've learned to build production-quality applications.

---

## Project 1: Image Blur ⭐⭐⭐
**File:** `project01-image-blur.cu`
**Duration:** 4-6 hours

**What you'll build:**
- Gaussian blur filter
- Shared memory optimization
- CPU/GPU comparison
- Real-time capable (1080p @ 30+ fps)

**Techniques used:**
- Shared memory tiling
- Constant memory for filter
- Efficient boundary handling
- Performance measurement

**Run it:**
```bash
cd ~/cuda-learning/05-projects
nvcc -o blur project01-image-blur.cu
./blur
```

**Expected speedup:** 20-50x over CPU

---

## Project 2: GPU Tokenizer & Embeddings ⭐⭐⭐⭐
**File:** `project02-tokenizer-embeddings.cu`
**Duration:** 4-6 hours

**What you'll build:**
- Character/word tokenizer
- Embedding lookup (like word2vec)
- Positional encoding
- Layer normalization
- Foundation for any NLP/LLM system

**Techniques used:**
- Massive parallel lookups
- Fused kernels
- Shared memory for reductions
- Mixed precision (optional)

**Run it:**
```bash
cd ~/cuda-learning/05-projects
nvcc -o tokenizer project02-tokenizer-embeddings.cu
./tokenizer
```

**Expected speedup:** 10-30x over CPU

---

## Project 3: Attention Mechanism ⭐⭐⭐⭐⭐
**File:** `project03-attention-mechanism.cu`
**Duration:** 6-8 hours

**What you'll build:**
- Scaled dot-product attention
- Multi-head attention
- Causal masking (GPT-style)
- Flash Attention concepts
- The heart of transformers!

**Techniques used:**
- Tiled matrix multiplication
- Softmax with numerical stability
- Fused attention kernels
- Memory-efficient algorithms

**Run it:**
```bash
cd ~/cuda-learning/05-projects
nvcc -o attention project03-attention-mechanism.cu
./attention
```

**Expected speedup:** 20-100x over CPU
**Note:** This is what powers ChatGPT, BERT, and all modern LLMs!

---

## Project 4: GPU Hash Table ⭐⭐⭐⭐
**File:** `project04-gpu-hash-table.cu`
**Duration:** 5-6 hours

**What you'll build:**
- High-performance concurrent hash table
- Linear probing & cuckoo hashing
- Lock-free operations
- Database operations on GPU

**Techniques used:**
- Atomic operations mastery
- Memory efficiency
- Load balancing
- Collision resolution

**Run it:**
```bash
cd ~/cuda-learning/05-projects
nvcc -o hashtable project04-gpu-hash-table.cu
./hashtable
```

**Expected performance:** 100M+ ops/second

---

## Project 5: Graph Algorithms ⭐⭐⭐⭐⭐
**File:** `project05-graph-algorithms.cu`
**Duration:** 6-8 hours

**What you'll build:**
- Breadth-First Search (BFS)
- Single Source Shortest Path
- PageRank algorithm
- CSR graph representation

**Techniques used:**
- Irregular parallelism
- Work-efficient algorithms
- Frontier-based processing
- Load balancing

**Run it:**
```bash
cd ~/cuda-learning/05-projects
nvcc -o graphs project05-graph-algorithms.cu
./graphs
```

**Expected speedup:** 20-50x over CPU

---

## Project 6: Sparse Matrix Operations ⭐⭐⭐⭐⭐
**File:** `project06-sparse-matrix.cu`
**Duration:** 5-7 hours

**What you'll build:**
- Sparse matrix formats (COO, CSR, ELL)
- SpMV optimization
- Format conversion
- cuSPARSE integration

**Techniques used:**
- Memory layout optimization
- Irregular data structures
- Format selection
- Library integration

**Run it:**
```bash
cd ~/cuda-learning/05-projects
nvcc -o sparse project06-sparse-matrix.cu -lcusparse
./sparse
```

**Expected performance:** 10-100x speedup for >90% sparse matrices

---

## Project 7: GPU Regular Expressions ⭐⭐⭐⭐⭐
**File:** `project07-gpu-regex.cu`
**Duration:** 5-6 hours

**What you'll build:**
- DFA-based regex engine
- Multi-pattern matching
- Log parsing at scale
- Text validation pipeline

**Techniques used:**
- Finite automata on GPU
- Warp-collaborative matching
- String processing optimization
- Real-world text mining

**Run it:**
```bash
cd ~/cuda-learning/05-projects
nvcc -o regex project07-gpu-regex.cu
./regex
```

**Expected performance:** Process millions of strings/second

---

## Project 8: Convolutional Neural Network ⭐⭐⭐⭐⭐
**File:** `project08-cnn.cu`
**Duration:** 6-8 hours

**What you'll build:**
- Convolution layers (naive, im2col, tiled)
- Pooling operations
- Activation functions
- Complete CNN architecture

**Techniques used:**
- Tensor operations
- Memory layout optimization
- Algorithm transformation
- cuDNN comparison

**Run it:**
```bash
cd ~/cuda-learning/05-projects
nvcc -o cnn project08-cnn.cu -lcudnn -lcublas
./cnn
```

**Expected speedup:** 10-50x over CPU, understand how PyTorch works!

---

## Project Development Process

### 1. Plan
- Define inputs/outputs
- Identify parallel portions
- Choose algorithms
- Estimate performance

### 2. Implement
- Start with CPU version
- Port to naive GPU
- Add optimizations incrementally
- Test at each step

### 3. Optimize
- Profile with NSight
- Identify bottlenecks
- Apply optimizations:
  - Shared memory
  - Coalescing
  - Occupancy tuning
  - Streams

### 4. Verify
- Correctness testing
- Performance benchmarking
- Compare to CPU
- Document results

---

## Project Showcase Template

### Title
**Description:** One sentence

**Performance:**
- Problem size: X
- CPU time: Y ms
- GPU time: Z ms
- Speedup: W x

**Techniques:**
- List optimizations used

**Code:**
- Link to source

**Visualization:**
- Screenshots/videos

---

## Building Your Portfolio

1. **GitHub Repository:**
   - Clean, documented code
   - README with results
   - Performance graphs
   - Usage instructions

2. **Blog Post:**
   - Explain the problem
   - Show optimization process
   - Include benchmarks
   - Share insights

3. **LinkedIn/Resume:**
   - "Achieved 50x speedup..."
   - "Optimized GPU kernel to..."
   - "Implemented production-ready..."

---

## Performance Targets

### Image Processing (1920x1080)
- CPU: ~50-100 ms per frame
- GPU: ~1-5 ms per frame
- **Target: 30+ fps (< 33ms)**

### N-Body Simulation
- CPU: ~10K bodies at 1 fps
- GPU: ~100K bodies at 60 fps
- **Target: Real-time with 100K+ bodies**

### Matrix Multiplication (1024x1024)
- CPU: ~500 ms
- GPU naive: ~50 ms
- GPU optimized: ~5 ms
- cuBLAS: ~2 ms
- **Target: Within 2x of cuBLAS**

---

## Resources

### Image Processing
- OpenCV (for I/O)
- stb_image.h (lightweight)
- NPP library (NVIDIA Performance Primitives)

### Scientific Computing
- cuBLAS, cuSPARSE
- Thrust for quick prototyping

### Graphics
- OpenGL/CUDA interop
- Optix for ray tracing

### Deep Learning
- cuDNN
- TensorRT

---

## Certification & Next Steps

### You're Ready For:
- ✓ NVIDIA Deep Learning Institute certification
- ✓ Contributing to open-source CUDA projects
- ✓ GPU programming job interviews
- ✓ Research in GPU computing

### Advanced Topics (Self-Study):
- Multi-GPU programming
- NCCL (multi-GPU communication)
- GPU Direct RDMA
- Custom CUDA C++ templates
- Kernel fusion techniques

---

## Congratulations!

You've completed the CUDA learning curriculum!

**What you've mastered:**
- ✓ CUDA fundamentals
- ✓ Memory optimization
- ✓ Performance tuning
- ✓ Advanced features
- ✓ Real-world applications

**Your skills:**
- Write efficient GPU kernels
- Achieve 10-100x CPU speedups
- Profile and optimize code
- Use CUDA libraries
- Build production applications

**You're now a GPU programmer.** 🚀

---

## Share Your Journey

- Post projects on GitHub
- Write blog posts
- Help others learn
- Contribute to CUDA community

**The best way to master something is to teach it.**
