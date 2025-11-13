# CUDA Learning Repository

Welcome to your comprehensive CUDA programming journey! 

## 🚀 Quick Start

```bash
cd ~/cuda-learning
./START
```

## 📚 Complete Curriculum Structure

### Week 1: Fundamentals (01-basics/)
- **[Lesson 01: First Kernel](./01-basics/lesson01-first-kernel.cu)** - Your first GPU program
- **[Lesson 02: Thread & Blocks](./01-basics/lesson02-thread-blocks.cu)** - Understanding parallelism  
- **[Lesson 03: Array Operations](./01-basics/lesson03-array-operation.cu)** - Parallel data processing
- **[Lesson 04: Memory Model](./01-basics/lesson04-memory-model.cu)** - GPU memory hierarchy
- **[Lesson 05: Vector Addition](./01-basics/lesson05-vector-add-from-scratch.cu)** - Complete application

### Week 2: Memory & Optimization (02-memory/)
- **[Lesson 06: Shared Memory](./02-memory/lesson06-shared-memory.cu)** - Fast on-chip memory
- **[Lesson 07: Parallel Reduction](./02-memory/lesson07-reduction.cu)** - Tree-based algorithms
- **[Lesson 08: CUDA Streams](./02-memory/lesson08-streams.cu)** - Asynchronous execution
- **[Lesson 09: Atomic Operations](./02-memory/lesson09-atomic-operations.cu)** - Thread-safe operations
- **[Lesson 10: Texture Memory](./02-memory/lesson10-texture-memory.cu)** - Specialized caching

### Week 3: Advanced Optimization (03-optimization/)
- **[Lesson 11: Warp Primitives](./03-optimization/lesson11-warp-primitives.cu)** - Low-level optimizations
- **[Lesson 12: Parallel Scan](./03-optimization/lesson12-parallel-scan.cu)** - Prefix sum algorithms
- **[Lesson 13: Tensor Cores](./03-optimization/lesson13-tensor-cores.cu)** - AI acceleration hardware

### Week 4: Advanced Features (04-advanced/)
- **[Lesson 14: Dynamic Parallelism](./04-advanced/lesson14-dynamic-parallelism.cu)** - GPU launches kernels
- **[Lesson 15: CUDA Graphs](./04-advanced/lesson15-cuda-graphs.cu)** - Eliminate launch overhead
- **[Lesson 16: Multi-GPU](./04-advanced/lesson16-multi-gpu.cu)** - Scale beyond single GPU

### Week 5: Real-World Projects (05-projects/)
- **[Project 01: Image Blur](./05-projects/project01-image-blur.cu)** - Real-world image processing
- **[Project 02: Tokenizer & Embeddings](./05-projects/project02-tokenizer-embeddings.cu)** - NLP fundamentals
- **[Project 03: Attention Mechanism](./05-projects/project03-attention-mechanism.cu)** - Transformer building blocks
- **[Project 04: GPU Hash Table](./05-projects/project04-gpu-hash-table.cu)** - High-performance data structures
- **[Project 05: Graph Algorithms](./05-projects/project05-graph-algorithms.cu)** - BFS, PageRank, SSSP
- **[Project 06: Sparse Matrices](./05-projects/project06-sparse-matrix.cu)** - Scientific computing
- **[Project 07: GPU Regex](./05-projects/project07-gpu-regex.cu)** - Text processing at scale
- **[Project 08: CNN](./05-projects/project08-cnn.cu)** - Deep learning from scratch
- **[Project 09: Monte Carlo Finance](./05-projects/project09-monte-carlo-finance.cu)** - Financial simulations
- **[Project 10: Scientific FFT](./05-projects/project10-scientific-fft.cu)** - Signal processing

### Week 6: Production & Deployment (06-production/)
- **[Lesson 17: Profiling & Optimization](./06-production/lesson17-profiling-optimization.cu)** - Systematic tuning
- **[Lesson 18: CUTLASS & Templates](./06-production/lesson18-cutlass-templates.cu)** - Reusable kernels
- **[Lesson 19: Error Handling](./06-production/lesson19-error-handling.cu)** - Robust GPU code
- **[Lesson 20: Deployment](./06-production/lesson20-deployment-integration.cu)** - Production systems

## 📈 Learning Path

### For AI/ML Engineers
```
Basics (1-5) → Memory (6,9) → Tensor Cores (13) → 
Projects (2,3,8) → Multi-GPU (16) → Production (17-20)
```

### For HPC Developers
```
Basics (1-5) → Memory (6-10) → Optimization (11-13) →
Projects (5,6,10) → Advanced (14-16) → Production (17-20)
```

### For Systems Programmers
```
Basics (1-5) → Atomic Ops (9) → Warp Primitives (11) →
Projects (4,7) → CUDA Graphs (15) → Production (18-20)
```

## 🎯 What You'll Master

- **20+ Comprehensive Lessons**: From basics to production
- **10 Major Projects**: Real-world applications
- **500+ Exercises**: Hands-on practice
- **10-1000x Performance**: Proven speedups
- **Modern GPU Features**: Including Tensor Cores, CUDA Graphs
- **Production Skills**: Deployment, profiling, error handling

## 💻 Compilation Commands

```bash
# Basic compilation
nvcc -O3 lesson.cu -o lesson

# With debugging
nvcc -g -G lesson.cu -o lesson

# With libraries
nvcc -O3 lesson.cu -lcublas -lcusparse -o lesson

# For Tensor Cores (Volta+)
nvcc -O3 -arch=sm_70 lesson.cu -o lesson
```

## 📊 Performance You'll Achieve

- Vector Operations: 10-100x speedup
- Matrix Operations: 50-500x with optimization
- Deep Learning: Understanding how PyTorch/TensorFlow work
- Text Processing: Millions of strings/second
- Scientific Computing: Real-time simulations

## 📖 Key Resources

- **[Progress Tracker](./PROGRESS.md)** - Track your learning journey
- **[Complete Overview](./COMPLETE-CURRICULUM.md)** - Full curriculum details
- **[How to Start](./HOW-TO-START.md)** - Detailed starting guide
- **[Welcome Guide](./WELCOME.md)** - Setup confirmation & overview
- **[Cheatsheet](./resources/cheatsheet.md)** - CUDA quick reference
- **[Free Resources](./resources/FREE-RESOURCES.md)** - Additional learning materials
- **[Week 1 Guide](./01-basics/README.md)** - Fundamentals overview
- **[Week 2 Guide](./02-memory/README.md)** - Memory optimization
- **[Week 3 Guide](./03-optimization/README.md)** - Advanced optimization
- **[Week 4 Guide](./04-advanced/README.md)** - Advanced features
- **[Week 5 Guide](./05-projects/README.md)** - Projects overview
- **[Week 6 Guide](./06-production/README.md)** - Production guide

## 🏆 Your Achievement

By completing this curriculum, you'll:
- ✅ Master GPU architecture and programming
- ✅ Build production-ready GPU applications
- ✅ Understand how AI frameworks work internally
- ✅ Join the elite group of GPU programmers

## 💡 Pro Tips

1. **Start Simple**: Don't skip the basics
2. **Measure Everything**: Always profile before optimizing
3. **Think Parallel**: Redesign algorithms for GPU
4. **Hardware First**: Understand the hardware limits
5. **Practice Daily**: Consistency is key

---

**Ready to accelerate your code by 10-1000x?** Start with [Lesson 1](./01-basics/lesson01-first-kernel.cu)! 🚀