# Week 1: CUDA Fundamentals (Code-First Approach)

## Philosophy: Learn by Building

Each lesson is a **complete, runnable program** that teaches one concept by implementing it from scratch. No theory without code. No code without understanding.

## Learning Method

1. **Read the code** - Every line is documented
2. **Run the program** - See it work
3. **Modify it** - Break it, fix it, experiment
4. **Build on it** - Each lesson extends the previous

This is how you truly learn: by doing, by experimenting, by building.

---

## Lessons Overview

### Lesson 1: Your First Kernel ★ START HERE
**File:** `lesson01-first-kernel.cu`
**Concept:** What is a kernel and how to launch it

```bash
nvcc -o lesson01 lesson01-first-kernel.cu
./lesson01
```

**What you'll build:**
- Your first GPU function
- Launch threads in parallel
- Understand thread IDs

**Key insight:** Threads execute the same code simultaneously

---

### Lesson 2: Blocks and Grids
**File:** `lesson02-thread-blocks.cu`
**Concept:** Understanding the thread hierarchy

```bash
nvcc -o lesson02 lesson02-thread-blocks.cu
./lesson02
```

**What you'll build:**
- Multi-block launches
- Global thread ID calculation
- Thread organization

**Key insight:** The formula that makes everything work:
```cuda
globalID = blockIdx.x * blockDim.x + threadIdx.x
```

---

### Lesson 3: Array Operations
**File:** `lesson03-array-operation.cu`
**Concept:** Mapping threads to data

```bash
nvcc -o lesson03 lesson03-array-operation.cu
./lesson03
```

**What you'll build:**
- Parallel array processing
- Bounds checking
- Thread-to-element mapping

**Key insight:** Each thread processes one array element

---

### Lesson 4: Memory Models
**File:** `lesson04-memory-model.cu`
**Concept:** CPU ↔ GPU memory management

```bash
nvcc -o lesson04 lesson04-memory-model.cu
./lesson04
```

**What you'll build:**
- Explicit memory (cudaMalloc/cudaMemcpy)
- Unified memory (cudaMallocManaged)
- Side-by-side comparison

**Key insight:** Memory management determines performance

---

### Lesson 5: Vector Addition From Scratch
**File:** `lesson05-vector-add-from-scratch.cu`
**Concept:** Complete algorithm with benchmarking

```bash
nvcc -o lesson05 lesson05-vector-add-from-scratch.cu
./lesson05
```

**What you'll build:**
- CPU baseline implementation
- GPU parallel implementation
- Performance comparison
- Verification system

**Key insight:** Seeing the speedup makes everything click

---

## Your First Day

### Hour 1: Get Your Feet Wet
```bash
cd ~/cuda-learning/01-basics

# Compile and run lesson 1
nvcc -o lesson01 lesson01-first-kernel.cu
./lesson01

# Read the source code - understand every line
nvim lesson01-first-kernel.cu

# Modify: Change 10 to 256 threads
# Recompile and run
```

### Hour 2: Understand Thread Organization
```bash
# Run lesson 2
nvcc -o lesson02 lesson02-thread-blocks.cu
./lesson02

# Study the global ID formula
# Modify: Try 10 blocks × 10 threads
# Calculate: What's the last global ID?
```

### Hour 3: Do Real Work
```bash
# Run lesson 3
nvcc -o lesson03 lesson03-array-operation.cu
./lesson03

# Modify: Change multiply by 2 to multiply by 3
# Modify: Change N to 1000
# Verify it still works
```

---

## How to Approach Each Lesson

### 1. Run First
Don't read the code yet. Just compile and run:
```bash
nvcc -o lessonXX lessonXX-name.cu
./lessonXX
```

See what it does. Read the output carefully.

### 2. Read the Code
Open in your editor and read **every comment**:
```bash
nvim lessonXX-name.cu
```

The code teaches itself. Comments explain:
- What each line does
- Why it's needed
- What would happen without it

### 3. Modify and Experiment
Do the exercises at the bottom of each file. Examples:
- Change array sizes
- Try different thread counts
- Implement variations
- **Break things intentionally** and fix them

### 4. Understand the Key Insight
Each lesson has a "KEY INSIGHT" section. This is the one thing you must understand before moving on.

---

## Compilation Quick Reference

### Basic Compilation
```bash
nvcc -o program file.cu
./program
```

### With Optimization (after lesson 5)
```bash
nvcc -O3 -arch=sm_86 -o program file.cu
```

### With Debug Info
```bash
nvcc -g -G -o program file.cu
cuda-gdb ./program
```

### Check for Errors
```bash
CUDA_LAUNCH_BLOCKING=1 ./program
cuda-memcheck ./program
```

---

## Week 1 Roadmap

### Day 1-2: Foundations
- ✓ Lesson 1: First kernel
- ✓ Lesson 2: Thread organization
- Exercises: Modify thread counts, observe output

### Day 3-4: Real Computing
- ✓ Lesson 3: Array operations
- ✓ Lesson 4: Memory models
- Exercises: Implement variations

### Day 5-6: Complete Algorithm
- ✓ Lesson 5: Vector addition
- Exercises: Benchmark, optimize, experiment

### Day 7: Build Something
Choose one:
- **Project A:** Vector dot product (Σ(aᵢ×bᵢ))
- **Project B:** Element-wise operations (add, sub, mul, div)
- **Project C:** Array statistics (min, max, sum)

---

## Exercises Philosophy

Each lesson has exercises. They're not optional. They're how you learn.

### Easy Exercises
- Modify constants
- Change operations
- Try different sizes

### Medium Exercises
- Implement variations
- Add features
- Compare approaches

### Hard Exercises
- Build new kernels
- Combine concepts
- Optimize performance

### Challenge Exercises
- These will stretch you
- You might need to research
- That's the point

---

## Common Issues & Solutions

### Issue: Compilation error
```
error: identifier "blockIdx" is undefined
```
**Solution:** Did you use `__global__` keyword?

### Issue: No output from kernel
```
// Kernel launched but nothing prints
```
**Solution:** Add `cudaDeviceSynchronize()` after kernel launch

### Issue: Wrong results
```
Expected: [2, 4, 6]
Got: [0, 0, 0]
```
**Solution:** Check bounds condition: `if (idx < n)`

### Issue: Segmentation fault
```
Segmentation fault (core dumped)
```
**Solution:** Check array indices. Are you accessing beyond bounds?

---

## Success Metrics

You're ready for Week 2 when you can:

- [ ] Write a kernel from scratch
- [ ] Calculate grid dimensions for any problem size
- [ ] Explain the global ID formula to someone else
- [ ] Choose between unified and explicit memory
- [ ] Debug basic CUDA errors
- [ ] Measure GPU vs CPU performance

---

## Next Steps After Week 1

Once you complete these 5 lessons and the Week 1 project:

### Week 2: Optimization
- Shared memory
- Bank conflicts
- Coalescing
- Tiling

### Week 3: Algorithms
- Reduction (sum, min, max)
- Scan (prefix sum)
- Matrix multiplication

### Week 4: Real Applications
- Image processing
- Signal processing
- Scientific computing

---

## Tips for Success

### 1. Type, Don't Copy-Paste
```bash
# Don't do this:
cp lesson01.cu my_version.cu

# Do this:
nvim my_version.cu
# Then type it yourself
```
Typing builds muscle memory and understanding.

### 2. Experiment Constantly
Change **one thing** at a time:
- Thread count
- Block count
- Array size
- Operation

See what breaks. Understand why.

### 3. Use Printf Debugging
```cuda
__global__ void myKernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Thread %d: blockIdx=%d threadIdx=%d\n",
           idx, blockIdx.x, threadIdx.x);
}
```

This is your best friend.

### 4. Build a Mental Model
After each lesson, close your eyes and visualize:
- How many threads are running?
- What is each thread doing?
- Where is the data?
- What happens in what order?

### 5. Teach Someone Else
Can't find anyone? Explain it to your rubber duck. Seriously.

If you can't explain it simply, you don't understand it yet.

---

## Your GPU: RTX 2050

```
CUDA Cores:    2,048
Memory:        4 GB GDDR6
Bandwidth:     ~112 GB/s
Compute:       8.6 (Ampere)
Max Threads:   1,024 per block
```

This is a **real GPU**. Not a toy. You can:
- Train small neural networks
- Process images in real-time
- Run scientific simulations
- Learn everything in this curriculum

Your RTX 2050 is more powerful than the supercomputers that NASA used to land on the moon. Use it wisely.

---

## Remember

> "If it disagrees with experiment, it's wrong. That's all there is to it."
> - Richard Feynman

In CUDA terms:
- If your kernel doesn't give correct results, fix it
- If your speedup is poor, understand why
- If you don't understand something, experiment
- **Always verify. Always measure. Always learn.**

---

```
Ready? Your GPU is waiting.

Start here: lesson01-first-kernel.cu
```

**Now go build something.**
