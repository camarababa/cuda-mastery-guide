# CUDA Learning Progress

## Week 1: Fundamentals

### Lesson 1: First Kernel ✓ READY
**Status:** ✓ Compiled and tested
**File:** `01-basics/lesson01-first-kernel.cu`

**Completed:**
- [ ] Run the program
- [ ] Understand every line
- [ ] Modify to 256 threads
- [ ] Complete exercises

**Key Concepts to Master:**
- [ ] What is a kernel (`__global__`)
- [ ] How to launch kernels (`<<<blocks, threads>>>`)
- [ ] What are thread IDs (`threadIdx.x`)
- [ ] Why we need `cudaDeviceSynchronize()`

**Notes:**


---

### Lesson 2: Thread Blocks
**Status:** ⏳ Ready to start
**File:** `01-basics/lesson02-thread-blocks.cu`

**Completed:**
- [ ] Run the program
- [ ] Understand global ID formula
- [ ] Calculate thread IDs manually
- [ ] Complete exercises

**Key Concepts to Master:**
- [ ] Block organization
- [ ] Global ID formula: `blockIdx.x * blockDim.x + threadIdx.x`
- [ ] Grid dimensions
- [ ] Why blocks exist

**Notes:**


---

### Lesson 3: Array Operations
**Status:** ⏳ Ready to start
**File:** `01-basics/lesson03-array-operation.cu`

**Completed:**
- [ ] Run the program
- [ ] Understand thread-to-data mapping
- [ ] Understand bounds checking
- [ ] Implement variations

**Key Concepts to Master:**
- [ ] Thread → array element mapping
- [ ] Bounds checking: `if (idx < n)`
- [ ] Why we launch extra threads
- [ ] cudaMallocManaged

**Notes:**


---

### Lesson 4: Memory Models
**Status:** ⏳ Ready to start
**File:** `01-basics/lesson04-memory-model.cu`

**Completed:**
- [ ] Run both examples
- [ ] Understand explicit memory
- [ ] Understand unified memory
- [ ] Know when to use each

**Key Concepts to Master:**
- [ ] cudaMalloc vs cudaMallocManaged
- [ ] cudaMemcpy directions (H2D, D2H, D2D)
- [ ] When to use explicit memory
- [ ] When to use unified memory

**Notes:**


---

### Lesson 5: Vector Addition From Scratch
**Status:** ⏳ Ready to start
**File:** `01-basics/lesson05-vector-add-from-scratch.cu`

**Completed:**
- [ ] Run the benchmark
- [ ] Achieve speedup > 2x
- [ ] Understand performance metrics
- [ ] Complete all exercises

**Key Concepts to Master:**
- [ ] Complete GPU workflow
- [ ] Performance measurement
- [ ] Speedup calculation
- [ ] When GPU is worth it

**Notes:**


---

## Week 1 Project

**Goal:** Build something from scratch using what you learned

**Choose One:**

### Option A: Vector Dot Product
```cuda
// Compute: result = Σ(a[i] * b[i])
// Hint: You'll need reduction (we'll learn this properly in Week 3)
```
**Difficulty:** ⭐⭐⭐

### Option B: Element-wise Operations
```cuda
// Implement: add, subtract, multiply, divide
// With bounds checking and verification
```
**Difficulty:** ⭐⭐

### Option C: Array Statistics
```cuda
// Find: min, max, sum of array
// Compare CPU vs GPU performance
```
**Difficulty:** ⭐⭐⭐

**Project Status:**
- [ ] Chosen project: _______________
- [ ] Implemented
- [ ] Tested
- [ ] Benchmarked

**Project Notes:**


---

## Skills Checklist

### By End of Week 1, I can:

**Understanding:**
- [ ] Explain what a GPU kernel is
- [ ] Draw the thread hierarchy (threads → blocks → grid)
- [ ] Calculate global thread ID from block/thread indices
- [ ] Explain CPU ↔ GPU memory flow

**Coding:**
- [ ] Write a kernel from scratch
- [ ] Calculate grid dimensions for any N
- [ ] Implement bounds checking
- [ ] Choose appropriate memory model

**Debugging:**
- [ ] Use `printf` in kernels
- [ ] Check CUDA errors
- [ ] Use `cuda-memcheck`
- [ ] Fix common kernel issues

**Performance:**
- [ ] Measure kernel execution time
- [ ] Calculate speedup
- [ ] Understand when GPU helps

---

## Learning Log

### Session 1:
**Date:**
**Duration:**
**Lessons:**
**Key Learnings:**


**Challenges:**


**Experiments:**


---

### Session 2:
**Date:**
**Duration:**
**Lessons:**


**Key Learnings:**


**Challenges:**


**Experiments:**


---

### Session 3:
**Date:**
**Duration:**
**Lessons:**


**Key Learnings:**


**Challenges:**


**Experiments:**


---

## Questions & Answers

**Q:** Why do threads not print in order?
**A:**

**Q:** What happens if I forget bounds checking?
**A:**

**Q:** When should I use unified vs explicit memory?
**A:**

**Q:** How do I calculate number of blocks needed?
**A:**

**Q:** What is a good threadsPerBlock value?
**A:**

---

## Performance Benchmarks

### My GPU: RTX 2050
**Specs:**
- CUDA Cores: 2,048
- Memory: 4 GB
- Compute Capability: 8.6

### Lesson 5 Results
**Vector Addition (N=1,000,000):**
- CPU Time: _____ ms
- GPU Time: _____ ms
- Speedup: _____ x
- Bandwidth: _____ GB/s

**With Different Thread Counts:**
| Threads/Block | Time (ms) | Speedup |
|---------------|-----------|---------|
| 64            |           |         |
| 128           |           |         |
| 256           |           |         |
| 512           |           |         |
| 1024          |           |         |

**Insights:**


---

## Next Week Preview

### Week 2: Optimization
Once you master Week 1, you'll learn:
- **Shared Memory:** On-chip cache for speed
- **Memory Coalescing:** Accessing memory efficiently
- **Bank Conflicts:** And how to avoid them
- **Tiled Algorithms:** Breaking problems into blocks

**Why it matters:** Turn 2x speedup into 50x speedup

---

## Resources Used

**Official Docs:**
- [ ] CUDA Programming Guide Ch 1-3
- [ ] CUDA Best Practices Guide

**Course Materials:**
- [✓] Lesson source code
- [ ] Cheatsheet
- [ ] Parallel algorithms guide

**External:**
- [ ] NVIDIA Developer Blog
- [ ] Stack Overflow CUDA tag
- [ ] NVIDIA Forums

---

## Motivation & Goals

**Why I'm learning CUDA:**


**What I want to build:**


**Success criteria:**


---

```
"The only way to learn a new programming language
 is by writing programs in it." - Dennis Ritchie

Applied to CUDA: Compile. Run. Modify. Repeat.
```

**Keep going. Every expert was once a beginner.**
