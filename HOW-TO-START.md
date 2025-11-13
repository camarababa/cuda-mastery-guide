# How to Start Learning CUDA

## Your Learning System

You have a **complete, hands-on CUDA course** based on cutting-edge research and the latest documentation. Everything is designed for learning by building.

---

## The Approach: Code First

Each lesson is a **complete program** that:
1. Runs immediately
2. Teaches one concept
3. Includes exercises
4. Builds on previous lessons

No theory without code. No code without understanding.

---

## Your First 30 Minutes (Right Now)

### Step 1: Navigate to basics (1 min)
```bash
cd ~/cuda-learning/01-basics
```

### Step 2: Read the philosophy (3 min)
```bash
cat README.md | less
```

### Step 3: Run your first program (1 min)
```bash
nvcc -o lesson01 lesson01-first-kernel.cu
./lesson01
```

You just ran code on your GPU! 🎉

### Step 4: Understand the code (10 min)
```bash
nvim lesson01-first-kernel.cu
```

Read **every comment**. The code explains itself.

### Step 5: Modify and experiment (15 min)
Change line ~18:
```cuda
printThreadID<<<1, 10>>>();  // Change 10 to 256
```

Recompile and run:
```bash
nvcc -o lesson01 lesson01-first-kernel.cu
./lesson01
```

What changed? Why?

---

## Learning Path

```
Week 1: Fundamentals
├── Lesson 1: First Kernel          [30 min] ← START HERE
├── Lesson 2: Thread Blocks         [45 min]
├── Lesson 3: Array Operations      [60 min]
├── Lesson 4: Memory Models         [60 min]
├── Lesson 5: Vector Addition       [90 min]
└── Week 1 Project                  [2-3 hours]
```

**Total:** ~8-10 hours for Week 1

---

## Files You Have

### Start Here
- **`01-basics/README.md`** - Your Week 1 guide
- **`01-basics/lesson01-first-kernel.cu`** - Begin here

### Lessons (in order)
1. `lesson01-first-kernel.cu` - Your first GPU code
2. `lesson02-thread-blocks.cu` - Understanding organization
3. `lesson03-array-operation.cu` - Real parallel work
4. `lesson04-memory-model.cu` - CPU ↔ GPU memory
5. `lesson05-vector-add-from-scratch.cu` - Complete algorithm

### Reference
- `resources/cheatsheet.md` - Quick syntax lookup
- `resources/parallel-algorithms-guide.md` - Patterns & algorithms
- `COMPREHENSIVE-CURRICULUM.md` - Full 12-week course
- `PROGRESS.md` - Track your learning

---

## Daily Routine (2 hours)

### Morning (30 min)
```bash
cd ~/cuda-learning/01-basics
cat lessonXX-*.cu | less  # Read today's lesson
```

### Afternoon (60 min)
```bash
nvcc -o lessonXX lessonXX-*.cu
./lessonXX                # Run
nvim lessonXX-*.cu        # Modify
./lessonXX                # Test changes
```

### Evening (30 min)
```bash
# Do exercises at bottom of lesson file
# Update PROGRESS.md with what you learned
```

---

## Quick Commands

### Compile and run
```bash
nvcc -o program file.cu && ./program
```

### Check for errors
```bash
cuda-memcheck ./program
```

### Monitor GPU while running
```bash
watch -n 1 nvidia-smi
```

### Quick edit-compile-run loop
```bash
nvim file.cu
nvcc -o test file.cu && ./test
```

---

## Week 1 Goals

By end of Week 1, you will:
- ✓ Write kernels from scratch
- ✓ Launch arbitrary grid sizes
- ✓ Manage GPU memory
- ✓ Measure performance
- ✓ Achieve >2x CPU speedup
- ✓ Debug CUDA programs

---

## Learning Philosophy

### 1. Always Run First
Don't read the code until you see it work.
```bash
nvcc -o lesson lesson.cu && ./lesson
```

### 2. Read Every Comment
The comments are the curriculum.
```cuda
// This isn't just a comment
// It's teaching you how CUDA works
```

### 3. Modify Everything
Change **one thing**, see what breaks:
- Thread counts
- Block sizes
- Array sizes
- Operations

Understanding comes from experimentation.

### 4. Do All Exercises
The exercises aren't optional. They're **how you learn**.

Easy → Medium → Hard → Challenge

Start easy. Push yourself to challenge.

### 5. Build Something
At the end of Week 1, you'll build a project from scratch.
Choose what interests you. Make it work. Make it fast.

---

## When You Get Stuck

### Compilation Error?
```bash
# Read the error message
nvcc -o program file.cu

# Check line number
# Most common: missing semicolon, wrong syntax
```

### Wrong Results?
```cuda
// Add debug prints in kernel
printf("Thread %d: idx=%d value=%f\n", threadIdx.x, idx, value);
```

### Segmentation Fault?
```bash
# Check memory errors
cuda-memcheck ./program

# Common cause: accessing array out of bounds
```

### Still Stuck?
1. Re-read the lesson
2. Check the CUDA error
3. Search the error message
4. Take a break and come back fresh

---

## Success Metrics

### You're making progress when:
- ✓ You can modify code without breaking it
- ✓ You understand error messages
- ✓ You complete exercises
- ✓ You can explain concepts to yourself

### You're ready for Week 2 when:
- ✓ You complete all 5 lessons
- ✓ You finish Week 1 project
- ✓ You can write a kernel from scratch
- ✓ You're excited to learn more

---

## What Makes This Different

### Traditional Courses:
1. Read theory
2. See examples
3. Maybe do exercises
4. Still don't really understand

### This Course:
1. **Run working code**
2. **Understand by reading**
3. **Learn by modifying**
4. **Master by building**

Code is the teacher. Experimentation is the curriculum.

---

## Your GPU is Ready

```
Device: RTX 2050
CUDA Cores: 2,048
Memory: 4 GB
Status: Ready to compute

Your installation: ✓ Complete
Your first program: ✓ Tested
Your learning path: ✓ Clear

The only thing left: Start.
```

---

## Three Commands to Begin

```bash
cd ~/cuda-learning/01-basics
nvcc -o lesson01 lesson01-first-kernel.cu
./lesson01
```

Then open `lesson01-first-kernel.cu` and read every line.

---

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 "The best way to learn is by doing."

 You have:
 ✓ A powerful GPU (RTX 2050)
 ✓ Complete curriculum (12 weeks)
 ✓ Working examples (all tested)
 ✓ Clear path (lesson by lesson)

 Start: lesson01-first-kernel.cu
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Your GPU is waiting. Go make it compute something.**
