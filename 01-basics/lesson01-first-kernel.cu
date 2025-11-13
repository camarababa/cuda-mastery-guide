/**
 * Lesson 1: Building Your First GPU Program From Scratch
 * ======================================================
 *
 * The Problem We're Solving:
 * -------------------------
 * Imagine you need to process 1 million numbers. On a CPU with 8 cores,
 * you can process 8 at a time. On a GPU with 2048 cores (like your RTX 2050),
 * you can process 2048 at a time. That's 256x more parallel work!
 *
 * What We'll Build:
 * ----------------
 * 1. First, understand WHY GPUs exist (the parallel computing problem)
 * 2. Write our first GPU function (kernel) from absolute scratch
 * 3. Launch parallel threads and see them work
 * 4. Measure and understand what's happening at the hardware level
 * 5. Build intuition for GPU vs CPU thinking
 *
 * Compile: nvcc -o lesson01 lesson01-first-kernel.cu
 * Run: ./lesson01
 */

#include <stdio.h>
#include <time.h>

/**
 * UNDERSTANDING THE HARDWARE:
 * --------------------------
 * Your RTX 2050 has:
 * - 2048 CUDA cores (think: 2048 tiny calculators)
 * - These cores are grouped into Streaming Multiprocessors (SMs)
 * - Each SM can run multiple threads simultaneously
 * 
 * CPU vs GPU Philosophy:
 * - CPU: Few powerful cores (4-16), optimized for sequential tasks
 * - GPU: Many simple cores (1000s), optimized for parallel tasks
 */

/**
 * YOUR FIRST GPU FUNCTION (KERNEL)
 * --------------------------------
 * This function will run on the GPU, not the CPU.
 * 
 * __global__ is CUDA's way of saying:
 * "This function runs on GPU but can be called from CPU"
 * 
 * Key insight: This SAME function will run on MULTIPLE cores simultaneously!
 * Each copy gets a different thread ID.
 */
__global__ void printThreadID() {
    // Built-in variable: threadIdx.x
    // This is how each thread knows "who" it is
    // Thread 0 gets threadIdx.x = 0
    // Thread 1 gets threadIdx.x = 1, etc.
    int tid = threadIdx.x;
    
    // Each thread prints its own ID
    // Notice: They might print in ANY order (true parallelism!)
    printf("Thread %d: I'm running on a GPU core!\n", tid);
}

/**
 * Let's build a more interesting kernel that does actual work
 */
__global__ void doMathWork() {
    int tid = threadIdx.x;
    
    // Each thread computes something different
    int result = tid * tid;  // Square the thread ID
    
    // Show the computation
    printf("Thread %d: %d squared = %d\n", tid, tid, result);
}

/**
 * A kernel that demonstrates threads are truly independent
 */
__global__ void demonstrateIndependence() {
    int tid = threadIdx.x;
    
    // Simulate different amounts of work per thread
    // Some threads do more work than others
    int work_amount = (tid % 3) + 1;  // 1, 2, or 3 iterations
    
    for (int i = 0; i < work_amount; i++) {
        printf("Thread %d: Doing work iteration %d/%d\n", 
               tid, i+1, work_amount);
    }
}

/**
 * Helper function to visualize what's happening
 */
void visualizeThreadExecution(int numThreads) {
    printf("\nVisualizing %d threads:\n", numThreads);
    printf("CPU would do:  ");
    for (int i = 0; i < numThreads && i < 8; i++) {
        printf("[%d]", i);
        if (i < numThreads - 1) printf("->");
    }
    if (numThreads > 8) printf("->...->%d (sequential)", numThreads-1);
    printf("\n");
    
    printf("GPU does:      ");
    for (int i = 0; i < numThreads && i < 16; i++) {
        printf("[%d]", i);
    }
    if (numThreads > 16) printf("...[%d]", numThreads-1);
    printf(" (ALL AT ONCE!)\n");
}

int main() {
    printf("===============================================\n");
    printf("LESSON 1: Building Your First GPU Program\n");
    printf("===============================================\n\n");
    
    // Part 1: Understanding the Problem
    printf("PART 1: Why Do We Need GPUs?\n");
    printf("----------------------------\n");
    printf("Imagine processing 1 million pixels in an image:\n");
    printf("- CPU (8 cores):    125,000 pixels per core (sequential)\n");
    printf("- GPU (2048 cores): ~488 pixels per core (parallel)\n");
    printf("- Result: GPU can be 100x+ faster for parallel tasks!\n\n");
    
    // Part 2: Your First Kernel Launch
    printf("PART 2: Launching Our First GPU Function\n");
    printf("----------------------------------------\n");
    printf("Let's start with just 10 threads to see what happens...\n\n");
    
    visualizeThreadExecution(10);
    
    printf("\nLaunching kernel with 10 threads...\n");
    printf("Syntax: kernelName<<<blocks, threadsPerBlock>>>()\n\n");
    
    // KERNEL LAUNCH #1: Basic thread ID printing
    printThreadID<<<1, 10>>>();
    
    // Critical: Wait for GPU to finish
    // This is like await in JavaScript or .join() in threading
    cudaDeviceSynchronize();
    
    printf("\nNOTICE: The threads might print OUT OF ORDER!\n");
    printf("This proves they're running in PARALLEL, not sequential.\n\n");
    
    // Part 3: Understanding What Happened
    printf("PART 3: What Just Happened?\n");
    printf("---------------------------\n");
    printf("1. CPU sent the kernel to GPU\n");
    printf("2. GPU created 10 threads\n");
    printf("3. Each thread ran the SAME function\n");
    printf("4. But each got a DIFFERENT threadIdx.x value\n");
    printf("5. All threads ran SIMULTANEOUSLY\n\n");
    
    // Part 4: Let's Do Actual Work
    printf("PART 4: Threads Doing Real Work\n");
    printf("-------------------------------\n");
    printf("Now let's have each thread compute something...\n\n");
    
    doMathWork<<<1, 16>>>();
    cudaDeviceSynchronize();
    
    printf("\nEach thread computed its square independently!\n\n");
    
    // Part 5: Proving Thread Independence
    printf("PART 5: Threads Are Truly Independent\n");
    printf("-------------------------------------\n");
    printf("Let's prove threads don't wait for each other...\n\n");
    
    demonstrateIndependence<<<1, 8>>>();
    cudaDeviceSynchronize();
    
    printf("\nNotice: Threads doing different amounts of work\n");
    printf("finished in DIFFERENT orders!\n\n");
    
    // Part 6: Scaling Up
    printf("PART 6: The Power of Parallelism\n");
    printf("--------------------------------\n");
    printf("Let's scale up to more threads...\n\n");
    
    int threadCounts[] = {32, 64, 128, 256};
    for (int i = 0; i < 4; i++) {
        printf("\nLaunching %d threads:\n", threadCounts[i]);
        visualizeThreadExecution(threadCounts[i]);
        
        // We'll use a simpler kernel for large thread counts
        printThreadID<<<1, threadCounts[i]>>>();
        cudaDeviceSynchronize();
        
        printf("...output truncated for brevity...\n");
    }
    
    // Part 7: Understanding the Hardware
    printf("\n\nPART 7: Your GPU Hardware\n");
    printf("-------------------------\n");
    
    // Query device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("Your GPU: %s\n", prop.name);
    printf("- CUDA Cores: ~2048\n");
    printf("- Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("- Warp size: %d (threads that execute in lockstep)\n", prop.warpSize);
    printf("- Compute capability: %d.%d\n", prop.major, prop.minor);
    
    // Part 8: Key Insights
    printf("\n\nKEY INSIGHTS FROM THIS LESSON\n");
    printf("=============================\n");
    printf("1. GPUs solve PARALLEL problems (same operation, different data)\n");
    printf("2. A kernel is just a function that runs on GPU\n");
    printf("3. Each thread gets a unique ID (threadIdx.x)\n");
    printf("4. Threads run SIMULTANEOUSLY, not sequentially\n");
    printf("5. This is why GPUs can be 100x+ faster than CPUs\n\n");
    
    return 0;
}

/**
 * COMPREHENSIVE EXERCISES
 * ======================
 * 
 * UNDERSTANDING EXERCISES:
 * 1. Run the program multiple times. Do threads always print in the same order?
 *    What does this tell you about parallel execution?
 * 
 * 2. What happens if you launch 10,000 threads? Does it still work?
 *    (Hint: Try it! The GPU will handle it.)
 * 
 * CODING EXERCISES:
 * 3. Create a kernel that prints both threadIdx.x and threadIdx.x * 2
 * 
 * 4. Create a kernel where even threads print "Even: X" and odd print "Odd: X"
 * 
 * 5. Create a kernel that simulates dice rolls:
 *    Each thread computes: (threadIdx.x % 6) + 1
 * 
 * EXPERIMENT EXERCISES:
 * 6. Remove cudaDeviceSynchronize(). What happens? Why?
 * 
 * 7. Launch two different kernels back-to-back. Do they wait for each other?
 * 
 * 8. Time how long it takes to launch 1 thread vs 1000 threads.
 *    Is there a significant difference? Why or why not?
 * 
 * THINKING EXERCISES:
 * 9. If you have 2048 cores and launch 4096 threads, what happens?
 *    How does the GPU handle more threads than cores?
 * 
 * 10. Why might GPUs be WORSE than CPUs for some tasks?
 *     (Hint: Think about sequential dependencies)
 * 
 * BUILD THIS:
 * 11. Create a kernel that simulates a "race condition":
 *     Have all threads try to increment a shared counter.
 *     What happens? (We'll fix this in later lessons)
 * 
 * 12. Build a "thread visualizer" kernel that prints:
 *     "Thread X starting... working... done!"
 *     With some computation in between.
 * 
 * DEEPER UNDERSTANDING:
 * - The <<<1, N>>> syntax means: 1 block with N threads
 * - We'll learn about multiple blocks in Lesson 2
 * - Each thread runs the ENTIRE kernel function
 * - Threads can execute in ANY order
 * - GPU scheduling is NOT deterministic
 * 
 * MENTAL MODEL:
 * Think of the GPU as a massive factory with 2048 workers.
 * A kernel is instructions for what each worker should do.
 * threadIdx.x is each worker's badge number.
 * All workers start at the same time and work independently.
 */
