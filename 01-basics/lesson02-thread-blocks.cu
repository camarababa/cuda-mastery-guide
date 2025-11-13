/**
 * Lesson 2: Building the Thread Hierarchy From First Principles
 * =============================================================
 *
 * The Problem We're Solving:
 * -------------------------
 * In Lesson 1, we launched up to 1024 threads. But what if we need
 * 1 MILLION threads? Or 1 BILLION? We need a hierarchical organization.
 *
 * What We'll Build:
 * ----------------
 * 1. Understand WHY we need blocks (hardware limitations)
 * 2. Build the global ID formula from scratch
 * 3. See how blocks map to actual GPU hardware
 * 4. Learn to calculate grid dimensions for any problem size
 * 5. Visualize the entire thread hierarchy
 *
 * Real-World Application:
 * -----------------------
 * Processing a 4K image (3840×2160 = 8,294,400 pixels)
 * We need 8.3 million threads - far more than 1024!
 *
 * Compile: nvcc -o lesson02 lesson02-thread-blocks.cu
 * Run: ./lesson02
 */

#include <stdio.h>
#include <assert.h>

/**
 * FIRST PRINCIPLES: Why Do We Need Blocks?
 * ----------------------------------------
 * Hardware Reality:
 * 1. GPU cores are grouped into Streaming Multiprocessors (SMs)
 * 2. Each SM can handle a limited number of threads (typically 1024-2048)
 * 3. Threads within a block can cooperate (share memory, synchronize)
 * 4. Blocks are independent and can run on any SM
 *
 * Think of it like this:
 * - GPU = Factory with multiple assembly lines (SMs)
 * - Block = A team of workers on one assembly line
 * - Thread = Individual worker
 * - Grid = All teams working on the entire project
 */

/**
 * Let's build up the concept step by step
 */

// Step 1: Understanding thread limits
__global__ void demonstrateThreadLimit() {
    // This kernel shows why we can't just launch infinite threads
    int tid = threadIdx.x;
    if (tid == 0) {
        printf("Block %d: I can have max %d threads (hardware limit)\n", 
               blockIdx.x, blockDim.x);
    }
}

// Step 2: Visualizing the hierarchy
__global__ void visualizeHierarchy() {
    // Every thread has THREE pieces of identity:
    int myThreadID = threadIdx.x;    // Who am I within my block? (0-255)
    int myBlockID = blockIdx.x;      // Which block am I in? (0-N)
    int threadsInMyBlock = blockDim.x; // How many threads in my block?
    
    // Only thread 0 of each block reports
    if (myThreadID == 0) {
        printf("Block %d reporting: I have %d threads (IDs 0-%d)\n",
               myBlockID, threadsInMyBlock, threadsInMyBlock-1);
    }
}

// Step 3: The fundamental formula - deriving global ID
__global__ void deriveGlobalID() {
    // Let's think about this like apartment addressing:
    // Building (grid) has Floors (blocks)
    // Each Floor has Apartments (threads)
    // Your global address = (Floor × ApartmentsPerFloor) + ApartmentNumber
    
    int localID = threadIdx.x;        // Apartment number on my floor
    int blockID = blockIdx.x;         // My floor number
    int threadsPerBlock = blockDim.x; // Apartments per floor
    
    // THE FORMULA: This is how we number threads globally
    int globalID = blockID * threadsPerBlock + localID;
    
    // Let's show the math
    if (localID < 3 || localID == threadsPerBlock - 1) { // First 3 and last
        printf("Block %d, Thread %d: (%d × %d) + %d = Global ID %d\n",
               blockID, localID, blockID, threadsPerBlock, localID, globalID);
    }
}

// Step 4: Practical application - processing array elements
__global__ void processArray(int *data, int n) {
    // This is THE pattern you'll use 90% of the time
    int globalID = blockIdx.x * blockDim.x + threadIdx.x;
    
    // CRITICAL: Check bounds! We might launch more threads than data
    if (globalID < n) {
        // Each thread processes one element
        data[globalID] = globalID * 2; // Simple operation for demo
        
        // Show what we're doing (only first few)
        if (globalID < 5) {
            printf("Thread (B%d,T%d) processing element %d: %d → %d\n",
                   blockIdx.x, threadIdx.x, globalID, globalID, data[globalID]);
        }
    }
}

// Helper function to visualize grid structure
void visualizeGrid(int blocks, int threadsPerBlock) {
    printf("\nGrid Visualization:\n");
    printf("[");
    for (int b = 0; b < blocks && b < 8; b++) {
        printf("Block%d(T0-T%d)", b, threadsPerBlock-1);
        if (b < blocks-1) printf(" | ");
    }
    if (blocks > 8) printf(" | ... | Block%d", blocks-1);
    printf("]\n");
    printf("Total threads: %d × %d = %d\n", blocks, threadsPerBlock, 
           blocks * threadsPerBlock);
}

// Helper to calculate grid dimensions for any problem size
void calculateGridDim(int n, int threadsPerBlock) {
    // THE CEILING DIVISION FORMULA - memorize this!
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("\nCalculating grid for %d elements with %d threads/block:\n", 
           n, threadsPerBlock);
    printf("  blocks = (%d + %d - 1) / %d = %d / %d = %d\n",
           n, threadsPerBlock, threadsPerBlock, 
           n + threadsPerBlock - 1, threadsPerBlock, blocks);
    printf("  Total threads: %d (we'll waste %d threads)\n", 
           blocks * threadsPerBlock, blocks * threadsPerBlock - n);
}

int main() {
    printf("================================================\n");
    printf("LESSON 2: Building Thread Hierarchy From Scratch\n");
    printf("================================================\n\n");
    
    // PART 1: Understanding Why We Need Blocks
    printf("PART 1: Why Can't We Just Launch a Million Threads?\n");
    printf("--------------------------------------------------\n");
    printf("Hardware Reality Check:\n");
    printf("- Your GPU has ~16-20 Streaming Multiprocessors (SMs)\n");
    printf("- Each SM can handle ~1024-2048 threads MAX\n");
    printf("- Threads must be organized into blocks\n");
    printf("- Each block runs on ONE SM\n\n");
    
    // Demonstrate thread limits
    demonstrateThreadLimit<<<3, 512>>>();
    cudaDeviceSynchronize();
    
    // PART 2: Visualizing the Hierarchy
    printf("\nPART 2: The Thread Hierarchy\n");
    printf("----------------------------\n");
    printf("Let's launch 4 blocks with 8 threads each...\n\n");
    
    visualizeGrid(4, 8);
    visualizeHierarchy<<<4, 8>>>();
    cudaDeviceSynchronize();
    
    // PART 3: Deriving the Global ID Formula
    printf("\nPART 3: The Most Important Formula in CUDA\n");
    printf("------------------------------------------\n");
    printf("How do we give each thread a unique global ID?\n\n");
    
    printf("Think of it like apartment numbering:\n");
    printf("- Building = Grid (all blocks)\n");
    printf("- Floor = Block\n");
    printf("- Apartment = Thread\n");
    printf("- Global address = (Floor × ApartmentsPerFloor) + ApartmentNumber\n\n");
    
    deriveGlobalID<<<3, 4>>>();  // 3 blocks, 4 threads each for clarity
    cudaDeviceSynchronize();
    
    // PART 4: Practical Application
    printf("\nPART 4: Using Blocks to Process Data\n");
    printf("------------------------------------\n");
    
    // Allocate and initialize array
    int n = 50;
    int *d_array;
    cudaMalloc(&d_array, n * sizeof(int));
    cudaMemset(d_array, 0, n * sizeof(int));
    
    // Calculate grid dimensions
    int threadsPerBlock = 16;
    calculateGridDim(n, threadsPerBlock);
    
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    printf("\nLaunching kernel to process %d elements...\n\n", n);
    
    processArray<<<blocks, threadsPerBlock>>>(d_array, n);
    cudaDeviceSynchronize();
    
    // PART 5: Scaling to Real Problems
    printf("\nPART 5: Scaling to Real-World Sizes\n");
    printf("-----------------------------------\n");
    
    // Show calculations for different problem sizes
    int sizes[] = {1000, 1000000, 8294400};  // 1K, 1M, 4K image
    const char* labels[] = {"1K elements", "1M elements", "4K image (8.3M pixels)"};
    
    for (int i = 0; i < 3; i++) {
        printf("\n%s:\n", labels[i]);
        calculateGridDim(sizes[i], 256);
    }
    
    // PART 6: Different Block Sizes
    printf("\nPART 6: Choosing Block Size\n");
    printf("---------------------------\n");
    printf("Common block sizes and their uses:\n");
    printf("- 64:   Minimum for good occupancy\n");
    printf("- 128:  Good for memory-bound kernels\n");
    printf("- 256:  General purpose, good default\n");
    printf("- 512:  Compute-intensive kernels\n");
    printf("- 1024: Maximum, rarely optimal\n\n");
    
    // PART 7: 2D and 3D Grids (Preview)
    printf("PART 7: Beyond 1D (Preview)\n");
    printf("---------------------------\n");
    printf("CUDA supports 2D and 3D grids for natural mapping:\n");
    printf("- Image processing: <<<dim3(width/16, height/16), dim3(16,16)>>>\n");
    printf("- Volume rendering: <<<dim3(x/8, y/8, z/8), dim3(8,8,8)>>>\n");
    printf("We'll cover this in advanced lessons!\n\n");
    
    // PART 8: Key Insights
    printf("KEY INSIGHTS FROM THIS LESSON\n");
    printf("=============================\n");
    printf("1. Blocks exist because of hardware limitations\n");
    printf("2. Global ID = blockIdx.x * blockDim.x + threadIdx.x\n");
    printf("3. Always check bounds when globalID >= problem size\n");
    printf("4. Grid size = (n + blockSize - 1) / blockSize\n");
    printf("5. 256 threads/block is a safe default\n\n");
    
    // Cleanup
    cudaFree(d_array);
    
    return 0;
}

/**
 * COMPREHENSIVE EXERCISES
 * ======================
 * 
 * UNDERSTANDING EXERCISES:
 * 1. Grid Math Practice:
 *    a) 10 blocks × 10 threads. What's the global ID of:
 *       - Last thread in block 3? (Answer: 3×10+9 = 39)
 *       - First thread in block 7? (Answer: 7×10+0 = 70)
 *       - Thread 5 in block 4? (Answer: 4×10+5 = 45)
 *    
 * 2. Reverse Engineering:
 *    Given global ID 157 and 32 threads/block:
 *    - Which block? (Answer: 157/32 = 4)
 *    - Which thread? (Answer: 157%32 = 29)
 *    - Verify: 4×32+29 = 157 ✓
 *
 * CODING EXERCISES:
 * 3. Create a kernel that prints only threads where globalID is:
 *    - Divisible by 3
 *    - Prime numbers
 *    - Perfect squares
 *
 * 4. Create a "block reporter" kernel where only thread 0 of each block
 *    reports the global ID range its block handles.
 *
 * 5. Implement a kernel that colors a "grid visualization":
 *    Even blocks print "█", odd blocks print "░"
 *
 * PRACTICAL EXERCISES:
 * 6. Grid Dimension Calculator:
 *    Write a function that takes array size N and suggests optimal
 *    block size based on:
 *    - N < 1024: use N threads (1 block)
 *    - N < 65536: use 256 threads/block
 *    - N >= 65536: use 512 threads/block
 *
 * 7. Bounds Checking Practice:
 *    Process an array of N=1000 with 256 threads/block.
 *    How many threads are "wasted"? (24)
 *    Show they correctly skip processing.
 *
 * EXPERIMENT EXERCISES:
 * 8. Launch with different configurations for N=10000:
 *    - 100 blocks × 100 threads
 *    - 40 blocks × 250 threads  
 *    - 10000 blocks × 1 thread (!)  
 *    Which is most efficient? Why?
 *
 * 9. Maximum Grid Test:
 *    What's the maximum number of blocks you can launch?
 *    (Hint: It's huge - try 1 million blocks!)
 *
 * BUILD THIS:
 * 10. "Wave Visualization":
 *     Create a kernel where each thread sleeps for
 *     (globalID / 100) milliseconds, then prints.
 *     You'll see "waves" of output!
 *
 * 11. Block Statistics:
 *     Each block calculates and reports:
 *     - Its first and last global ID
 *     - How many elements it will process
 *     - Whether it's the "edge" block (might have unused threads)
 *
 * 12. Grid Configurator Tool:
 *     Build a function that takes:
 *     - Problem size N
 *     - Desired threads per block
 *     And outputs:
 *     - Number of blocks needed
 *     - Total threads launched
 *     - Wasted threads
 *     - Efficiency percentage
 *
 * ADVANCED THINKING:
 * 13. Why can't we just use 1 thread per block for everything?
 *     (Think about: occupancy, memory access, cooperation)
 *
 * 14. When would you choose 64 vs 256 vs 1024 threads/block?
 *     (Hint: It depends on your kernel's requirements)
 *
 * 15. How does the grid map to actual GPU hardware?
 *     Research: What's a "warp" and why is 32 special?
 *
 * REAL-WORLD APPLICATION:
 * 16. Image Processing Grid:
 *     For a 1920×1080 image:
 *     - Design a 2D grid to process it
 *     - Each thread handles one pixel
 *     - Hint: <<<dim3(120,68), dim3(16,16)>>>
 *
 * MENTAL MODEL:
 * Think of the GPU as a massive hotel:
 * - Grid = Entire hotel
 * - Block = One floor  
 * - Thread = One room
 * - Global ID = Room number for the entire hotel
 * - blockIdx = Floor number
 * - threadIdx = Room number on that floor
 * 
 * The formula gives each room a unique number across
 * the entire hotel, no matter which floor it's on!
 */
