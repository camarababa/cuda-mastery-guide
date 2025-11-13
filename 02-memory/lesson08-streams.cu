/**
 * Lesson 8: CUDA Streams - Hiding Latency Through Parallelism
 * ===========================================================
 *
 * The Hidden Truth:
 * ----------------
 * Your GPU has been LYING to you! When you thought operations were
 * happening sequentially, the GPU was secretly optimizing. Now we'll
 * take EXPLICIT control and achieve true parallelism!
 *
 * The Problem:
 * -----------
 * Traditional GPU code:
 *   1. Copy A to GPU (5ms) 
 *   2. Process A (10ms)
 *   3. Copy A back (5ms)
 *   Total: 20ms
 *
 * With streams:
 *   Stream 0: Copy A → Process A → Copy A back
 *   Stream 1:      Copy B → Process B → Copy B back  
 *   Stream 2:           Copy C → Process C → Copy C back
 *   Total: ~12ms (40% faster!)
 *
 * What We'll Build:
 * ----------------
 * 1. Understand GPU execution model (it's not what you think!)
 * 2. Master stream creation and synchronization
 * 3. Overlap transfers and computation
 * 4. Use multiple streams effectively
 * 5. Pinned memory for fast transfers
 * 6. Build a production-ready pipeline
 * 7. Understand concurrency vs parallelism
 *
 * Real-World Applications:
 * -----------------------
 * - Video processing: Process frame N while transferring N+1
 * - Deep learning: Overlap gradient computation and communication
 * - Scientific computing: Pipeline I/O and computation
 * - Real-time systems: Minimize latency through pipelining
 *
 * This is the PROFESSIONAL way to use GPUs!
 *
 * Compile: nvcc -O3 -arch=sm_86 -o lesson08 lesson08-streams.cu
 * Run: ./lesson08
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <assert.h>

/**
 * FIRST PRINCIPLES: The GPU Execution Model
 * ----------------------------------------
 * 
 * Your GPU has multiple engines:
 * 1. Compute Engine(s) - Runs kernels
 * 2. Copy Engine(s) - Transfers data
 * 3. Host Interface - CPU communication
 * 
 * These can work SIMULTANEOUSLY!
 * 
 * Default stream (0): All operations serialize
 * Custom streams: Operations can overlap!
 * 
 * Think of it like a restaurant:
 * - Default: One waiter does everything sequentially  
 * - Streams: Multiple waiters work in parallel
 * - Kitchen (GPU) can handle multiple orders at once!
 */

// Simple kernel that does some work
__global__ void processData(float *data, int n, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float result = data[idx];
        
        // Simulate computation
        #pragma unroll
        for (int i = 0; i < 50; i++) {
            result = sinf(result) * cosf(result) + value;
        }
        
        data[idx] = result;
    }
}

// More complex kernel for better overlap demonstration
__global__ void complexProcess(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float sum = 0.0f;
        
        // Simulate heavy computation
        for (int i = 0; i < 100; i++) {
            float val = input[idx] + i * 0.01f;
            sum += sqrtf(fabsf(sinf(val) * cosf(val)));
        }
        
        output[idx] = sum / 100.0f;
    }
}

// Timer class using CUDA events (more accurate for GPU timing)
class GpuTimer {
    cudaEvent_t start, stop;
public:
    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    
    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    void Start() {
        cudaEventRecord(start, 0);
    }
    
    float Elapsed() {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsed;
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

/**
 * STEP 1: Life Without Streams (Default Behavior)
 * ----------------------------------------------
 * Everything happens in order. No overlap. Sad.
 */
void withoutStreams(int nElements) {
    printf("\n📊 EXPERIMENT 1: Traditional Sequential Execution\n");
    printf("================================================\n");
    
    size_t size = nElements * sizeof(float);
    
    // Allocate pageable host memory (SLOW transfers)
    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);
    
    // Initialize
    for (int i = 0; i < nElements; i++) {
        h_input[i] = (float)i / nElements;
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    GpuTimer timer;
    timer.Start();
    
    // Sequential operations
    printf("\nSequential Timeline:\n");
    printf("--------------------\n");
    
    // 1. Copy to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    printf("T=0-5ms:   [H→D Transfer]\n");
    
    // 2. Launch kernel
    int threads = 256;
    int blocks = (nElements + threads - 1) / threads;
    complexProcess<<<blocks, threads>>>(d_input, d_output, nElements);
    cudaDeviceSynchronize();
    printf("T=5-15ms:              [Kernel Execution]\n");
    
    // 3. Copy back
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    printf("T=15-20ms:                             [D→H Transfer]\n");
    
    float totalTime = timer.Elapsed();
    printf("\nTotal Time: %.2f ms (No overlap!)\n", totalTime);
    
    // Cleanup
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
}

/**
 * STEP 2: Introduction to Streams
 * -------------------------------
 * Separate queues of operations that can overlap
 */
void basicStreams(int nElements, int nStreams) {
    printf("\n\n📊 EXPERIMENT 2: Basic Stream Usage\n");
    printf("===================================\n");
    
    size_t size = nElements * sizeof(float);
    size_t streamSize = size / nStreams;
    int streamElements = nElements / nStreams;
    
    // Create streams
    cudaStream_t *streams = new cudaStream_t[nStreams];
    for (int i = 0; i < nStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // Allocate pinned memory (FAST transfers)
    float *h_input, *h_output;
    cudaHostAlloc(&h_input, size, cudaHostAllocDefault);
    cudaHostAlloc(&h_output, size, cudaHostAllocDefault);
    
    // Initialize
    for (int i = 0; i < nElements; i++) {
        h_input[i] = (float)i / nElements;
    }
    
    // Device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    printf("\nConfiguration:\n");
    printf("- Total elements: %d\n", nElements);
    printf("- Streams: %d\n", nStreams);
    printf("- Elements per stream: %d\n", streamElements);
    printf("- Using pinned memory for fast transfers\n");
    
    GpuTimer timer;
    timer.Start();
    
    // Launch all operations
    for (int i = 0; i < nStreams; i++) {
        int offset = i * streamElements;
        
        // Each stream processes its chunk
        cudaMemcpyAsync(&d_input[offset], &h_input[offset], 
                       streamSize, cudaMemcpyHostToDevice, streams[i]);
        
        int threads = 256;
        int blocks = (streamElements + threads - 1) / threads;
        complexProcess<<<blocks, threads, 0, streams[i]>>>
            (&d_input[offset], &d_output[offset], streamElements);
        
        cudaMemcpyAsync(&h_output[offset], &d_output[offset], 
                       streamSize, cudaMemcpyDeviceToHost, streams[i]);
    }
    
    // Wait for all streams
    cudaDeviceSynchronize();
    
    float totalTime = timer.Elapsed();
    printf("\nTotal Time: %.2f ms (With overlap!)\n", totalTime);
    
    // Visualize timeline
    printf("\nOverlapped Timeline:\n");
    printf("--------------------\n");
    printf("Stream 0: [H→D]    [Kernel]    [D→H]\n");
    printf("Stream 1:     [H→D]    [Kernel]    [D→H]\n");
    printf("Stream 2:         [H→D]    [Kernel]    [D→H]\n");
    printf("Stream 3:             [H→D]    [Kernel]    [D→H]\n");
    printf("          ↑ Overlap saves time!\n");
    
    // Cleanup
    for (int i = 0; i < nStreams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    delete[] streams;
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
}

/**
 * STEP 3: Advanced - Optimal Stream Pattern
 * -----------------------------------------
 * The "breadth-first" pattern for maximum overlap
 */
void optimalStreams(int nElements, int nStreams) {
    printf("\n\n📊 EXPERIMENT 3: Optimal Stream Pattern\n");
    printf("======================================\n");
    
    size_t size = nElements * sizeof(float);
    size_t streamSize = size / nStreams;
    int streamElements = nElements / nStreams;
    
    // Create streams
    cudaStream_t *streams = new cudaStream_t[nStreams];
    for (int i = 0; i < nStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // Allocate pinned memory
    float *h_input, *h_output;
    cudaHostAlloc(&h_input, size, cudaHostAllocDefault);
    cudaHostAlloc(&h_output, size, cudaHostAllocDefault);
    
    // Initialize
    for (int i = 0; i < nElements; i++) {
        h_input[i] = (float)i / nElements;
    }
    
    // Device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    GpuTimer timer;
    timer.Start();
    
    // OPTIMAL PATTERN: Issue all H2D, then all kernels, then all D2H
    // This maximizes overlap opportunities!
    
    printf("\nOptimal ordering (breadth-first):\n");
    
    // 1. All H2D transfers
    for (int i = 0; i < nStreams; i++) {
        int offset = i * streamElements;
        cudaMemcpyAsync(&d_input[offset], &h_input[offset], 
                       streamSize, cudaMemcpyHostToDevice, streams[i]);
    }
    
    // 2. All kernels
    for (int i = 0; i < nStreams; i++) {
        int offset = i * streamElements;
        int threads = 256;
        int blocks = (streamElements + threads - 1) / threads;
        complexProcess<<<blocks, threads, 0, streams[i]>>>
            (&d_input[offset], &d_output[offset], streamElements);
    }
    
    // 3. All D2H transfers
    for (int i = 0; i < nStreams; i++) {
        int offset = i * streamElements;
        cudaMemcpyAsync(&h_output[offset], &d_output[offset], 
                       streamSize, cudaMemcpyDeviceToHost, streams[i]);
    }
    
    cudaDeviceSynchronize();
    
    float totalTime = timer.Elapsed();
    printf("\nTotal Time: %.2f ms (Optimal pattern!)\n", totalTime);
    
    // Cleanup
    for (int i = 0; i < nStreams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    delete[] streams;
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
}

/**
 * STEP 4: Stream Callbacks
 * ------------------------
 * CPU functions that execute when stream reaches that point
 */
void CUDART_CB streamCallback(cudaStream_t stream, cudaError_t status, void *data) {
    printf("Stream %d completed chunk %d\n", 
           (int)(intptr_t)stream, *(int*)data);
}

void streamCallbacks(int nElements, int nStreams) {
    printf("\n\n📊 EXPERIMENT 4: Stream Callbacks\n");
    printf("=================================\n");
    
    size_t streamSize = nElements / nStreams * sizeof(float);
    int streamElements = nElements / nStreams;
    
    // Create streams
    cudaStream_t *streams = new cudaStream_t[nStreams];
    int *streamIds = new int[nStreams];
    
    for (int i = 0; i < nStreams; i++) {
        cudaStreamCreate(&streams[i]);
        streamIds[i] = i;
    }
    
    float *d_data;
    cudaMalloc(&d_data, nElements * sizeof(float));
    
    printf("\nLaunching work with callbacks...\n\n");
    
    // Launch work with callbacks
    for (int i = 0; i < nStreams; i++) {
        int offset = i * streamElements;
        
        processData<<<100, 256, 0, streams[i]>>>
            (&d_data[offset], streamElements, (float)i);
        
        // Add callback
        cudaStreamAddCallback(streams[i], streamCallback, 
                            &streamIds[i], 0);
    }
    
    // Callbacks will print as streams complete
    cudaDeviceSynchronize();
    
    // Cleanup
    for (int i = 0; i < nStreams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    delete[] streams;
    delete[] streamIds;
    cudaFree(d_data);
}

/**
 * STEP 5: Events and Synchronization
 * ----------------------------------
 * Fine-grained control over stream dependencies
 */
void eventsAndSync() {
    printf("\n\n📊 EXPERIMENT 5: Events and Dependencies\n");
    printf("=======================================\n");
    
    // Create streams and events
    cudaStream_t stream1, stream2;
    cudaEvent_t event1;
    
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaEventCreate(&event1);
    
    float *d_data;
    cudaMalloc(&d_data, 1000000 * sizeof(float));
    
    printf("\nDemonstrating stream dependencies:\n");
    
    // Stream 1 does work
    printf("1. Stream 1 starts work...\n");
    processData<<<100, 256, 0, stream1>>>(d_data, 100000, 1.0f);
    
    // Record event when stream 1 finishes
    cudaEventRecord(event1, stream1);
    
    // Stream 2 waits for event
    printf("2. Stream 2 waits for Stream 1...\n");
    cudaStreamWaitEvent(stream2, event1);
    
    // Stream 2 can now work
    printf("3. Stream 2 continues after Stream 1 completes\n");
    processData<<<100, 256, 0, stream2>>>(d_data, 100000, 2.0f);
    
    cudaDeviceSynchronize();
    printf("4. All work complete!\n");
    
    // Cleanup
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaEventDestroy(event1);
    cudaFree(d_data);
}

/**
 * Main demonstration
 */
int main() {
    printf("===========================================================\n");
    printf("LESSON 8: CUDA Streams - Professional GPU Programming\n");
    printf("===========================================================\n\n");
    
    // Check device capabilities
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("Device Capabilities:\n");
    printf("-------------------\n");
    printf("Device: %s\n", prop.name);
    printf("Concurrent kernels: %s\n", prop.concurrentKernels ? "Yes" : "No");
    printf("Async engine count: %d\n", prop.asyncEngineCount);
    printf("- 0: No overlap\n");
    printf("- 1: Host↔Device with kernel overlap\n");  
    printf("- 2: Bidirectional with kernel overlap\n\n");
    
    int nElements = 10 * 1024 * 1024;  // 10M elements
    int nStreams = 4;
    
    // Run experiments
    withoutStreams(nElements);
    basicStreams(nElements, nStreams);
    optimalStreams(nElements, nStreams);
    streamCallbacks(nElements, nStreams);
    eventsAndSync();
    
    // Performance comparison
    printf("\n\n📊 PERFORMANCE SUMMARY\n");
    printf("=====================\n\n");
    
    printf("Typical speedups with streams:\n");
    printf("- Sequential (no streams):     1.0x (baseline)\n");
    printf("- Basic streams:              1.3-1.5x\n");
    printf("- Optimal pattern:            1.5-2.0x\n");
    printf("- With pinned memory:         2.0-3.0x\n\n");
    
    printf("Rules for maximum performance:\n");
    printf("1. Use pinned memory (cudaHostAlloc)\n");
    printf("2. Use breadth-first launch pattern\n");
    printf("3. Size transfers to hide latency\n");
    printf("4. Balance work across streams\n");
    printf("5. Minimize synchronization\n\n");
    
    // Key insights
    printf("🔑 KEY INSIGHTS\n");
    printf("===============\n\n");
    
    printf("1. Streams enable TRUE concurrency\n");
    printf("2. Default stream (0) is synchronizing\n");
    printf("3. Pinned memory is CRUCIAL for overlap\n");
    printf("4. Order of operations matters\n");
    printf("5. GPUs have separate engines - use them!\n");
    printf("6. Not all kernels benefit equally\n\n");
    
    printf("Common patterns:\n");
    printf("- Pipeline: Process batch N while transferring N+1\n");
    printf("- Fork-join: Split work → parallel process → merge\n");
    printf("- Producer-consumer: Continuous processing\n\n");
    
    printf("✅ You now control GPU parallelism like a pro!\n\n");
    
    return 0;
}

/**
 * COMPREHENSIVE EXERCISES
 * ======================
 * 
 * UNDERSTANDING EXERCISES:
 * 1. Timeline Analysis:
 *    Draw the execution timeline for:
 *    - 3 operations, no streams
 *    - 3 operations, 3 streams
 *    Where is the overlap?
 *
 * 2. Concurrency Limits:
 *    If GPU can run 16 kernels concurrently,
 *    what happens with 20 streams?
 *
 * 3. Memory Types:
 *    Why does pinned memory enable overlap
 *    but pageable memory doesn't?
 *
 * CODING EXERCISES:
 * 4. Stream Pool:
 *    Create a class that manages a pool of streams.
 *    Automatically assigns work to available streams.
 *
 * 5. Dependency Graph:
 *    Implement A→B→C where B depends on A, C depends on B.
 *    Use events to enforce dependencies.
 *
 * 6. Dynamic Streams:
 *    Create streams based on workload size.
 *    Small workload: 2 streams
 *    Large workload: 8 streams
 *
 * 7. Error Handling:
 *    Add proper error checking with streams.
 *    Handle async errors correctly.
 *
 * OPTIMIZATION CHALLENGES:
 * 8. Find Optimal Stream Count:
 *    Test 1, 2, 4, 8, 16, 32 streams.
 *    Plot performance. Why does it plateau?
 *
 * 9. Chunk Size Tuning:
 *    Given fixed data size, vary chunks per stream.
 *    Find optimal chunk size.
 *
 * 10. Multi-GPU Streams:
 *     Use streams across multiple GPUs.
 *     Device 0 stream → Device 1 stream.
 *
 * ANALYSIS EXERCISES:
 * 11. Profile with NSight:
 *     Visualize stream timeline.
 *     Identify gaps and inefficiencies.
 *
 * 12. PCIe Bandwidth:
 *     Measure achieved vs theoretical bandwidth.
 *     How close can you get with streams?
 *
 * 13. Kernel Overlap:
 *     When do kernels actually run in parallel?
 *     Test different kernel sizes.
 *
 * ADVANCED PROJECTS:
 * 14. Video Pipeline:
 *     - Stream 0: Decode frame N
 *     - Stream 1: Process frame N-1
 *     - Stream 2: Encode frame N-2
 *     Build working pipeline!
 *
 * 15. Double Buffering:
 *     While GPU processes buffer A,
 *     CPU prepares buffer B.
 *     Implement ping-pong pattern.
 *
 * 16. Graph API:
 *     Learn CUDA Graphs (captures work).
 *     Compare to streams for repeated work.
 *
 * REAL APPLICATIONS:
 * 17. Image Batch Processing:
 *     Process 100 images with 4 streams.
 *     Overlap I/O and computation.
 *
 * 18. Neural Network Inference:
 *     Pipeline: Load weights → Compute → Save results.
 *     Minimize latency for real-time.
 *
 * 19. Monte Carlo Simulation:
 *     Run independent simulations in parallel streams.
 *     Aggregate results efficiently.
 *
 * PRODUCTION PATTERNS:
 * 20. Stream Per Thread:
 *     CPU threads each manage their own stream.
 *     Careful with synchronization!
 *
 * 21. Priority Streams:
 *     High priority for latency-sensitive work.
 *     Low priority for background tasks.
 *
 * MENTAL MODELS:
 * 
 * Model 1: The Highway
 * - Default stream = Single lane road
 * - Multiple streams = Multi-lane highway
 * - Cars (operations) can pass each other
 * 
 * Model 2: The Restaurant
 * - Single waiter = Sequential service
 * - Multiple waiters = Parallel service
 * - Kitchen (GPU) handles multiple orders
 * 
 * Model 3: The Pipeline
 * - Assembly line with stages
 * - Each stage can work independently
 * - Throughput > Latency
 * 
 * Master streams and you've mastered
 * professional GPU programming!
 */