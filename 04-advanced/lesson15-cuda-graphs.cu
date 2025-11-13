/*
 * Lesson 15: CUDA Graphs
 * Optimizing Kernel Launch Overhead
 *
 * When microseconds matter, CUDA Graphs eliminate the CPU-GPU 
 * communication bottleneck by pre-recording entire workflows.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <chrono>
#include <cmath>

// =====================================================
// PART 1: FIRST PRINCIPLES - Why CUDA Graphs?
// =====================================================

/*
 * THE PROBLEM:
 * 
 * Every kernel launch has overhead:
 * - CPU prepares arguments (~1-5 μs)
 * - CPU sends command to GPU (~5-20 μs)
 * - GPU starts execution
 * 
 * For small kernels, overhead > actual work!
 * 
 * SOLUTION: CUDA Graphs
 * - Record once, replay many times
 * - Near-zero launch overhead
 * - Perfect for iterative algorithms
 * 
 * Real-world impact:
 * - ML inference: 2-5x faster
 * - Iterative solvers: 10x+ speedup
 * - Real-time systems: Consistent latency
 */

// Timer
class Timer {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point start;
public:
    Timer() : start(Clock::now()) {}
    float elapsed() {
        return std::chrono::duration<float, std::micro>(  // microseconds!
            Clock::now() - start).count();
    }
};

// =====================================================
// PART 2: SIMPLE ITERATIVE WORKLOAD
// =====================================================

// Simple kernel that we'll call many times
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Another kernel for the workflow
__global__ void vectorScale(float *a, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] *= scale;
    }
}

// Complex workflow with multiple kernels
void iterativeWorkflow(float *d_a, float *d_b, float *d_c, int n, int iterations) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    
    for (int i = 0; i < iterations; i++) {
        // Step 1: Add vectors
        vectorAdd<<<grid, block>>>(d_a, d_b, d_c, n);
        
        // Step 2: Scale result
        vectorScale<<<grid, block>>>(d_c, 0.99f, n);
        
        // Step 3: Add back to a
        vectorAdd<<<grid, block>>>(d_a, d_c, d_a, n);
        
        // Step 4: Scale a
        vectorScale<<<grid, block>>>(d_a, 1.01f, n);
    }
}

// =====================================================
// PART 3: MANUAL GRAPH CREATION
// =====================================================

void demonstrateManualGraph(float *d_a, float *d_b, float *d_c, int n, int iterations) {
    printf("\n=== Manual Graph Creation ===\n");
    
    // Create graph
    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);
    
    // Create nodes
    std::vector<cudaGraphNode_t> nodes;
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    
    // Node 1: vectorAdd
    cudaKernelNodeParams addParams = {0};
    addParams.func = (void*)vectorAdd;
    addParams.gridDim = grid;
    addParams.blockDim = block;
    void* addArgs[] = {&d_a, &d_b, &d_c, &n};
    addParams.kernelParams = addArgs;
    
    cudaGraphNode_t addNode;
    cudaGraphAddKernelNode(&addNode, graph, nullptr, 0, &addParams);
    nodes.push_back(addNode);
    
    // Node 2: vectorScale (depends on Node 1)
    cudaKernelNodeParams scaleParams = {0};
    scaleParams.func = (void*)vectorScale;
    scaleParams.gridDim = grid;
    scaleParams.blockDim = block;
    float scale1 = 0.99f;
    void* scaleArgs[] = {&d_c, &scale1, &n};
    scaleParams.kernelParams = scaleArgs;
    
    cudaGraphNode_t scaleNode;
    cudaGraphAddKernelNode(&scaleNode, graph, &addNode, 1, &scaleParams);
    nodes.push_back(scaleNode);
    
    // Instantiate graph
    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    
    // Execute graph multiple times
    Timer timer;
    for (int i = 0; i < iterations; i++) {
        cudaGraphLaunch(graphExec, 0);
    }
    cudaDeviceSynchronize();
    float time = timer.elapsed();
    
    printf("Manual graph time: %.2f μs (%.2f μs per iteration)\n", 
           time, time / iterations);
    
    // Cleanup
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
}

// =====================================================
// PART 4: STREAM CAPTURE - THE EASY WAY
// =====================================================

void demonstrateStreamCapture(float *d_a, float *d_b, float *d_c, int n, int iterations) {
    printf("\n=== Stream Capture ===\n");
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Begin capture
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    
    // Record the workflow
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    
    vectorAdd<<<grid, block, 0, stream>>>(d_a, d_b, d_c, n);
    vectorScale<<<grid, block, 0, stream>>>(d_c, 0.99f, n);
    vectorAdd<<<grid, block, 0, stream>>>(d_a, d_c, d_a, n);
    vectorScale<<<grid, block, 0, stream>>>(d_a, 1.01f, n);
    
    // End capture and create graph
    cudaGraph_t graph;
    cudaStreamEndCapture(stream, &graph);
    
    // Instantiate
    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    
    // Execute
    Timer timer;
    for (int i = 0; i < iterations; i++) {
        cudaGraphLaunch(graphExec, stream);
    }
    cudaStreamSynchronize(stream);
    float time = timer.elapsed();
    
    printf("Stream capture time: %.2f μs (%.2f μs per iteration)\n", 
           time, time / iterations);
    
    // Cleanup
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
}

// =====================================================
// PART 5: GRAPH UPDATES - CHANGING PARAMETERS
// =====================================================

__global__ void parameterizedKernel(float *data, float alpha, float beta, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = alpha * data[idx] + beta;
    }
}

void demonstrateGraphUpdate(float *d_data, int n) {
    printf("\n=== Graph Updates ===\n");
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    
    // Initial parameters
    float alpha = 2.0f, beta = 1.0f;
    
    // Capture graph
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    parameterizedKernel<<<grid, block, 0, stream>>>(d_data, alpha, beta, n);
    cudaGraph_t graph;
    cudaStreamEndCapture(stream, &graph);
    
    // Get kernel node
    size_t numNodes = 0;
    cudaGraphGetNodes(graph, nullptr, &numNodes);
    std::vector<cudaGraphNode_t> nodes(numNodes);
    cudaGraphGetNodes(graph, nodes.data(), &numNodes);
    
    // Instantiate
    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    
    // Execute with different parameters
    for (int i = 0; i < 5; i++) {
        // Update parameters
        alpha = 1.0f + i * 0.1f;
        beta = 2.0f - i * 0.1f;
        
        // Update kernel node
        cudaKernelNodeParams params;
        cudaGraphKernelNodeGetParams(nodes[0], &params);
        void* args[] = {&d_data, &alpha, &beta, &n};
        params.kernelParams = args;
        
        // Try to update existing graph
        cudaGraphExecKernelNodeSetParams(graphExec, nodes[0], &params);
        
        // Launch
        cudaGraphLaunch(graphExec, stream);
        cudaStreamSynchronize(stream);
        
        printf("Iteration %d: alpha=%.1f, beta=%.1f\n", i, alpha, beta);
    }
    
    // Cleanup
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
}

// =====================================================
// PART 6: CONDITIONAL EXECUTION & GRAPH TEMPLATES
// =====================================================

// Different paths based on condition
__global__ void conditionalPath1(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = sqrtf(data[idx]);
}

__global__ void conditionalPath2(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = data[idx] * data[idx];
}

void demonstrateConditionalGraph(float *d_data, int n) {
    printf("\n=== Conditional Graphs ===\n");
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    
    // Create two graph templates
    cudaGraph_t graphPath1, graphPath2;
    cudaGraphExec_t execPath1, execPath2;
    
    // Path 1: Square root
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    conditionalPath1<<<grid, block, 0, stream>>>(d_data, n);
    cudaStreamEndCapture(stream, &graphPath1);
    cudaGraphInstantiate(&execPath1, graphPath1, nullptr, nullptr, 0);
    
    // Path 2: Square
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    conditionalPath2<<<grid, block, 0, stream>>>(d_data, n);
    cudaStreamEndCapture(stream, &graphPath2);
    cudaGraphInstantiate(&execPath2, graphPath2, nullptr, nullptr, 0);
    
    // Execute based on runtime condition
    for (int i = 0; i < 10; i++) {
        if (i % 2 == 0) {
            cudaGraphLaunch(execPath1, stream);
            printf("Iteration %d: Taking square root\n", i);
        } else {
            cudaGraphLaunch(execPath2, stream);
            printf("Iteration %d: Squaring values\n", i);
        }
    }
    cudaStreamSynchronize(stream);
    
    // Cleanup
    cudaGraphExecDestroy(execPath1);
    cudaGraphExecDestroy(execPath2);
    cudaGraphDestroy(graphPath1);
    cudaGraphDestroy(graphPath2);
    cudaStreamDestroy(stream);
}

// =====================================================
// PART 7: PERFORMANCE COMPARISON
// =====================================================

// Jacobi iteration - perfect use case for graphs
__global__ void jacobiKernel(float *u_new, const float *u_old, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;  // Skip boundaries
    
    if (i < n - 1) {
        // 1D Jacobi: u_new[i] = 0.5 * (u_old[i-1] + u_old[i+1])
        u_new[i] = 0.5f * (u_old[i-1] + u_old[i+1]);
    }
}

void comparePerformance(int n, int iterations) {
    printf("\n=== Performance Comparison: Jacobi Iteration ===\n");
    printf("Grid size: %d, Iterations: %d\n", n, iterations);
    
    // Allocate memory
    float *d_u1, *d_u2;
    cudaMalloc(&d_u1, n * sizeof(float));
    cudaMalloc(&d_u2, n * sizeof(float));
    
    // Initialize
    cudaMemset(d_u1, 0, n * sizeof(float));
    cudaMemset(d_u2, 0, n * sizeof(float));
    
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    
    // Method 1: Traditional launches
    Timer timer1;
    for (int iter = 0; iter < iterations; iter++) {
        if (iter % 2 == 0) {
            jacobiKernel<<<grid, block>>>(d_u2, d_u1, n);
        } else {
            jacobiKernel<<<grid, block>>>(d_u1, d_u2, n);
        }
    }
    cudaDeviceSynchronize();
    float traditional_time = timer1.elapsed();
    printf("Traditional: %.2f μs\n", traditional_time);
    
    // Method 2: With streams
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    Timer timer2;
    for (int iter = 0; iter < iterations; iter++) {
        if (iter % 2 == 0) {
            jacobiKernel<<<grid, block, 0, stream>>>(d_u2, d_u1, n);
        } else {
            jacobiKernel<<<grid, block, 0, stream>>>(d_u1, d_u2, n);
        }
    }
    cudaStreamSynchronize(stream);
    float stream_time = timer2.elapsed();
    printf("With streams: %.2f μs\n", stream_time);
    
    // Method 3: CUDA Graph
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    
    // Record both iterations as one graph
    jacobiKernel<<<grid, block, 0, stream>>>(d_u2, d_u1, n);
    jacobiKernel<<<grid, block, 0, stream>>>(d_u1, d_u2, n);
    
    cudaGraph_t graph;
    cudaStreamEndCapture(stream, &graph);
    
    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    
    Timer timer3;
    for (int iter = 0; iter < iterations / 2; iter++) {
        cudaGraphLaunch(graphExec, stream);
    }
    cudaStreamSynchronize(stream);
    float graph_time = timer3.elapsed();
    printf("CUDA Graph: %.2f μs (%.1fx speedup!)\n", 
           graph_time, traditional_time / graph_time);
    
    // Calculate overhead
    float overhead_per_launch = (traditional_time - graph_time) / iterations;
    printf("Launch overhead: %.2f μs per kernel\n", overhead_per_launch);
    
    // Cleanup
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_u1);
    cudaFree(d_u2);
}

// =====================================================
// PART 8: MAIN - COMPREHENSIVE DEMONSTRATION
// =====================================================

int main() {
    printf("==================================================\n");
    printf("CUDA GRAPHS\n");
    printf("==================================================\n");
    
    // Setup
    const int N = 1000000;
    const int ITERATIONS = 1000;
    
    float *d_a, *d_b, *d_c, *d_data;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));
    cudaMalloc(&d_data, N * sizeof(float));
    
    // Initialize
    cudaMemset(d_a, 1, N * sizeof(float));
    cudaMemset(d_b, 2, N * sizeof(float));
    cudaMemset(d_data, 3, N * sizeof(float));
    
    // Test 1: Compare traditional vs graph
    printf("=== Traditional vs Graph Launch ===\n");
    
    // Traditional
    Timer trad_timer;
    iterativeWorkflow(d_a, d_b, d_c, N, ITERATIONS);
    cudaDeviceSynchronize();
    float trad_time = trad_timer.elapsed();
    printf("Traditional: %.2f μs total (%.3f μs per iteration)\n", 
           trad_time, trad_time / ITERATIONS);
    
    // Reset data
    cudaMemset(d_a, 1, N * sizeof(float));
    
    // With graph (stream capture)
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    iterativeWorkflow(d_a, d_b, d_c, N, 1);  // Capture one iteration
    cudaGraph_t graph;
    cudaStreamEndCapture(stream, &graph);
    
    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    
    Timer graph_timer;
    for (int i = 0; i < ITERATIONS; i++) {
        cudaGraphLaunch(graphExec, stream);
    }
    cudaStreamSynchronize(stream);
    float graph_time = graph_timer.elapsed();
    printf("Graph: %.2f μs total (%.3f μs per iteration)\n", 
           graph_time, graph_time / ITERATIONS);
    printf("Speedup: %.2fx\n", trad_time / graph_time);
    
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    
    // Test other features
    demonstrateManualGraph(d_a, d_b, d_c, N, 100);
    demonstrateStreamCapture(d_a, d_b, d_c, N, 100);
    demonstrateGraphUpdate(d_data, N);
    demonstrateConditionalGraph(d_data, N);
    
    // Performance comparison with real workload
    comparePerformance(10000, 10000);
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_data);
    
    printf("\n==================================================\n");
    printf("KEY INSIGHTS\n");
    printf("==================================================\n");
    printf("1. Graphs eliminate kernel launch overhead\n");
    printf("2. Best for iterative/repetitive workloads\n");
    printf("3. Stream capture is easiest to use\n");
    printf("4. Can update parameters without rebuilding\n");
    printf("5. Overhead reduction: 5-20μs → ~0.1μs per launch\n");
    printf("6. Essential for low-latency inference\n");
    
    return 0;
}

/*
 * COMPREHENSIVE EXERCISES:
 *
 * === Understanding ===
 * 1. Measure kernel launch overhead on your GPU
 * 2. When do graphs provide the most benefit?
 * 3. What operations cannot be captured in graphs?
 * 4. How do graphs interact with CUDA streams?
 * 5. Compare graph overhead vs kernel execution time
 *
 * === Implementation ===
 * 6. Build a graph for FFT operations
 * 7. Create a graph-based neural network inference
 * 8. Implement graph-based image processing pipeline
 * 9. Build conditional graphs for adaptive algorithms
 * 10. Create a graph pool for different batch sizes
 *
 * === Optimization ===
 * 11. Minimize graph instantiation time
 * 12. Optimize graph update frequency
 * 13. Compare different graph topologies
 * 14. Profile graph execution with Nsight
 * 15. Combine graphs with persistent kernels
 *
 * === Advanced ===
 * 16. Build hierarchical graphs (graphs of graphs)
 * 17. Implement graph-based auto-tuning
 * 18. Create dynamic graph generation
 * 19. Mix CPU callbacks with GPU graphs
 * 20. Build a graph compiler for DSL
 *
 * === Production ===
 * 21. Error handling in graph execution
 * 22. Graph versioning and caching
 * 23. Multi-GPU graph distribution
 * 24. Graph debugging techniques
 * 25. Performance regression testing
 */

/*
 * MENTAL MODELS:
 *
 * 1. "Recipe vs Cooking"
 *    - Traditional: Chef reads each step
 *    - Graph: Pre-recorded instructions
 *    - Result: Much faster execution
 *
 * 2. "Highway vs City Streets"
 *    - Traditional: Stop at every intersection
 *    - Graph: Express lane, no stops
 *    - Overhead: Traffic lights eliminated
 *
 * 3. "Compiled vs Interpreted"
 *    - Traditional: Interpret each command
 *    - Graph: Pre-compiled execution plan
 *    - Benefit: Near-zero overhead
 *
 * 4. When to Use Graphs:
 *    - Many small kernels (overhead dominates)
 *    - Iterative algorithms (same pattern)
 *    - Low-latency requirements (every μs counts)
 *    - Static workflows (graph reuse)
 */
