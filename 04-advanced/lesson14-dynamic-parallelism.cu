/*
 * Lesson 14: Dynamic Parallelism
 * Launching Kernels from Kernels
 *
 * The GPU becomes self-sufficient - no more CPU babysitting!
 * Perfect for adaptive algorithms and recursive problems.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <vector>

// =====================================================
// PART 1: FIRST PRINCIPLES - Why Dynamic Parallelism?
// =====================================================

/*
 * TRADITIONAL GPU PROGRAMMING:
 * CPU: "Do task A"
 * GPU: "Done"
 * CPU: "Now do task B"
 * GPU: "Done"
 * ... lots of back-and-forth
 * 
 * DYNAMIC PARALLELISM:
 * CPU: "Solve this problem"
 * GPU: "I'll figure out the details myself!"
 * 
 * PERFECT FOR:
 * - Adaptive algorithms (subdivide until small enough)
 * - Recursive problems (QuickSort, tree traversal)
 * - Irregular parallelism (spawn work as needed)
 * - Reducing CPU-GPU synchronization
 * 
 * Note: Requires CC 3.5+ and -rdc=true compilation
 */

// Timer
class Timer {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point start;
public:
    Timer() : start(Clock::now()) {}
    float elapsed() {
        return std::chrono::duration<float, std::milli>(
            Clock::now() - start).count();
    }
};

// =====================================================
// PART 2: BASIC DYNAMIC PARALLELISM
// =====================================================

// Child kernel - called from parent
__global__ void childKernel(int *data, int offset, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    
    if (idx < offset + n) {
        // Simple work
        data[idx] = data[idx] * 2 + 1;
        
        // Child can also use printf
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("Child kernel: processing elements %d to %d\n", 
                   offset, offset + n - 1);
        }
    }
}

// Parent kernel - launches children
__global__ void parentKernel(int *data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid == 0) {
        printf("Parent kernel: Launching children...\n");
        
        // Launch child kernels
        int chunk_size = n / 4;
        for (int i = 0; i < 4; i++) {
            int offset = i * chunk_size;
            int size = (i == 3) ? n - offset : chunk_size;
            
            // Launch from device!
            childKernel<<<(size + 255) / 256, 256>>>(data, offset, size);
        }
        
        // IMPORTANT: Synchronize with children
        cudaDeviceSynchronize();
        
        printf("Parent kernel: All children completed\n");
    }
}

// =====================================================
// PART 3: RECURSIVE ALGORITHMS - QUICKSORT
// =====================================================

__device__ void swap(int &a, int &b) {
    int temp = a;
    a = b;
    b = temp;
}

__device__ int partition(int *data, int left, int right) {
    int pivot = data[right];
    int i = left - 1;
    
    for (int j = left; j < right; j++) {
        if (data[j] <= pivot) {
            i++;
            swap(data[i], data[j]);
        }
    }
    
    swap(data[i + 1], data[right]);
    return i + 1;
}

// GPU QuickSort using dynamic parallelism
__global__ void quickSortGPU(int *data, int left, int right, int depth) {
    if (left < right) {
        // Partition
        int pivotIndex = partition(data, left, right);
        
        // Limit recursion depth to avoid stack overflow
        if (depth < 16) {
            // Launch two child kernels for sub-arrays
            cudaStream_t s1, s2;
            cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
            cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
            
            // Sort left part
            if (pivotIndex - left > 1) {
                quickSortGPU<<<1, 1, 0, s1>>>(data, left, pivotIndex - 1, depth + 1);
            }
            
            // Sort right part
            if (right - pivotIndex > 1) {
                quickSortGPU<<<1, 1, 0, s2>>>(data, pivotIndex + 1, right, depth + 1);
            }
            
            // Wait for children
            cudaStreamDestroy(s1);
            cudaStreamDestroy(s2);
            cudaDeviceSynchronize();
        }
    }
}

// =====================================================
// PART 4: ADAPTIVE ALGORITHMS - QUADTREE
// =====================================================

struct Point {
    float x, y;
    float value;
};

struct QuadNode {
    float x_min, x_max, y_min, y_max;
    float sum;
    int count;
    int children[4];  // Indices of children (-1 if none)
};

__device__ int node_counter = 0;

// Adaptive quadtree construction
__global__ void buildQuadtree(Point *points, int n_points, 
                             QuadNode *nodes, int node_idx,
                             float x_min, float x_max, 
                             float y_min, float y_max,
                             int max_depth, int depth) {
    
    if (depth >= max_depth) return;
    
    // Count points in this node
    int count = 0;
    float sum = 0.0f;
    
    for (int i = 0; i < n_points; i++) {
        if (points[i].x >= x_min && points[i].x <= x_max &&
            points[i].y >= y_min && points[i].y <= y_max) {
            count++;
            sum += points[i].value;
        }
    }
    
    // Store node info
    nodes[node_idx].x_min = x_min;
    nodes[node_idx].x_max = x_max;
    nodes[node_idx].y_min = y_min;
    nodes[node_idx].y_max = y_max;
    nodes[node_idx].sum = sum;
    nodes[node_idx].count = count;
    
    // Subdivide if needed (adaptive!)
    if (count > 10) {  // Threshold
        float x_mid = (x_min + x_max) / 2;
        float y_mid = (y_min + y_max) / 2;
        
        // Allocate children
        int base_child = atomicAdd(&node_counter, 4);
        
        for (int i = 0; i < 4; i++) {
            nodes[node_idx].children[i] = base_child + i;
        }
        
        // Launch kernels for children
        cudaStream_t streams[4];
        for (int i = 0; i < 4; i++) {
            cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
        }
        
        // Quadrant subdivision
        buildQuadtree<<<1, 1, 0, streams[0]>>>(
            points, n_points, nodes, base_child + 0,
            x_min, x_mid, y_min, y_mid, max_depth, depth + 1);
            
        buildQuadtree<<<1, 1, 0, streams[1]>>>(
            points, n_points, nodes, base_child + 1,
            x_mid, x_max, y_min, y_mid, max_depth, depth + 1);
            
        buildQuadtree<<<1, 1, 0, streams[2]>>>(
            points, n_points, nodes, base_child + 2,
            x_min, x_mid, y_mid, y_max, max_depth, depth + 1);
            
        buildQuadtree<<<1, 1, 0, streams[3]>>>(
            points, n_points, nodes, base_child + 3,
            x_mid, x_max, y_mid, y_max, max_depth, depth + 1);
        
        // Clean up streams
        for (int i = 0; i < 4; i++) {
            cudaStreamDestroy(streams[i]);
        }
        
        cudaDeviceSynchronize();
    } else {
        // Leaf node
        for (int i = 0; i < 4; i++) {
            nodes[node_idx].children[i] = -1;
        }
    }
}

// =====================================================
// PART 5: WORK GENERATION - MANDELBROT SET
// =====================================================

__device__ int mandelbrot(float x0, float y0, int max_iter) {
    float x = 0, y = 0;
    int iter = 0;
    
    while (x*x + y*y <= 4 && iter < max_iter) {
        float xtemp = x*x - y*y + x0;
        y = 2*x*y + y0;
        x = xtemp;
        iter++;
    }
    
    return iter;
}

// Adaptive Mandelbrot - subdivide interesting regions
__global__ void mandelbrotAdaptive(int *output, int width, int height,
                                  float x_min, float x_max,
                                  float y_min, float y_max,
                                  int level, int max_level) {
    
    // Sample the region
    int samples = 8;
    int variations = 0;
    int last_value = -1;
    
    for (int sy = 0; sy < samples; sy++) {
        for (int sx = 0; sx < samples; sx++) {
            float x = x_min + (x_max - x_min) * sx / (samples - 1);
            float y = y_min + (y_max - y_min) * sy / (samples - 1);
            int value = mandelbrot(x, y, 256);
            
            if (last_value != -1 && value != last_value) {
                variations++;
            }
            last_value = value;
        }
    }
    
    // Decide whether to subdivide
    if (variations > 2 && level < max_level) {
        // Subdivide into 4 quadrants
        float x_mid = (x_min + x_max) / 2;
        float y_mid = (y_min + y_max) / 2;
        
        dim3 grid(2, 2);
        mandelbrotAdaptive<<<grid, 1>>>(
            output, width, height, 
            x_min, x_mid, y_min, y_mid, 
            level + 1, max_level);
            
        // ... launch other 3 quadrants
        
        cudaDeviceSynchronize();
    } else {
        // Compute this region at full resolution
        int x_start = x_min * width;
        int x_end = x_max * width;
        int y_start = y_min * height;
        int y_end = y_max * height;
        
        dim3 block(16, 16);
        dim3 grid((x_end - x_start + 15) / 16, 
                  (y_end - y_start + 15) / 16);
        
        // Launch fine-grained computation
        // ... (actual Mandelbrot computation kernel)
    }
}

// =====================================================
// PART 6: MEMORY MANAGEMENT IN DYNAMIC PARALLELISM
// =====================================================

__global__ void memoryExample() {
    // Device malloc works in kernels!
    int *device_array = (int*)malloc(100 * sizeof(int));
    
    if (device_array != nullptr) {
        // Initialize
        for (int i = 0; i < 100; i++) {
            device_array[i] = i;
        }
        
        // Launch child with dynamic memory
        // childKernel<<<1, 100>>>(device_array, 0, 100);
        // cudaDeviceSynchronize();
        
        // Don't forget to free!
        free(device_array);
    }
}

// =====================================================
// PART 7: PERFORMANCE CONSIDERATIONS
// =====================================================

void comparePerformance() {
    printf("\n=== Performance Comparison ===\n");
    
    const int N = 1000000;
    int *h_data = new int[N];
    int *d_data;
    
    // Initialize
    for (int i = 0; i < N; i++) {
        h_data[i] = rand() % 1000;
    }
    
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Traditional approach - CPU launches multiple kernels
    Timer cpu_launch_timer;
    for (int i = 0; i < 4; i++) {
        int offset = i * (N / 4);
        int size = (i == 3) ? N - offset : N / 4;
        childKernel<<<(size + 255) / 256, 256>>>(d_data, offset, size);
    }
    cudaDeviceSynchronize();
    float cpu_launch_time = cpu_launch_timer.elapsed();
    
    // Reset data
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Dynamic parallelism approach
    Timer dynamic_timer;
    parentKernel<<<1, 1>>>(d_data, N);
    cudaDeviceSynchronize();
    float dynamic_time = dynamic_timer.elapsed();
    
    printf("Traditional (CPU launches): %.2f ms\n", cpu_launch_time);
    printf("Dynamic parallelism: %.2f ms\n", dynamic_time);
    printf("Overhead: %.2f ms\n", dynamic_time - cpu_launch_time);
    
    // Cleanup
    cudaFree(d_data);
    delete[] h_data;
}

// =====================================================
// PART 8: MAIN - DEMONSTRATION
// =====================================================

// Check if device supports dynamic parallelism
bool checkDynamicParallelism() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    
    if (prop.major < 3 || (prop.major == 3 && prop.minor < 5)) {
        printf("✗ Dynamic Parallelism NOT supported (need CC 3.5+)\n");
        return false;
    }
    
    printf("✓ Dynamic Parallelism supported\n");
    printf("Max depth: %d\n", prop.maxGridSize[2]);
    return true;
}

int main() {
    printf("==================================================\n");
    printf("DYNAMIC PARALLELISM\n");
    printf("==================================================\n\n");
    
    if (!checkDynamicParallelism()) {
        printf("\nThis GPU doesn't support dynamic parallelism.\n");
        printf("You need a GPU with Compute Capability 3.5 or higher.\n");
        return 1;
    }
    
    printf("\n=== Basic Example ===\n");
    
    // Basic parent-child example
    int *d_data;
    int n = 1024;
    cudaMalloc(&d_data, n * sizeof(int));
    
    // Initialize
    int *h_data = new int[n];
    for (int i = 0; i < n; i++) h_data[i] = i;
    cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch parent
    parentKernel<<<1, 1>>>(d_data, n);
    cudaDeviceSynchronize();
    
    // Verify
    cudaMemcpy(h_data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (h_data[i] != i * 2 + 1) {
            correct = false;
            break;
        }
    }
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");
    
    // QuickSort example
    printf("\n=== QuickSort Example ===\n");
    
    // Generate random data
    for (int i = 0; i < n; i++) {
        h_data[i] = rand() % 1000;
    }
    cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);
    
    // Sort
    Timer sort_timer;
    quickSortGPU<<<1, 1>>>(d_data, 0, n - 1, 0);
    cudaDeviceSynchronize();
    float sort_time = sort_timer.elapsed();
    
    // Verify sorted
    cudaMemcpy(h_data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);
    bool sorted = true;
    for (int i = 1; i < n; i++) {
        if (h_data[i] < h_data[i-1]) {
            sorted = false;
            break;
        }
    }
    
    printf("QuickSort time: %.2f ms\n", sort_time);
    printf("Sorted correctly: %s\n", sorted ? "YES" : "NO");
    
    // Performance comparison
    comparePerformance();
    
    // Cleanup
    cudaFree(d_data);
    delete[] h_data;
    
    printf("\n==================================================\n");
    printf("KEY INSIGHTS\n");
    printf("==================================================\n");
    printf("1. GPU can launch its own work (no CPU needed)\n");
    printf("2. Perfect for adaptive/recursive algorithms\n");
    printf("3. Overhead exists - use for coarse-grained work\n");
    printf("4. Limited recursion depth (stack size)\n");
    printf("5. Device malloc/free available in kernels\n");
    printf("6. Compile with -rdc=true -lcudadevrt\n");
    
    printf("\n=== Compilation ===\n");
    printf("nvcc -arch=sm_35 -rdc=true -lcudadevrt dynamic.cu\n");
    
    return 0;
}

/*
 * COMPREHENSIVE EXERCISES:
 *
 * === Understanding ===
 * 1. Why does DP require CC 3.5+?
 * 2. Calculate launch overhead vs computation
 * 3. When is DP faster than CPU control?
 * 4. What's the stack limit for recursion?
 * 5. How does device malloc differ from cudaMalloc?
 *
 * === Implementation ===
 * 6. Implement merge sort with DP
 * 7. Create adaptive mesh refinement
 * 8. Build recursive tree traversal
 * 9. Implement parallel BFS with DP
 * 10. Create work-stealing queue
 *
 * === Optimization ===
 * 11. Minimize kernel launch overhead
 * 12. Balance recursion depth vs parallelism
 * 13. Optimize stream usage in DP
 * 14. Profile parent-child communication
 * 15. Compare DP vs iterative approaches
 *
 * === Advanced ===
 * 16. Build recursive neural network
 * 17. Implement Barnes-Hut algorithm
 * 18. Create adaptive ray tracing
 * 19. Build GPU-only solver
 * 20. Implement divide-and-conquer FFT
 *
 * === Applications ===
 * 21. Adaptive physics simulation
 * 22. Hierarchical clustering
 * 23. Recursive image processing
 * 24. Dynamic load balancing
 * 25. Autonomous GPU computing
 */

/*
 * MENTAL MODELS:
 *
 * 1. "Autonomous GPU"
 *    - GPU becomes independent
 *    - Decides its own work
 *    - No CPU supervision needed
 *
 * 2. "Recursive Thinking"
 *    - Break problem into subproblems
 *    - GPU spawns children
 *    - Natural for tree algorithms
 *
 * 3. "Adaptive Computation"
 *    - Compute where needed
 *    - Skip where not needed
 *    - Dynamic work generation
 *
 * 4. When to Use:
 *    - Irregular workloads
 *    - Recursive algorithms
 *    - Adaptive refinement
 *    - Reducing CPU-GPU sync
 */
