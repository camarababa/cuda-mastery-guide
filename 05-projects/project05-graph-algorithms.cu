/*
 * PROJECT 5: GPU GRAPH ALGORITHMS
 * Mastering Irregular Parallelism
 *
 * Graphs are everywhere: social networks, web pages, road networks,
 * neural networks. This project teaches you to process millions of
 * edges in parallel on GPU.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <queue>
#include <algorithm>
#include <random>
#include <chrono>
#include <limits>

// =====================================================
// PART 1: FIRST PRINCIPLES - Why Graph Algorithms on GPU?
// =====================================================

/*
 * THE CHALLENGE:
 * 
 * Graphs have irregular structure:
 * - Vertices have varying degrees
 * - No regular memory access pattern
 * - Work imbalance between threads
 * - Lots of pointer chasing
 * 
 * THE OPPORTUNITY:
 * 
 * Many graph algorithms have massive parallelism:
 * - Process all vertices/edges simultaneously
 * - Frontier-based exploration
 * - Matrix-style operations
 * 
 * Real-world applications:
 * - Social network analysis (Facebook graph)
 * - Web ranking (Google PageRank)
 * - Route planning (GPS navigation)
 * - Recommendation systems
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
// PART 2: GRAPH REPRESENTATIONS
// =====================================================

// Compressed Sparse Row (CSR) format - most common for GPU
struct CSRGraph {
    int num_vertices;
    int num_edges;
    int *row_offsets;     // Size: num_vertices + 1
    int *column_indices;  // Size: num_edges
    float *edge_weights;  // Size: num_edges (optional)
    
    // Constructor
    CSRGraph(int v, int e) : num_vertices(v), num_edges(e) {
        cudaMalloc(&row_offsets, (num_vertices + 1) * sizeof(int));
        cudaMalloc(&column_indices, num_edges * sizeof(int));
        cudaMalloc(&edge_weights, num_edges * sizeof(float));
    }
    
    ~CSRGraph() {
        cudaFree(row_offsets);
        cudaFree(column_indices);
        cudaFree(edge_weights);
    }
    
    // Get neighbors of vertex v
    __device__ void get_neighbors(int v, int &start, int &end) {
        start = row_offsets[v];
        end = row_offsets[v + 1];
    }
};

// Edge list format (for dynamic graphs)
struct Edge {
    int src;
    int dst;
    float weight;
};

// =====================================================
// PART 3: BREADTH-FIRST SEARCH (BFS)
// =====================================================

// CPU BFS for verification
void bfsCPU(CSRGraph &g, int source, int *distances) {
    std::vector<int> h_row_offsets(g.num_vertices + 1);
    std::vector<int> h_col_indices(g.num_edges);
    
    cudaMemcpy(h_row_offsets.data(), g.row_offsets, 
              (g.num_vertices + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_col_indices.data(), g.column_indices, 
              g.num_edges * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Initialize distances
    std::fill(distances, distances + g.num_vertices, -1);
    distances[source] = 0;
    
    std::queue<int> frontier;
    frontier.push(source);
    
    while (!frontier.empty()) {
        int v = frontier.front();
        frontier.pop();
        
        // Visit neighbors
        for (int i = h_row_offsets[v]; i < h_row_offsets[v + 1]; i++) {
            int neighbor = h_col_indices[i];
            if (distances[neighbor] == -1) {
                distances[neighbor] = distances[v] + 1;
                frontier.push(neighbor);
            }
        }
    }
}

// GPU BFS - Vertex-parallel (each thread processes one vertex)
__global__ void bfsKernel(CSRGraph g, int *distances, bool *frontier, 
                         bool *visited, bool *done, int level) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (v < g.num_vertices && frontier[v] && !visited[v]) {
        visited[v] = true;
        distances[v] = level;
        
        // Visit neighbors
        int start, end;
        g.get_neighbors(v, start, end);
        
        for (int i = start; i < end; i++) {
            int neighbor = g.column_indices[i];
            if (!visited[neighbor]) {
                frontier[neighbor] = true;
                *done = false;
            }
        }
    }
}

// GPU BFS - Edge-parallel (better for high-degree graphs)
__global__ void bfsEdgeParallel(CSRGraph g, int *distances, bool *frontier,
                               bool *next_frontier, bool *done, int level) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < g.num_edges) {
        // Find which vertex owns this edge
        int src = -1;
        for (int v = 0; v < g.num_vertices; v++) {
            if (tid >= g.row_offsets[v] && tid < g.row_offsets[v + 1]) {
                src = v;
                break;
            }
        }
        
        if (src != -1 && frontier[src] && distances[src] == level - 1) {
            int dst = g.column_indices[tid];
            if (distances[dst] == -1) {
                distances[dst] = level;
                next_frontier[dst] = true;
                *done = false;
            }
        }
    }
}

// =====================================================
// PART 4: SINGLE SOURCE SHORTEST PATH (SSSP)
// =====================================================

// Bellman-Ford SSSP
__global__ void bellmanFordKernel(CSRGraph g, float *distances, bool *updated) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (v < g.num_vertices) {
        int start, end;
        g.get_neighbors(v, start, end);
        
        for (int i = start; i < end; i++) {
            int neighbor = g.column_indices[i];
            float weight = g.edge_weights[i];
            float new_dist = distances[v] + weight;
            
            // Atomic min for race condition
            float old_dist = atomicExch(&distances[neighbor], new_dist);
            if (new_dist < old_dist) {
                atomicMin(&distances[neighbor], new_dist);
                *updated = true;
            } else {
                atomicMin(&distances[neighbor], old_dist);
            }
        }
    }
}

// =====================================================
// PART 5: PAGERANK ALGORITHM
// =====================================================

// Initialize PageRank values
__global__ void initPageRank(float *pagerank, int n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) {
        pagerank[v] = 1.0f / n;
    }
}

// PageRank iteration - pull-based
__global__ void pageRankPull(CSRGraph g, float *pagerank, float *new_pagerank,
                            float damping) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (v < g.num_vertices) {
        float sum = 0.0f;
        int start, end;
        g.get_neighbors(v, start, end);
        
        // Sum contributions from incoming edges
        for (int i = start; i < end; i++) {
            int neighbor = g.column_indices[i];
            
            // Get out-degree of neighbor
            int neighbor_degree = g.row_offsets[neighbor + 1] - 
                                g.row_offsets[neighbor];
            
            if (neighbor_degree > 0) {
                sum += pagerank[neighbor] / neighbor_degree;
            }
        }
        
        new_pagerank[v] = (1.0f - damping) / g.num_vertices + damping * sum;
    }
}

// PageRank convergence check
__global__ void checkConvergence(float *pagerank, float *new_pagerank,
                               float *diff, int n) {
    __shared__ float shared_diff[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    float local_diff = 0.0f;
    if (gid < n) {
        local_diff = fabsf(new_pagerank[gid] - pagerank[gid]);
    }
    
    shared_diff[tid] = local_diff;
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_diff[tid] = fmaxf(shared_diff[tid], shared_diff[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicMax((int*)diff, __float_as_int(shared_diff[0]));
    }
}

// =====================================================
// PART 6: GRAPH GENERATION AND UTILITIES
// =====================================================

// Generate random graph (Erdős–Rényi model)
CSRGraph* generateRandomGraph(int num_vertices, float edge_probability) {
    std::vector<std::vector<std::pair<int, float>>> adj_list(num_vertices);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    int num_edges = 0;
    for (int i = 0; i < num_vertices; i++) {
        for (int j = 0; j < num_vertices; j++) {
            if (i != j && dist(gen) < edge_probability) {
                float weight = dist(gen) * 10.0f + 1.0f;
                adj_list[i].push_back({j, weight});
                num_edges++;
            }
        }
    }
    
    // Convert to CSR
    CSRGraph* g = new CSRGraph(num_vertices, num_edges);
    
    std::vector<int> h_row_offsets(num_vertices + 1);
    std::vector<int> h_col_indices;
    std::vector<float> h_weights;
    
    int offset = 0;
    for (int i = 0; i < num_vertices; i++) {
        h_row_offsets[i] = offset;
        for (auto &edge : adj_list[i]) {
            h_col_indices.push_back(edge.first);
            h_weights.push_back(edge.second);
        }
        offset += adj_list[i].size();
    }
    h_row_offsets[num_vertices] = offset;
    
    // Copy to GPU
    cudaMemcpy(g->row_offsets, h_row_offsets.data(), 
              (num_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g->column_indices, h_col_indices.data(), 
              num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g->edge_weights, h_weights.data(), 
              num_edges * sizeof(float), cudaMemcpyHostToDevice);
    
    return g;
}

// =====================================================
// PART 7: OPTIMIZATIONS - WORK EFFICIENCY
// =====================================================

// Warp-centric BFS (better load balancing)
__global__ void bfsWarpCentric(CSRGraph g, int *distances, bool *frontier,
                              bool *next_frontier, int level) {
    __shared__ int shared_vertices[1024];
    __shared__ int shared_count;
    
    if (threadIdx.x == 0) shared_count = 0;
    __syncthreads();
    
    // Cooperative loading of frontier vertices
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    // Load frontier vertices to shared memory
    for (int v = tid; v < g.num_vertices; v += blockDim.x) {
        if (frontier[v] && distances[v] == level - 1) {
            int pos = atomicAdd(&shared_count, 1);
            if (pos < 1024) shared_vertices[pos] = v;
        }
    }
    __syncthreads();
    
    // Process vertices with warps
    for (int i = warp_id; i < shared_count; i += blockDim.x / 32) {
        int v = shared_vertices[i];
        int start = g.row_offsets[v];
        int end = g.row_offsets[v + 1];
        
        // Warp processes neighbors
        for (int j = start + lane; j < end; j += 32) {
            int neighbor = g.column_indices[j];
            if (atomicCAS(&distances[neighbor], -1, level) == -1) {
                next_frontier[neighbor] = true;
            }
        }
    }
}

// =====================================================
// PART 8: MAIN - COMPREHENSIVE TESTING
// =====================================================

void testBFS(CSRGraph *g, int source) {
    printf("\n=== Breadth-First Search ===\n");
    
    // Allocate memory
    int *d_distances, *h_distances_gpu, *h_distances_cpu;
    bool *d_frontier, *d_visited, *d_done;
    
    cudaMalloc(&d_distances, g->num_vertices * sizeof(int));
    cudaMalloc(&d_frontier, g->num_vertices * sizeof(bool));
    cudaMalloc(&d_visited, g->num_vertices * sizeof(bool));
    cudaMalloc(&d_done, sizeof(bool));
    
    h_distances_gpu = new int[g->num_vertices];
    h_distances_cpu = new int[g->num_vertices];
    
    // CPU BFS
    Timer cpu_timer;
    bfsCPU(*g, source, h_distances_cpu);
    float cpu_time = cpu_timer.elapsed();
    printf("CPU BFS: %.2f ms\n", cpu_time);
    
    // GPU BFS initialization
    cudaMemset(d_distances, -1, g->num_vertices * sizeof(int));
    cudaMemset(d_frontier, 0, g->num_vertices * sizeof(bool));
    cudaMemset(d_visited, 0, g->num_vertices * sizeof(bool));
    
    // Set source
    cudaMemcpy(&d_distances[source], &source, sizeof(int), cudaMemcpyHostToDevice);
    bool true_val = true;
    cudaMemcpy(&d_frontier[source], &true_val, sizeof(bool), cudaMemcpyHostToDevice);
    
    // GPU BFS
    Timer gpu_timer;
    int level = 0;
    bool done = false;
    
    dim3 block(256);
    dim3 grid((g->num_vertices + block.x - 1) / block.x);
    
    while (!done) {
        done = true;
        cudaMemcpy(d_done, &done, sizeof(bool), cudaMemcpyHostToDevice);
        
        bfsKernel<<<grid, block>>>(*g, d_distances, d_frontier, 
                                  d_visited, d_done, level);
        
        cudaMemcpy(&done, d_done, sizeof(bool), cudaMemcpyDeviceToHost);
        level++;
    }
    
    cudaDeviceSynchronize();
    float gpu_time = gpu_timer.elapsed();
    printf("GPU BFS: %.2f ms (speedup: %.2fx)\n", 
           gpu_time, cpu_time / gpu_time);
    
    // Verify results
    cudaMemcpy(h_distances_gpu, d_distances, 
              g->num_vertices * sizeof(int), cudaMemcpyDeviceToHost);
    
    bool correct = true;
    for (int i = 0; i < std::min(100, g->num_vertices); i++) {
        if (h_distances_cpu[i] != h_distances_gpu[i]) {
            correct = false;
            break;
        }
    }
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");
    printf("Max depth: %d\n", level - 1);
    
    // Cleanup
    cudaFree(d_distances);
    cudaFree(d_frontier);
    cudaFree(d_visited);
    cudaFree(d_done);
    delete[] h_distances_gpu;
    delete[] h_distances_cpu;
}

void testPageRank(CSRGraph *g) {
    printf("\n=== PageRank ===\n");
    
    const float damping = 0.85f;
    const float tolerance = 1e-4f;
    const int max_iterations = 100;
    
    // Allocate memory
    float *d_pagerank, *d_new_pagerank, *d_diff;
    cudaMalloc(&d_pagerank, g->num_vertices * sizeof(float));
    cudaMalloc(&d_new_pagerank, g->num_vertices * sizeof(float));
    cudaMalloc(&d_diff, sizeof(float));
    
    // Initialize
    dim3 block(256);
    dim3 grid((g->num_vertices + block.x - 1) / block.x);
    
    initPageRank<<<grid, block>>>(d_pagerank, g->num_vertices);
    
    // Iterate
    Timer timer;
    int iter = 0;
    float diff = 1.0f;
    
    while (diff > tolerance && iter < max_iterations) {
        // PageRank iteration
        pageRankPull<<<grid, block>>>(*g, d_pagerank, d_new_pagerank, damping);
        
        // Check convergence
        cudaMemset(d_diff, 0, sizeof(float));
        checkConvergence<<<grid, block>>>(d_pagerank, d_new_pagerank, 
                                         d_diff, g->num_vertices);
        
        cudaMemcpy(&diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
        diff = __int_as_float((int)diff);
        
        // Swap pointers
        std::swap(d_pagerank, d_new_pagerank);
        iter++;
    }
    
    cudaDeviceSynchronize();
    float time = timer.elapsed();
    
    printf("PageRank converged in %d iterations\n", iter);
    printf("Time: %.2f ms (%.2f ms per iteration)\n", time, time / iter);
    
    // Get top vertices
    std::vector<float> h_pagerank(g->num_vertices);
    cudaMemcpy(h_pagerank.data(), d_pagerank, 
              g->num_vertices * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::vector<std::pair<float, int>> ranked;
    for (int i = 0; i < g->num_vertices; i++) {
        ranked.push_back({h_pagerank[i], i});
    }
    std::sort(ranked.begin(), ranked.end(), std::greater<>());
    
    printf("Top 5 vertices by PageRank:\n");
    for (int i = 0; i < std::min(5, g->num_vertices); i++) {
        printf("  Vertex %d: %.6f\n", ranked[i].second, ranked[i].first);
    }
    
    // Cleanup
    cudaFree(d_pagerank);
    cudaFree(d_new_pagerank);
    cudaFree(d_diff);
}

int main() {
    printf("==================================================\n");
    printf("GPU GRAPH ALGORITHMS\n");
    printf("==================================================\n");
    
    // Test different graph sizes
    std::vector<std::pair<int, float>> test_cases = {
        {1000, 0.01f},    // Small sparse
        {1000, 0.1f},     // Small dense
        {10000, 0.001f},  // Medium sparse
        {10000, 0.01f},   // Medium dense
    };
    
    for (auto &test : test_cases) {
        int num_vertices = test.first;
        float edge_prob = test.second;
        
        printf("\n==================================================\n");
        printf("Graph: %d vertices, %.1f%% edge probability\n", 
               num_vertices, edge_prob * 100);
        
        // Generate graph
        CSRGraph *g = generateRandomGraph(num_vertices, edge_prob);
        printf("Generated %d edges (avg degree: %.1f)\n", 
               g->num_edges, (float)g->num_edges / num_vertices);
        
        // Test algorithms
        testBFS(g, 0);
        testPageRank(g);
        
        delete g;
    }
    
    printf("\n==================================================\n");
    printf("KEY INSIGHTS\n");
    printf("==================================================\n");
    printf("1. Graph algorithms have irregular parallelism\n");
    printf("2. Load balancing is crucial for performance\n");
    printf("3. Different algorithms need different approaches\n");
    printf("4. Memory access pattern matters more than compute\n");
    printf("5. Warp-centric methods handle imbalance better\n");
    printf("6. Real graphs need specialized optimizations\n");
    
    return 0;
}

/*
 * COMPREHENSIVE EXERCISES:
 *
 * === Understanding ===
 * 1. Why is load balancing hard for graphs?
 * 2. Compare CSR vs adjacency list representations
 * 3. When to use vertex-parallel vs edge-parallel?
 * 4. How does graph structure affect performance?
 * 5. Why are atomic operations common in graph algorithms?
 *
 * === Implementation ===
 * 6. Implement Dijkstra's algorithm on GPU
 * 7. Create connected components algorithm
 * 8. Build triangle counting kernel
 * 9. Implement betweenness centrality
 * 10. Create A* pathfinding for grid graphs
 *
 * === Optimization ===
 * 11. Implement direction-optimizing BFS
 * 12. Create push-pull PageRank hybrid
 * 13. Add vertex reordering for locality
 * 14. Implement graph compression
 * 15. Create multi-GPU graph partitioning
 *
 * === Advanced ===
 * 16. Build dynamic graph updates
 * 17. Implement GraphBLAS primitives
 * 18. Create GPU-accelerated GNN operations
 * 19. Build streaming graph algorithms
 * 20. Implement parallel graph coloring
 *
 * === Applications ===
 * 21. Social network influence propagation
 * 22. Web crawling and ranking
 * 23. Circuit simulation
 * 24. Transportation routing
 * 25. Biological network analysis
 */

/*
 * MENTAL MODELS:
 *
 * 1. "Frontier Expansion"
 *    - Like water spreading through pipes
 *    - Process active frontier in parallel
 *    - Expand to neighbors
 *
 * 2. "Load Balancing Challenge"
 *    - Some vertices: 1 neighbor
 *    - Others: 1000s of neighbors
 *    - Need smart work distribution
 *
 * 3. "Pull vs Push"
 *    - Pull: Vertices gather from neighbors
 *    - Push: Vertices send to neighbors
 *    - Choose based on frontier size
 *
 * 4. Graph Processing Patterns:
 *    - Vertex-centric: Think like a vertex
 *    - Edge-centric: Process all edges
 *    - Subgraph-centric: Process components
 */
