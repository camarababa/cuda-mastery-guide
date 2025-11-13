/*
 * PROJECT 4: GPU HASH TABLE
 * Building High-Performance Concurrent Data Structures
 *
 * Hash tables are fundamental to databases, caches, and many algorithms.
 * This project teaches you to build one that handles millions of 
 * operations per second on GPU.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <cassert>

// =====================================================
// PART 1: FIRST PRINCIPLES - Why GPU Hash Tables?
// =====================================================

/*
 * CHALLENGE:
 * CPUs handle hash tables well for sequential access.
 * But what if you need to process millions of lookups in parallel?
 * 
 * GPU ADVANTAGES:
 * - Massive parallelism for bulk operations
 * - High memory bandwidth
 * - Perfect for database joins, deduplication, analytics
 * 
 * GPU CHALLENGES:
 * - Concurrent insertion conflicts
 * - Dynamic memory allocation
 * - Load balancing with skewed data
 * 
 * We'll solve these systematically!
 */

// Configuration
const int EMPTY_KEY = -1;
const int EMPTY_VALUE = -1;
const int DELETED_KEY = -2;

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
// PART 2: BASIC HASH TABLE STRUCTURE
// =====================================================

struct KeyValue {
    int key;
    int value;
};

class GPUHashTable {
public:
    KeyValue* table;
    size_t capacity;
    size_t size;
    
    // Hash function (multiplicative hashing)
    __device__ __host__ inline uint32_t hash(int key) const {
        // Knuth's multiplicative hash
        return ((uint32_t)key * 2654435761u) % capacity;
    }
    
    // Constructor
    GPUHashTable(size_t cap) : capacity(cap), size(0) {
        cudaMalloc(&table, capacity * sizeof(KeyValue));
        
        // Initialize to empty
        KeyValue empty = {EMPTY_KEY, EMPTY_VALUE};
        KeyValue* h_table = new KeyValue[capacity];
        std::fill(h_table, h_table + capacity, empty);
        cudaMemcpy(table, h_table, capacity * sizeof(KeyValue), 
                  cudaMemcpyHostToDevice);
        delete[] h_table;
    }
    
    ~GPUHashTable() {
        cudaFree(table);
    }
};

// =====================================================
// PART 3: LINEAR PROBING IMPLEMENTATION
// =====================================================

// Insert with linear probing
__global__ void insertLinearProbing(GPUHashTable ht, int* keys, int* values, 
                                   int n, int* success_count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        int key = keys[tid];
        int value = values[tid];
        
        uint32_t slot = ht.hash(key);
        
        // Linear probe until we find empty slot or key
        for (int i = 0; i < ht.capacity; i++) {
            uint32_t current = (slot + i) % ht.capacity;
            
            // Try to insert
            int old_key = atomicCAS(&ht.table[current].key, EMPTY_KEY, key);
            
            if (old_key == EMPTY_KEY || old_key == key) {
                // Successfully claimed slot or found existing key
                ht.table[current].value = value;
                if (old_key == EMPTY_KEY) {
                    atomicAdd(success_count, 1);
                }
                return;
            }
        }
        
        // Table full!
    }
}

// Lookup with linear probing
__global__ void lookupLinearProbing(GPUHashTable ht, int* keys, int* results, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        int key = keys[tid];
        uint32_t slot = ht.hash(key);
        
        // Linear probe
        for (int i = 0; i < ht.capacity; i++) {
            uint32_t current = (slot + i) % ht.capacity;
            int table_key = ht.table[current].key;
            
            if (table_key == key) {
                results[tid] = ht.table[current].value;
                return;
            } else if (table_key == EMPTY_KEY) {
                results[tid] = EMPTY_VALUE;  // Not found
                return;
            }
        }
        
        results[tid] = EMPTY_VALUE;  // Not found (table full)
    }
}

// Delete with tombstones
__global__ void deleteLinearProbing(GPUHashTable ht, int* keys, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        int key = keys[tid];
        uint32_t slot = ht.hash(key);
        
        for (int i = 0; i < ht.capacity; i++) {
            uint32_t current = (slot + i) % ht.capacity;
            
            if (ht.table[current].key == key) {
                // Mark as deleted (tombstone)
                atomicExch(&ht.table[current].key, DELETED_KEY);
                return;
            } else if (ht.table[current].key == EMPTY_KEY) {
                return;  // Not found
            }
        }
    }
}

// =====================================================
// PART 4: CUCKOO HASHING - BETTER PERFORMANCE
// =====================================================

class CuckooHashTable {
public:
    KeyValue* table1;
    KeyValue* table2;
    size_t capacity;
    
    // Two hash functions
    __device__ __host__ inline uint32_t hash1(int key) const {
        return ((uint32_t)key * 2654435761u) % capacity;
    }
    
    __device__ __host__ inline uint32_t hash2(int key) const {
        return ((uint32_t)key * 1103515245u + 12345) % capacity;
    }
    
    CuckooHashTable(size_t cap) : capacity(cap) {
        cudaMalloc(&table1, capacity * sizeof(KeyValue));
        cudaMalloc(&table2, capacity * sizeof(KeyValue));
        
        // Initialize
        KeyValue empty = {EMPTY_KEY, EMPTY_VALUE};
        KeyValue* h_table = new KeyValue[capacity];
        std::fill(h_table, h_table + capacity, empty);
        
        cudaMemcpy(table1, h_table, capacity * sizeof(KeyValue), 
                  cudaMemcpyHostToDevice);
        cudaMemcpy(table2, h_table, capacity * sizeof(KeyValue), 
                  cudaMemcpyHostToDevice);
        delete[] h_table;
    }
    
    ~CuckooHashTable() {
        cudaFree(table1);
        cudaFree(table2);
    }
};

// Cuckoo insert - handles evictions
__global__ void insertCuckoo(CuckooHashTable ht, int* keys, int* values, 
                            int n, int* success_count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        int key = keys[tid];
        int value = values[tid];
        
        // Try up to 32 evictions
        for (int i = 0; i < 32; i++) {
            // Try table 1
            uint32_t slot1 = ht.hash1(key);
            int old_key = atomicCAS(&ht.table1[slot1].key, EMPTY_KEY, key);
            
            if (old_key == EMPTY_KEY || old_key == key) {
                ht.table1[slot1].value = value;
                if (old_key == EMPTY_KEY) atomicAdd(success_count, 1);
                return;
            }
            
            // Evict and try table 2
            int evicted_key = old_key;
            int evicted_value = ht.table1[slot1].value;
            ht.table1[slot1].value = value;
            
            key = evicted_key;
            value = evicted_value;
            
            // Try table 2
            uint32_t slot2 = ht.hash2(key);
            old_key = atomicCAS(&ht.table2[slot2].key, EMPTY_KEY, key);
            
            if (old_key == EMPTY_KEY || old_key == key) {
                ht.table2[slot2].value = value;
                if (old_key == EMPTY_KEY) atomicAdd(success_count, 1);
                return;
            }
            
            // Evict from table 2 and continue
            evicted_key = old_key;
            evicted_value = ht.table2[slot2].value;
            ht.table2[slot2].value = value;
            
            key = evicted_key;
            value = evicted_value;
        }
        
        // Failed after max evictions - need resize
    }
}

// Cuckoo lookup - check both tables
__global__ void lookupCuckoo(CuckooHashTable ht, int* keys, int* results, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        int key = keys[tid];
        
        // Check table 1
        uint32_t slot1 = ht.hash1(key);
        if (ht.table1[slot1].key == key) {
            results[tid] = ht.table1[slot1].value;
            return;
        }
        
        // Check table 2
        uint32_t slot2 = ht.hash2(key);
        if (ht.table2[slot2].key == key) {
            results[tid] = ht.table2[slot2].value;
            return;
        }
        
        results[tid] = EMPTY_VALUE;  // Not found
    }
}

// =====================================================
// PART 5: PERFORMANCE OPTIMIZATIONS
// =====================================================

// Coalesced insert - sort keys first for better locality
__global__ void insertCoalesced(GPUHashTable ht, int* keys, int* values, 
                               int n, int* success_count) {
    __shared__ int shared_keys[256];
    __shared__ int shared_values[256];
    __shared__ int shared_slots[256];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    // Load to shared memory
    if (gid < n) {
        shared_keys[tid] = keys[gid];
        shared_values[tid] = values[gid];
        shared_slots[tid] = ht.hash(shared_keys[tid]);
    }
    __syncthreads();
    
    // Sort by slot for coalesced access
    // (Simple bubble sort for demonstration)
    for (int i = 0; i < blockDim.x; i++) {
        if (tid < blockDim.x - 1 && gid < n - 1) {
            if (shared_slots[tid] > shared_slots[tid + 1]) {
                // Swap
                int temp_key = shared_keys[tid];
                int temp_value = shared_values[tid];
                int temp_slot = shared_slots[tid];
                
                shared_keys[tid] = shared_keys[tid + 1];
                shared_values[tid] = shared_values[tid + 1];
                shared_slots[tid] = shared_slots[tid + 1];
                
                shared_keys[tid + 1] = temp_key;
                shared_values[tid + 1] = temp_value;
                shared_slots[tid + 1] = temp_slot;
            }
        }
        __syncthreads();
    }
    
    // Now insert with better locality
    if (gid < n) {
        int key = shared_keys[tid];
        int value = shared_values[tid];
        uint32_t slot = shared_slots[tid];
        
        // Linear probe
        for (int i = 0; i < ht.capacity; i++) {
            uint32_t current = (slot + i) % ht.capacity;
            int old_key = atomicCAS(&ht.table[current].key, EMPTY_KEY, key);
            
            if (old_key == EMPTY_KEY || old_key == key) {
                ht.table[current].value = value;
                if (old_key == EMPTY_KEY) atomicAdd(success_count, 1);
                return;
            }
        }
    }
}

// =====================================================
// PART 6: APPLICATIONS - JOIN OPERATION
// =====================================================

// Build phase of hash join
__global__ void buildHashTable(GPUHashTable ht, int* build_keys, 
                              int* build_values, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        int key = build_keys[tid];
        int value = build_values[tid];
        uint32_t slot = ht.hash(key);
        
        // Insert into hash table
        for (int i = 0; i < ht.capacity; i++) {
            uint32_t current = (slot + i) % ht.capacity;
            int old_key = atomicCAS(&ht.table[current].key, EMPTY_KEY, key);
            
            if (old_key == EMPTY_KEY || old_key == key) {
                ht.table[current].value = value;
                return;
            }
        }
    }
}

// Probe phase of hash join
__global__ void probeHashTable(GPUHashTable ht, int* probe_keys, 
                              int* probe_values, int* output_keys,
                              int* output_values, int n, int* output_count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        int key = probe_keys[tid];
        uint32_t slot = ht.hash(key);
        
        // Probe hash table
        for (int i = 0; i < ht.capacity; i++) {
            uint32_t current = (slot + i) % ht.capacity;
            
            if (ht.table[current].key == key) {
                // Found match - output join result
                int pos = atomicAdd(output_count, 1);
                output_keys[pos] = key;
                output_values[pos] = ht.table[current].value + probe_values[tid];
                return;
            } else if (ht.table[current].key == EMPTY_KEY) {
                return;  // Not found
            }
        }
    }
}

// =====================================================
// PART 7: MAIN - COMPREHENSIVE TESTING
// =====================================================

void testHashTablePerformance(int n, float load_factor) {
    printf("\n=== Testing with %d elements, load factor %.2f ===\n", 
           n, load_factor);
    
    size_t capacity = n / load_factor;
    
    // Generate test data
    std::vector<int> h_keys(n);
    std::vector<int> h_values(n);
    std::vector<int> h_results(n);
    
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, n * 10);
    
    for (int i = 0; i < n; i++) {
        h_keys[i] = dist(rng);
        h_values[i] = i;
    }
    
    // Allocate GPU memory
    int *d_keys, *d_values, *d_results, *d_success_count;
    cudaMalloc(&d_keys, n * sizeof(int));
    cudaMalloc(&d_values, n * sizeof(int));
    cudaMalloc(&d_results, n * sizeof(int));
    cudaMalloc(&d_success_count, sizeof(int));
    
    cudaMemcpy(d_keys, h_keys.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    // Test 1: Linear Probing
    {
        GPUHashTable ht_linear(capacity);
        cudaMemset(d_success_count, 0, sizeof(int));
        
        Timer insert_timer;
        insertLinearProbing<<<blocks, threads>>>(ht_linear, d_keys, d_values, 
                                               n, d_success_count);
        cudaDeviceSynchronize();
        float insert_time = insert_timer.elapsed();
        
        int success_count;
        cudaMemcpy(&success_count, d_success_count, sizeof(int), 
                  cudaMemcpyDeviceToHost);
        
        Timer lookup_timer;
        lookupLinearProbing<<<blocks, threads>>>(ht_linear, d_keys, d_results, n);
        cudaDeviceSynchronize();
        float lookup_time = lookup_timer.elapsed();
        
        printf("Linear Probing:\n");
        printf("  Insert: %.2f ms (%.2f M ops/sec)\n", 
               insert_time, n / insert_time / 1000);
        printf("  Lookup: %.2f ms (%.2f M ops/sec)\n", 
               lookup_time, n / lookup_time / 1000);
        printf("  Success rate: %.2f%%\n", 100.0f * success_count / n);
    }
    
    // Test 2: Cuckoo Hashing
    {
        CuckooHashTable ht_cuckoo(capacity / 2);  // Two tables
        cudaMemset(d_success_count, 0, sizeof(int));
        
        Timer insert_timer;
        insertCuckoo<<<blocks, threads>>>(ht_cuckoo, d_keys, d_values, 
                                        n, d_success_count);
        cudaDeviceSynchronize();
        float insert_time = insert_timer.elapsed();
        
        int success_count;
        cudaMemcpy(&success_count, d_success_count, sizeof(int), 
                  cudaMemcpyDeviceToHost);
        
        Timer lookup_timer;
        lookupCuckoo<<<blocks, threads>>>(ht_cuckoo, d_keys, d_results, n);
        cudaDeviceSynchronize();
        float lookup_time = lookup_timer.elapsed();
        
        printf("Cuckoo Hashing:\n");
        printf("  Insert: %.2f ms (%.2f M ops/sec)\n", 
               insert_time, n / insert_time / 1000);
        printf("  Lookup: %.2f ms (%.2f M ops/sec)\n", 
               lookup_time, n / lookup_time / 1000);
        printf("  Success rate: %.2f%%\n", 100.0f * success_count / n);
    }
    
    // Cleanup
    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_results);
    cudaFree(d_success_count);
}

// Test hash join
void testHashJoin(int build_size, int probe_size) {
    printf("\n=== Hash Join: %d x %d ===\n", build_size, probe_size);
    
    // Generate data
    std::vector<int> h_build_keys(build_size);
    std::vector<int> h_build_values(build_size);
    std::vector<int> h_probe_keys(probe_size);
    std::vector<int> h_probe_values(probe_size);
    
    // Build table: keys 0 to build_size-1
    for (int i = 0; i < build_size; i++) {
        h_build_keys[i] = i;
        h_build_values[i] = i * 100;
    }
    
    // Probe table: 50% match rate
    std::mt19937 rng(42);
    for (int i = 0; i < probe_size; i++) {
        h_probe_keys[i] = rng() % (build_size * 2);
        h_probe_values[i] = i;
    }
    
    // GPU arrays
    int *d_build_keys, *d_build_values;
    int *d_probe_keys, *d_probe_values;
    int *d_output_keys, *d_output_values, *d_output_count;
    
    cudaMalloc(&d_build_keys, build_size * sizeof(int));
    cudaMalloc(&d_build_values, build_size * sizeof(int));
    cudaMalloc(&d_probe_keys, probe_size * sizeof(int));
    cudaMalloc(&d_probe_values, probe_size * sizeof(int));
    cudaMalloc(&d_output_keys, probe_size * sizeof(int));
    cudaMalloc(&d_output_values, probe_size * sizeof(int));
    cudaMalloc(&d_output_count, sizeof(int));
    
    cudaMemcpy(d_build_keys, h_build_keys.data(), build_size * sizeof(int), 
              cudaMemcpyHostToDevice);
    cudaMemcpy(d_build_values, h_build_values.data(), build_size * sizeof(int), 
              cudaMemcpyHostToDevice);
    cudaMemcpy(d_probe_keys, h_probe_keys.data(), probe_size * sizeof(int), 
              cudaMemcpyHostToDevice);
    cudaMemcpy(d_probe_values, h_probe_values.data(), probe_size * sizeof(int), 
              cudaMemcpyHostToDevice);
    cudaMemset(d_output_count, 0, sizeof(int));
    
    // Create hash table and perform join
    GPUHashTable ht(build_size * 2);  // 50% load factor
    
    Timer timer;
    
    // Build phase
    buildHashTable<<<(build_size + 255) / 256, 256>>>(
        ht, d_build_keys, d_build_values, build_size);
    cudaDeviceSynchronize();
    float build_time = timer.elapsed();
    
    // Probe phase
    Timer probe_timer;
    probeHashTable<<<(probe_size + 255) / 256, 256>>>(
        ht, d_probe_keys, d_probe_values, d_output_keys, 
        d_output_values, probe_size, d_output_count);
    cudaDeviceSynchronize();
    float probe_time = probe_timer.elapsed();
    
    int output_count;
    cudaMemcpy(&output_count, d_output_count, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Build time: %.2f ms\n", build_time);
    printf("Probe time: %.2f ms\n", probe_time);
    printf("Total time: %.2f ms\n", build_time + probe_time);
    printf("Output tuples: %d\n", output_count);
    printf("Throughput: %.2f M tuples/sec\n", 
           (build_size + probe_size) / (build_time + probe_time) / 1000);
    
    // Cleanup
    cudaFree(d_build_keys);
    cudaFree(d_build_values);
    cudaFree(d_probe_keys);
    cudaFree(d_probe_values);
    cudaFree(d_output_keys);
    cudaFree(d_output_values);
    cudaFree(d_output_count);
}

int main() {
    printf("==================================================\n");
    printf("GPU HASH TABLE\n");
    printf("==================================================\n");
    
    // Test different sizes and load factors
    testHashTablePerformance(1000000, 0.5f);   // 50% load
    testHashTablePerformance(1000000, 0.7f);   // 70% load
    testHashTablePerformance(1000000, 0.9f);   // 90% load
    
    // Test hash join
    testHashJoin(1000000, 2000000);
    
    printf("\n==================================================\n");
    printf("KEY INSIGHTS\n");
    printf("==================================================\n");
    printf("1. Linear probing simple but suffers from clustering\n");
    printf("2. Cuckoo hashing has constant lookup time\n");
    printf("3. Load factor critically affects performance\n");
    printf("4. Coalesced memory access improves throughput\n");
    printf("5. Hash joins can process millions of tuples/sec\n");
    
    return 0;
}

/*
 * COMPREHENSIVE EXERCISES:
 *
 * === Understanding ===
 * 1. Why does linear probing cause clustering?
 * 2. Calculate theoretical memory bandwidth utilization
 * 3. How does warp divergence affect hash table ops?
 * 4. Compare with CPU hash table performance
 * 5. When to use hash table vs sorted array?
 *
 * === Implementation ===
 * 6. Add Robin Hood hashing (minimize variance)
 * 7. Implement dynamic resizing
 * 8. Create chained hash table variant
 * 9. Add support for variable-length keys
 * 10. Build concurrent hash set (unique values)
 *
 * === Optimization ===
 * 11. Use warp-cooperative insertion
 * 12. Implement lock-free deletion
 * 13. Add bloom filter for faster misses
 * 14. Create SIMD-friendly hash functions
 * 15. Optimize for skewed key distributions
 *
 * === Advanced Features ===
 * 16. Build persistent hash table (save to disk)
 * 17. Implement consistent hashing
 * 18. Add support for transactions
 * 19. Create distributed GPU hash table
 * 20. Build time-decay hash table (LRU-like)
 *
 * === Applications ===
 * 21. Implement GROUP BY aggregation
 * 22. Build full SQL-style hash join
 * 23. Create GPU-accelerated cache
 * 24. Implement duplicate detection system
 * 25. Build real-time analytics engine
 */

/*
 * MENTAL MODELS:
 *
 * 1. "Parking Lot"
 *    - Cars (keys) looking for spots (slots)
 *    - Linear probing: Check next spot
 *    - Cuckoo: Two preferred spots
 *    - Collision: Multiple cars want same spot
 *
 * 2. "Lock-Free Dance"
 *    - Threads coordinate without explicit locks
 *    - CAS operations for atomic updates
 *    - Retry on conflict
 *
 * 3. Performance Factors:
 *    - Load factor: How full is the table?
 *    - Probe distance: How far to search?
 *    - Memory access: Coalesced vs random
 *    - Synchronization: Atomic operation cost
 */
