/*
 * PROJECT 7: GPU REGULAR EXPRESSIONS
 * Text Processing at Scale
 *
 * Process millions of strings per second with parallel regex matching.
 * Perfect for log analysis, data validation, and text mining.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <string>
#include <chrono>
#include <cstring>

// =====================================================
// PART 1: FIRST PRINCIPLES - Parallel Pattern Matching
// =====================================================

/*
 * REGEX ON GPU CHALLENGES:
 * 
 * Traditional regex:
 * - Sequential state machines
 * - Backtracking
 * - Complex control flow
 * 
 * GPU-friendly approach:
 * - Convert to DFA (deterministic finite automaton)
 * - Process multiple strings in parallel
 * - Avoid divergence with careful design
 * 
 * Applications:
 * - Log file analysis (millions of lines)
 * - Data validation at scale
 * - Network packet inspection
 * - Bioinformatics pattern matching
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
// PART 2: DFA REPRESENTATION
// =====================================================

// DFA state transition
struct DFATransition {
    int from_state;
    char symbol;
    int to_state;
};

// DFA structure for GPU
struct DFA {
    int* transition_table;  // [state][symbol] -> next_state
    bool* accept_states;    // [state] -> is_accepting
    int num_states;
    int alphabet_size;
    int start_state;
    
    __device__ int transition(int state, unsigned char symbol) const {
        if (symbol >= alphabet_size) return -1;  // Invalid transition
        return transition_table[state * alphabet_size + symbol];
    }
    
    __device__ bool is_accepting(int state) const {
        return accept_states[state];
    }
};

// =====================================================
// PART 3: SIMPLE PATTERN MATCHERS
// =====================================================

// Exact string match (parallel Boyer-Moore style)
__global__ void exactMatch(bool* results, const char* texts, 
                          int num_texts, int text_length,
                          const char* pattern, int pattern_length) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_texts) {
        const char* text = texts + tid * text_length;
        bool found = false;
        
        // Simple search (can be optimized with better algorithms)
        for (int i = 0; i <= text_length - pattern_length; i++) {
            bool match = true;
            
            // Check pattern at position i
            for (int j = 0; j < pattern_length; j++) {
                if (text[i + j] != pattern[j]) {
                    match = false;
                    break;
                }
            }
            
            if (match) {
                found = true;
                break;
            }
        }
        
        results[tid] = found;
    }
}

// Character class matching (e.g., [a-z], [0-9])
__device__ bool matchCharClass(char c, const char* class_def) {
    // Simple implementation for demonstration
    if (class_def[0] == 'a' && class_def[1] == '-' && class_def[2] == 'z') {
        return c >= 'a' && c <= 'z';
    } else if (class_def[0] == '0' && class_def[1] == '-' && class_def[2] == '9') {
        return c >= '0' && c <= '9';
    }
    return false;
}

// =====================================================
// PART 4: DFA-BASED REGEX MATCHING
// =====================================================

// Run DFA on a single string
__device__ bool runDFA(const DFA& dfa, const char* text, int length) {
    int state = dfa.start_state;
    
    for (int i = 0; i < length && text[i] != '\0'; i++) {
        state = dfa.transition(state, (unsigned char)text[i]);
        if (state == -1) {
            return false;  // No valid transition
        }
    }
    
    return dfa.is_accepting(state);
}

// Parallel DFA matching
__global__ void dfaMatch(bool* results, const char* texts,
                        int num_texts, int text_length,
                        DFA dfa) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_texts) {
        const char* text = texts + tid * text_length;
        results[tid] = runDFA(dfa, text, text_length);
    }
}

// =====================================================
// PART 5: ADVANCED - MULTI-PATTERN MATCHING
// =====================================================

// Aho-Corasick style multi-pattern matching
struct ACNode {
    int next[256];      // Transitions
    int failure;        // Failure link
    int output;         // Pattern ID if match (-1 if none)
};

__global__ void multiPatternMatch(int* results, const char* texts,
                                 int num_texts, int text_length,
                                 ACNode* trie, int num_patterns) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_texts) {
        const char* text = texts + tid * text_length;
        int state = 0;  // Root
        results[tid] = -1;  // No match
        
        for (int i = 0; i < text_length && text[i] != '\0'; i++) {
            unsigned char c = text[i];
            
            // Follow transitions
            while (state != 0 && trie[state].next[c] == -1) {
                state = trie[state].failure;
            }
            
            if (trie[state].next[c] != -1) {
                state = trie[state].next[c];
            }
            
            // Check for matches
            if (trie[state].output != -1) {
                results[tid] = trie[state].output;
                return;  // Found a match
            }
        }
    }
}

// =====================================================
// PART 6: OPTIMIZATIONS
// =====================================================

// Warp-collaborative pattern matching
__global__ void warpPatternMatch(bool* results, const char* texts,
                                int num_texts, int text_length,
                                const char* pattern, int pattern_length) {
    // Each warp processes one text
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    
    if (warp_id < num_texts) {
        const char* text = texts + warp_id * text_length;
        bool found = false;
        
        // Each thread checks different positions
        for (int i = lane_id; i <= text_length - pattern_length; i += 32) {
            bool match = true;
            
            for (int j = 0; j < pattern_length; j++) {
                if (text[i + j] != pattern[j]) {
                    match = false;
                    break;
                }
            }
            
            if (match) {
                found = true;
            }
        }
        
        // Reduce across warp
        found = __any_sync(0xffffffff, found);
        
        if (lane_id == 0) {
            results[warp_id] = found;
        }
    }
}

// =====================================================
// PART 7: REAL-WORLD PATTERNS
// =====================================================

// Email validation regex (simplified DFA)
DFA createEmailDFA() {
    // States: 0=start, 1=user, 2=@, 3=domain, 4=dot, 5=tld, 6=accept
    const int num_states = 7;
    const int alphabet_size = 256;
    
    DFA dfa;
    dfa.num_states = num_states;
    dfa.alphabet_size = alphabet_size;
    dfa.start_state = 0;
    
    // Allocate and initialize
    cudaMalloc(&dfa.transition_table, num_states * alphabet_size * sizeof(int));
    cudaMalloc(&dfa.accept_states, num_states * sizeof(bool));
    
    // Initialize transitions (simplified)
    int* h_transitions = new int[num_states * alphabet_size];
    bool* h_accept = new bool[num_states];
    
    // Fill with -1 (no transition)
    std::fill(h_transitions, h_transitions + num_states * alphabet_size, -1);
    std::fill(h_accept, h_accept + num_states, false);
    
    // Define transitions (very simplified email pattern)
    // State 0 -> State 1 on alphanumeric
    for (char c = 'a'; c <= 'z'; c++) {
        h_transitions[0 * alphabet_size + c] = 1;
        h_transitions[0 * alphabet_size + toupper(c)] = 1;
    }
    for (char c = '0'; c <= '9'; c++) {
        h_transitions[0 * alphabet_size + c] = 1;
    }
    
    // State 1 -> State 1 on alphanumeric, State 2 on @
    for (char c = 'a'; c <= 'z'; c++) {
        h_transitions[1 * alphabet_size + c] = 1;
        h_transitions[1 * alphabet_size + toupper(c)] = 1;
    }
    for (char c = '0'; c <= '9'; c++) {
        h_transitions[1 * alphabet_size + c] = 1;
    }
    h_transitions[1 * alphabet_size + '@'] = 2;
    
    // Continue building DFA...
    h_accept[6] = true;  // Accept state
    
    // Copy to device
    cudaMemcpy(dfa.transition_table, h_transitions, 
              num_states * alphabet_size * sizeof(int), 
              cudaMemcpyHostToDevice);
    cudaMemcpy(dfa.accept_states, h_accept, 
              num_states * sizeof(bool), 
              cudaMemcpyHostToDevice);
    
    delete[] h_transitions;
    delete[] h_accept;
    
    return dfa;
}

// Log parsing patterns
__global__ void parseLogLevel(int* levels, const char* logs,
                             int num_logs, int log_length) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_logs) {
        const char* log = logs + tid * log_length;
        
        // Look for common log level patterns
        if (strncmp(log, "[ERROR]", 7) == 0) {
            levels[tid] = 3;  // ERROR
        } else if (strncmp(log, "[WARN]", 6) == 0) {
            levels[tid] = 2;  // WARNING
        } else if (strncmp(log, "[INFO]", 6) == 0) {
            levels[tid] = 1;  // INFO
        } else if (strncmp(log, "[DEBUG]", 7) == 0) {
            levels[tid] = 0;  // DEBUG
        } else {
            levels[tid] = -1; // Unknown
        }
    }
}

// =====================================================
// PART 8: PERFORMANCE COMPARISON
// =====================================================

void comparePerformance(int num_strings, int string_length) {
    printf("\n=== Performance Comparison ===\n");
    printf("Processing %d strings of length %d\n", num_strings, string_length);
    
    // Generate test data
    std::vector<std::string> test_strings;
    for (int i = 0; i < num_strings; i++) {
        std::string s;
        for (int j = 0; j < string_length - 1; j++) {
            s += 'a' + (rand() % 26);
        }
        test_strings.push_back(s);
    }
    
    // Flatten for GPU
    char* h_texts = new char[num_strings * string_length];
    for (int i = 0; i < num_strings; i++) {
        strncpy(h_texts + i * string_length, 
                test_strings[i].c_str(), string_length);
    }
    
    // Allocate GPU memory
    char* d_texts;
    bool* d_results;
    cudaMalloc(&d_texts, num_strings * string_length);
    cudaMalloc(&d_results, num_strings * sizeof(bool));
    
    cudaMemcpy(d_texts, h_texts, num_strings * string_length, 
              cudaMemcpyHostToDevice);
    
    // Test pattern
    const char* pattern = "hello";
    char* d_pattern;
    cudaMalloc(&d_pattern, strlen(pattern) + 1);
    cudaMemcpy(d_pattern, pattern, strlen(pattern) + 1, 
              cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (num_strings + threads - 1) / threads;
    
    // Test 1: Simple exact match
    Timer timer1;
    exactMatch<<<blocks, threads>>>(d_results, d_texts, num_strings, 
                                  string_length, d_pattern, strlen(pattern));
    cudaDeviceSynchronize();
    float time1 = timer1.elapsed();
    printf("Exact match: %.2f ms (%.2f M strings/sec)\n", 
           time1, num_strings / time1 / 1000);
    
    // Test 2: Warp-collaborative
    int warp_blocks = (num_strings + 7) / 8;  // 8 warps per block
    Timer timer2;
    warpPatternMatch<<<warp_blocks, 256>>>(d_results, d_texts, num_strings,
                                          string_length, d_pattern, strlen(pattern));
    cudaDeviceSynchronize();
    float time2 = timer2.elapsed();
    printf("Warp collaborative: %.2f ms (%.2f M strings/sec)\n",
           time2, num_strings / time2 / 1000);
    
    // Cleanup
    delete[] h_texts;
    cudaFree(d_texts);
    cudaFree(d_results);
    cudaFree(d_pattern);
}

// =====================================================
// PART 9: MAIN - COMPREHENSIVE DEMO
// =====================================================

int main() {
    printf("==================================================\n");
    printf("GPU REGULAR EXPRESSIONS\n");
    printf("==================================================\n");
    
    // Test basic pattern matching
    printf("\n=== Basic Pattern Matching ===\n");
    
    const int num_test = 10;
    const int max_length = 100;
    
    std::vector<std::string> test_texts = {
        "The quick brown fox jumps over the lazy dog",
        "Hello, World!",
        "GPU regex matching is fast",
        "Pattern matching at scale",
        "CUDA makes text processing parallel",
        "Regular expressions on GPU",
        "Massive parallelism for string processing",
        "Log analysis with GPU acceleration",
        "Data validation at high throughput",
        "Text mining with CUDA"
    };
    
    // Prepare data
    char* h_texts = new char[num_test * max_length]();
    for (int i = 0; i < num_test; i++) {
        strncpy(h_texts + i * max_length, test_texts[i].c_str(), max_length - 1);
    }
    
    char* d_texts;
    bool* d_results;
    bool* h_results = new bool[num_test];
    
    cudaMalloc(&d_texts, num_test * max_length);
    cudaMalloc(&d_results, num_test * sizeof(bool));
    cudaMemcpy(d_texts, h_texts, num_test * max_length, cudaMemcpyHostToDevice);
    
    // Test different patterns
    std::vector<std::string> patterns = {"GPU", "regex", "the"};
    
    for (const auto& pattern : patterns) {
        char* d_pattern;
        cudaMalloc(&d_pattern, pattern.length() + 1);
        cudaMemcpy(d_pattern, pattern.c_str(), pattern.length() + 1, 
                  cudaMemcpyHostToDevice);
        
        exactMatch<<<1, 32>>>(d_results, d_texts, num_test, max_length,
                            d_pattern, pattern.length());
        
        cudaMemcpy(h_results, d_results, num_test * sizeof(bool), 
                  cudaMemcpyDeviceToHost);
        
        printf("\nPattern '%s' found in:\n", pattern.c_str());
        for (int i = 0; i < num_test; i++) {
            if (h_results[i]) {
                printf("  - %s\n", test_texts[i].c_str());
            }
        }
        
        cudaFree(d_pattern);
    }
    
    // Performance tests
    comparePerformance(100000, 100);
    comparePerformance(1000000, 50);
    
    // Log parsing demo
    printf("\n=== Log Parsing Demo ===\n");
    
    std::vector<std::string> log_lines = {
        "[ERROR] Failed to connect to database",
        "[INFO] Application started successfully",
        "[DEBUG] Processing request ID: 12345",
        "[WARN] Memory usage above 80%",
        "[ERROR] Invalid user credentials"
    };
    
    char* h_logs = new char[log_lines.size() * max_length]();
    int* h_levels = new int[log_lines.size()];
    
    for (size_t i = 0; i < log_lines.size(); i++) {
        strncpy(h_logs + i * max_length, log_lines[i].c_str(), max_length - 1);
    }
    
    char* d_logs;
    int* d_levels;
    cudaMalloc(&d_logs, log_lines.size() * max_length);
    cudaMalloc(&d_levels, log_lines.size() * sizeof(int));
    
    cudaMemcpy(d_logs, h_logs, log_lines.size() * max_length, 
              cudaMemcpyHostToDevice);
    
    parseLogLevel<<<1, 32>>>(d_levels, d_logs, log_lines.size(), max_length);
    
    cudaMemcpy(h_levels, d_levels, log_lines.size() * sizeof(int), 
              cudaMemcpyDeviceToHost);
    
    const char* level_names[] = {"DEBUG", "INFO", "WARN", "ERROR"};
    for (size_t i = 0; i < log_lines.size(); i++) {
        printf("Log: %s\n", log_lines[i].c_str());
        if (h_levels[i] >= 0) {
            printf("  Level: %s\n", level_names[h_levels[i]]);
        }
    }
    
    // Cleanup
    delete[] h_texts;
    delete[] h_results;
    delete[] h_logs;
    delete[] h_levels;
    cudaFree(d_texts);
    cudaFree(d_results);
    cudaFree(d_logs);
    cudaFree(d_levels);
    
    printf("\n==================================================\n");
    printf("KEY INSIGHTS\n");
    printf("==================================================\n");
    printf("1. DFA-based matching avoids backtracking\n");
    printf("2. Process millions of strings in parallel\n");
    printf("3. Warp collaboration improves efficiency\n");
    printf("4. Perfect for log analysis and validation\n");
    printf("5. Complex patterns need careful DFA design\n");
    printf("6. Memory coalescing crucial for performance\n");
    
    return 0;
}

/*
 * COMPREHENSIVE EXERCISES:
 *
 * === Understanding ===
 * 1. Why is DFA better than NFA for GPU?
 * 2. Calculate memory bandwidth for pattern matching
 * 3. How does string length affect performance?
 * 4. When to use exact vs regex matching?
 * 5. Compare with CPU regex libraries
 *
 * === Implementation ===
 * 6. Build NFA to DFA converter
 * 7. Implement wildcard matching (*, ?)
 * 8. Create Unicode support
 * 9. Add capture groups
 * 10. Build regex compiler
 *
 * === Optimization ===
 * 11. Implement bit-parallel algorithms
 * 12. Add string preprocessing (Boyer-Moore)
 * 13. Create SIMD character classes
 * 14. Optimize for different alphabets
 * 15. Build hybrid CPU-GPU matcher
 *
 * === Advanced ===
 * 16. Implement PCRE subset
 * 17. Create streaming regex engine
 * 18. Build approximate matching
 * 19. Add lookahead/lookbehind
 * 20. Implement regex JIT compiler
 *
 * === Applications ===
 * 21. Web server log analyzer
 * 22. Network intrusion detection
 * 23. DNA sequence matching
 * 24. Data validation pipeline
 * 25. Real-time stream processing
 */

/*
 * MENTAL MODELS:
 *
 * 1. "Assembly Line"
 *    - Each GPU thread = one string
 *    - All process same pattern
 *    - Massive throughput
 *
 * 2. "State Machine Highway"
 *    - DFA = road map
 *    - Characters = directions
 *    - Parallel drivers (threads)
 *
 * 3. "Pattern Broadcasting"
 *    - One pattern, many texts
 *    - Broadcast pattern to all
 *    - Parallel matching
 *
 * 4. GPU Regex Reality:
 *    - Simple patterns: huge wins
 *    - Complex patterns: need care
 *    - Batch processing: always wins
 *    - Real-time: consider latency
 */
