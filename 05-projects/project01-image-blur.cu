/**
 * Project 1: Building a Real-World GPU Application - Image Blur
 * ==============================================================
 *
 * The Journey So Far:
 * ------------------
 * Week 1: You learned kernels, threads, memory
 * Week 2: You mastered shared memory
 * Week 3: You optimized algorithms
 * Week 4: You handled streams and advanced features
 * 
 * Now: Build something REAL that combines it all!
 *
 * The Challenge:
 * -------------
 * Process a 4K image (3840×2160 pixels) with Gaussian blur in real-time.
 * That's 8.3 million pixels, 25 million values (RGB), 30 times per second.
 * Can your GPU handle 750 million operations per second?
 *
 * What We'll Build:
 * ----------------
 * 1. Understand image processing fundamentals
 * 2. Implement naive GPU blur (often slower than CPU!)
 * 3. Add shared memory optimization (10x faster)
 * 4. Handle edge cases properly
 * 5. Measure and analyze performance
 * 6. Create production-quality code
 *
 * Real-World Applications:
 * -----------------------
 * - Instagram filters
 * - Video game post-processing
 * - Medical image denoising
 * - Computer vision preprocessing
 * - Real-time video effects
 *
 * Compile: nvcc -O3 -arch=sm_86 -o blur project01-image-blur.cu -lm
 * Run: ./blur [image.ppm]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

/**
 * FIRST PRINCIPLES: What is Image Blur?
 * ------------------------------------
 * 
 * Blurring = Weighted average of neighboring pixels
 * 
 * Original:        Blurred:
 * ┌─┬─┬─┐         ┌─┬─┬─┐
 * │1│9│2│         │3│4│3│
 * ├─┼─┼─┤   -->   ├─┼─┼─┤
 * │3│5│1│         │4│5│4│
 * ├─┼─┼─┤         ├─┼─┼─┤
 * │2│8│3│         │3│4│3│
 * └─┴─┴─┘         └─┴─┴─┘
 * 
 * Each output pixel = weighted sum of input neighborhood
 * 
 * Gaussian Blur: Weights follow Gaussian (bell curve) distribution
 * - Center pixel: highest weight
 * - Further pixels: exponentially less weight
 * - Preserves edges better than box blur
 */

// Configuration
#define TILE_WIDTH 16
#define FILTER_RADIUS 4
#define FILTER_SIZE (2 * FILTER_RADIUS + 1)  // 9x9 filter

/**
 * STEP 1: Generate Gaussian Filter
 * --------------------------------
 * Instead of hard-coding, let's understand the math
 */
void generateGaussianKernel(float *kernel, int size, float sigma) {
    int radius = size / 2;
    float sum = 0.0f;
    
    printf("Generating %dx%d Gaussian kernel (σ=%.1f):\n", size, size, sigma);
    
    // Gaussian formula: G(x,y) = (1/2πσ²) * e^(-(x²+y²)/2σ²)
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            float value = expf(-(x*x + y*y) / (2.0f * sigma * sigma));
            kernel[(y + radius) * size + (x + radius)] = value;
            sum += value;
        }
    }
    
    // Normalize so sum = 1.0
    for (int i = 0; i < size * size; i++) {
        kernel[i] /= sum;
    }
    
    // Display kernel (center row)
    printf("Center row: ");
    for (int x = 0; x < size; x++) {
        printf("%.4f ", kernel[radius * size + x]);
    }
    printf("\n\n");
}

// Store filter in constant memory for fast access
__constant__ float c_filter[FILTER_SIZE * FILTER_SIZE];

/**
 * Image structure for easier handling
 */
struct Image {
    unsigned char *data;
    int width, height, channels;
    size_t size;
    
    void allocate(int w, int h, int c) {
        width = w;
        height = h;
        channels = c;
        size = w * h * c * sizeof(unsigned char);
        data = (unsigned char*)malloc(size);
    }
    
    void free() {
        ::free(data);
    }
    
    unsigned char& at(int x, int y, int c) {
        return data[(y * width + x) * channels + c];
    }
};

/**
 * STEP 2: CPU Baseline Implementation
 * -----------------------------------
 * This is our reference for correctness and performance
 */
void blurCPU(Image &input, Image &output, float *filter, int filterSize) {
    int radius = filterSize / 2;
    
    // For each output pixel
    for (int y = 0; y < input.height; y++) {
        for (int x = 0; x < input.width; x++) {
            // For each color channel
            for (int c = 0; c < input.channels; c++) {
                float sum = 0.0f;
                
                // Apply filter
                for (int fy = -radius; fy <= radius; fy++) {
                    for (int fx = -radius; fx <= radius; fx++) {
                        // Handle boundaries with clamping
                        int px = min(max(x + fx, 0), input.width - 1);
                        int py = min(max(y + fy, 0), input.height - 1);
                        
                        float filterVal = filter[(fy + radius) * filterSize + (fx + radius)];
                        sum += input.at(px, py, c) * filterVal;
                    }
                }
                
                output.at(x, y, c) = (unsigned char)min(max(sum, 0.0f), 255.0f);
            }
        }
    }
}

/**
 * STEP 3: Naive GPU Implementation
 * --------------------------------
 * Direct port of CPU code - one thread per pixel
 */
__global__ void blurGPU_Naive(unsigned char *input, unsigned char *output,
                              int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int radius = FILTER_RADIUS;
    
    // Process each channel
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        // Apply filter
        for (int fy = -radius; fy <= radius; fy++) {
            for (int fx = -radius; fx <= radius; fx++) {
                // Clamp to image boundaries
                int px = min(max(x + fx, 0), width - 1);
                int py = min(max(y + fy, 0), height - 1);
                
                // Global memory access (SLOW!)
                int pixelIdx = (py * width + px) * channels + c;
                float filterVal = c_filter[(fy + radius) * FILTER_SIZE + (fx + radius)];
                
                sum += input[pixelIdx] * filterVal;
            }
        }
        
        // Write result
        int outputIdx = (y * width + x) * channels + c;
        output[outputIdx] = (unsigned char)min(max(sum, 0.0f), 255.0f);
    }
}

/**
 * STEP 4: Shared Memory Optimization
 * ----------------------------------
 * Key insight: Neighboring threads access overlapping pixels
 * Solution: Load tile to shared memory, reuse for all threads
 */
__global__ void blurGPU_Shared(unsigned char *input, unsigned char *output,
                              int width, int height, int channels) {
    // Shared memory tile (includes halo for filter)
    const int TILE_SIZE = TILE_WIDTH + 2 * FILTER_RADIUS;
    __shared__ float tile[TILE_SIZE][TILE_SIZE][3];  // Assuming RGB
    
    // Global thread position
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Local thread position
    int lx = threadIdx.x;
    int ly = threadIdx.y;
    
    // Load tile including halo region
    // This is complex but crucial for performance!
    int radius = FILTER_RADIUS;
    
    // Each thread loads multiple pixels to fill the tile
    for (int dy = ly; dy < TILE_SIZE; dy += blockDim.y) {
        for (int dx = lx; dx < TILE_SIZE; dx += blockDim.x) {
            // Calculate source pixel
            int sx = blockIdx.x * blockDim.x - radius + dx;
            int sy = blockIdx.y * blockDim.y - radius + dy;
            
            // Clamp to image bounds
            sx = min(max(sx, 0), width - 1);
            sy = min(max(sy, 0), height - 1);
            
            // Load all channels
            for (int c = 0; c < channels && c < 3; c++) {
                int idx = (sy * width + sx) * channels + c;
                tile[dy][dx][c] = input[idx];
            }
        }
    }
    
    __syncthreads();  // Wait for tile to be loaded
    
    // Only process if within image bounds
    if (gx >= width || gy >= height) return;
    
    // Now compute blur using shared memory
    for (int c = 0; c < channels && c < 3; c++) {
        float sum = 0.0f;
        
        // Apply filter using shared memory (FAST!)
        for (int fy = -radius; fy <= radius; fy++) {
            for (int fx = -radius; fx <= radius; fx++) {
                int tx = lx + radius + fx;
                int ty = ly + radius + fy;
                
                float filterVal = c_filter[(fy + radius) * FILTER_SIZE + (fx + radius)];
                sum += tile[ty][tx][c] * filterVal;
            }
        }
        
        // Write result
        int outputIdx = (gy * width + gx) * channels + c;
        output[outputIdx] = (unsigned char)min(max(sum, 0.0f), 255.0f);
    }
}

/**
 * STEP 5: Separable Filter Optimization
 * -------------------------------------
 * Gaussian blur is separable: 2D filter = 1D horizontal * 1D vertical
 * This reduces complexity from O(r²) to O(2r)
 */
__global__ void blurGPU_Horizontal(unsigned char *input, float *temp,
                                  int width, int height, int channels) {
    // Implementation left as exercise
    // Hint: Process rows with 1D filter
}

__global__ void blurGPU_Vertical(float *temp, unsigned char *output,
                                int width, int height, int channels) {
    // Implementation left as exercise
    // Hint: Process columns with 1D filter
}

/**
 * Timer class for accurate measurements
 */
class Timer {
    cudaEvent_t start, stop;
public:
    Timer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~Timer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void startTimer() {
        cudaEventRecord(start, 0);
    }
    float stopTimer() {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float time;
        cudaEventElapsedTime(&time, start, stop);
        return time;
    }
};

/**
 * PPM Image I/O (Simple format for teaching)
 */
bool loadPPM(const char* filename, Image &img) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) return false;
    
    char header[3];
    fscanf(fp, "%s", header);
    if (strcmp(header, "P6") != 0) {
        fclose(fp);
        return false;
    }
    
    int width, height, maxval;
    fscanf(fp, "%d %d %d", &width, &height, &maxval);
    fgetc(fp); // Skip newline
    
    img.allocate(width, height, 3);
    fread(img.data, 1, img.size, fp);
    fclose(fp);
    
    return true;
}

bool savePPM(const char* filename, Image &img) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) return false;
    
    fprintf(fp, "P6\n%d %d\n255\n", img.width, img.height);
    fwrite(img.data, 1, img.size, fp);
    fclose(fp);
    
    return true;
}

/**
 * Generate test image if no input provided
 */
void generateTestImage(Image &img, int width, int height) {
    img.allocate(width, height, 3);
    
    // Create gradient with some shapes
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Gradient background
            img.at(x, y, 0) = (x * 255) / width;        // R
            img.at(x, y, 1) = (y * 255) / height;       // G
            img.at(x, y, 2) = ((x + y) * 255) / (width + height); // B
            
            // Add some circles
            int cx = width / 2;
            int cy = height / 2;
            int dist = sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy));
            if (dist < 100) {
                img.at(x, y, 0) = 255;
                img.at(x, y, 1) = 255;
                img.at(x, y, 2) = 255;
            }
        }
    }
}

/**
 * Main Performance Test
 */
int main(int argc, char **argv) {
    printf("===========================================================\n");
    printf("PROJECT 1: Real-World GPU Application - Image Blur\n");
    printf("===========================================================\n\n");
    
    // Check GPU
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SMs: %d, Max threads/block: %d\n\n", 
           prop.multiProcessorCount, prop.maxThreadsPerBlock);
    
    // Generate Gaussian filter
    float *h_filter = new float[FILTER_SIZE * FILTER_SIZE];
    generateGaussianKernel(h_filter, FILTER_SIZE, 1.5f);
    
    // Copy filter to constant memory
    cudaMemcpyToSymbol(c_filter, h_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float));
    
    // Load or generate image
    Image input, output_cpu, output_gpu;
    
    if (argc > 1) {
        if (!loadPPM(argv[1], input)) {
            printf("Failed to load %s\n", argv[1]);
            return 1;
        }
        printf("Loaded image: %dx%d\n", input.width, input.height);
    } else {
        int width = 1920;   // Full HD
        int height = 1080;
        generateTestImage(input, width, height);
        printf("Generated test image: %dx%d\n", width, height);
        savePPM("input.ppm", input);
    }
    
    // Allocate output images
    output_cpu.allocate(input.width, input.height, input.channels);
    output_gpu.allocate(input.width, input.height, input.channels);
    
    printf("\nImage specs:\n");
    printf("- Resolution: %dx%d\n", input.width, input.height);
    printf("- Pixels: %.2f million\n", (input.width * input.height) / 1e6);
    printf("- Data size: %.2f MB\n\n", input.size / (1024.0 * 1024.0));
    
    // ===============================
    // CPU Baseline
    // ===============================
    printf("CPU IMPLEMENTATION\n");
    printf("------------------\n");
    
    clock_t cpu_start = clock();
    blurCPU(input, output_cpu, h_filter, FILTER_SIZE);
    clock_t cpu_end = clock();
    
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0;
    printf("Time: %.2f ms\n", cpu_time);
    printf("Throughput: %.2f MPixels/s\n\n", 
           (input.width * input.height) / (cpu_time * 1000.0));
    
    // Save CPU result
    savePPM("output_cpu.ppm", output_cpu);
    
    // ===============================
    // GPU Setup
    // ===============================
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, input.size);
    cudaMalloc(&d_output, input.size);
    cudaMemcpy(d_input, input.data, input.size, cudaMemcpyHostToDevice);
    
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize(
        (input.width + blockSize.x - 1) / blockSize.x,
        (input.height + blockSize.y - 1) / blockSize.y
    );
    
    printf("GPU CONFIGURATION\n");
    printf("-----------------\n");
    printf("Block size: %dx%d = %d threads\n", 
           blockSize.x, blockSize.y, blockSize.x * blockSize.y);
    printf("Grid size: %dx%d = %d blocks\n", 
           gridSize.x, gridSize.y, gridSize.x * gridSize.y);
    printf("Total threads: %d\n\n", 
           gridSize.x * gridSize.y * blockSize.x * blockSize.y);
    
    Timer timer;
    
    // ===============================
    // GPU Naive Implementation
    // ===============================
    printf("GPU NAIVE IMPLEMENTATION\n");
    printf("------------------------\n");
    
    // Warmup
    blurGPU_Naive<<<gridSize, blockSize>>>(d_input, d_output, 
                                           input.width, input.height, input.channels);
    cudaDeviceSynchronize();
    
    // Measure
    timer.startTimer();
    blurGPU_Naive<<<gridSize, blockSize>>>(d_input, d_output,
                                           input.width, input.height, input.channels);
    float gpu_naive_time = timer.stopTimer();
    
    printf("Kernel time: %.2f ms\n", gpu_naive_time);
    printf("Speedup vs CPU: %.2fx\n", cpu_time / gpu_naive_time);
    printf("Throughput: %.2f MPixels/s\n\n",
           (input.width * input.height) / (gpu_naive_time * 1000.0));
    
    // Copy result and verify
    cudaMemcpy(output_gpu.data, d_output, input.size, cudaMemcpyDeviceToHost);
    savePPM("output_gpu_naive.ppm", output_gpu);
    
    // ===============================
    // GPU Shared Memory Implementation
    // ===============================
    printf("GPU SHARED MEMORY IMPLEMENTATION\n");
    printf("--------------------------------\n");
    
    // Warmup
    blurGPU_Shared<<<gridSize, blockSize>>>(d_input, d_output,
                                            input.width, input.height, input.channels);
    cudaDeviceSynchronize();
    
    // Measure
    timer.startTimer();
    blurGPU_Shared<<<gridSize, blockSize>>>(d_input, d_output,
                                            input.width, input.height, input.channels);
    float gpu_shared_time = timer.stopTimer();
    
    printf("Kernel time: %.2f ms\n", gpu_shared_time);
    printf("Speedup vs CPU: %.2fx\n", cpu_time / gpu_shared_time);
    printf("Speedup vs Naive GPU: %.2fx\n", gpu_naive_time / gpu_shared_time);
    printf("Throughput: %.2f MPixels/s\n\n",
           (input.width * input.height) / (gpu_shared_time * 1000.0));
    
    // Copy result
    cudaMemcpy(output_gpu.data, d_output, input.size, cudaMemcpyDeviceToHost);
    savePPM("output_gpu_shared.ppm", output_gpu);
    
    // ===============================
    // Performance Analysis
    // ===============================
    printf("PERFORMANCE ANALYSIS\n");
    printf("===================\n\n");
    
    // Calculate theoretical limits
    float ops_per_pixel = FILTER_SIZE * FILTER_SIZE * 2 * input.channels; // mul + add
    float total_gflops = (input.width * input.height * ops_per_pixel) / 1e9;
    
    printf("Computational Analysis:\n");
    printf("- Filter size: %dx%d\n", FILTER_SIZE, FILTER_SIZE);
    printf("- Operations per pixel: %.0f\n", ops_per_pixel);
    printf("- Total operations: %.2f GFLOP\n", total_gflops);
    printf("\nAchieved Performance:\n");
    printf("- CPU: %.2f GFLOPS\n", total_gflops / (cpu_time / 1000.0));
    printf("- GPU Naive: %.2f GFLOPS\n", total_gflops / (gpu_naive_time / 1000.0));
    printf("- GPU Shared: %.2f GFLOPS\n", total_gflops / (gpu_shared_time / 1000.0));
    
    // Memory bandwidth analysis
    float bytes_per_pixel = FILTER_SIZE * FILTER_SIZE * input.channels + input.channels;
    float total_gb = (input.width * input.height * bytes_per_pixel) / 1e9;
    
    printf("\nMemory Bandwidth Analysis:\n");
    printf("- Naive GPU bandwidth: %.2f GB/s\n", total_gb / (gpu_naive_time / 1000.0));
    printf("- Shared GPU bandwidth: %.2f GB/s (effective)\n", 
           total_gb / (gpu_shared_time / 1000.0));
    printf("- Theoretical peak: ~112 GB/s (RTX 2050)\n");
    
    // Real-time capability
    float target_fps = 30.0f;
    float target_ms = 1000.0f / target_fps;
    
    printf("\nReal-Time Capability (30 FPS = %.1f ms):\n", target_ms);
    printf("- CPU: %.1f FPS %s\n", 1000.0f / cpu_time, 
           cpu_time < target_ms ? "✅" : "❌");
    printf("- GPU Naive: %.1f FPS %s\n", 1000.0f / gpu_naive_time,
           gpu_naive_time < target_ms ? "✅" : "❌");
    printf("- GPU Shared: %.1f FPS %s\n", 1000.0f / gpu_shared_time,
           gpu_shared_time < target_ms ? "✅" : "❌");
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    input.free();
    output_cpu.free();
    output_gpu.free();
    delete[] h_filter;
    
    printf("\n✅ Project complete! Check output images.\n\n");
    
    return 0;
}

/**
 * COMPREHENSIVE EXERCISES
 * ======================
 * 
 * OPTIMIZATION CHALLENGES:
 * 1. Separable Filter:
 *    Implement the two-pass approach (horizontal + vertical)
 *    This should give another 3-5x speedup!
 *
 * 2. Multiple Channels:
 *    Current code processes RGB sequentially.
 *    Can you process all 3 channels in parallel?
 *
 * 3. Texture Memory:
 *    Use texture memory for better cache performance
 *    with non-aligned accesses.
 *
 * 4. Different Tile Sizes:
 *    Experiment with 8x8, 16x16, 32x32 tiles
 *    Which gives best performance? Why?
 *
 * FEATURE ADDITIONS:
 * 5. Variable Filter Size:
 *    Support 3x3, 5x5, 7x7, 9x9 filters
 *    How does performance scale?
 *
 * 6. Different Blur Types:
 *    - Box blur (uniform weights)
 *    - Motion blur (directional)
 *    - Radial blur (zoom effect)
 *
 * 7. Edge Detection:
 *    Implement Sobel filter using same framework
 *    Compare performance characteristics
 *
 * 8. Video Processing:
 *    Process video frames in real-time
 *    Use streams to overlap I/O and compute
 *
 * ANALYSIS EXERCISES:
 * 9. Occupancy Analysis:
 *    Use Nsight Compute to analyze:
 *    - Achieved occupancy
 *    - Memory throughput
 *    - Instruction throughput
 *
 * 10. Roofline Model:
 *     Plot your implementations on roofline model
 *     Are you compute or memory bound?
 *
 * 11. Power Efficiency:
 *     Measure power consumption (nvidia-smi)
 *     Calculate operations per watt
 *
 * 12. Scalability Study:
 *     Test with different image sizes:
 *     - 256x256, 512x512, 1024x1024, 2048x2048, 4096x4096
 *     How does speedup change?
 *
 * ADVANCED PROJECTS:
 * 13. Real-Time Camera:
 *     Connect to webcam, process frames live
 *     Can you maintain 60 FPS?
 *
 * 14. Instagram-Style Filters:
 *     Implement multiple effects:
 *     - Vintage (color mapping)
 *     - Vignette (darkened corners)
 *     - Sharpen (inverse blur)
 *
 * 15. AI Integration:
 *     Use blur as preprocessing for neural network
 *     Implement efficient pipeline
 *
 * PRODUCTION CONSIDERATIONS:
 * 16. Error Handling:
 *     Add comprehensive error checking
 *     Handle different image formats
 *     Validate filter parameters
 *
 * 17. Multi-GPU:
 *     Split image across multiple GPUs
 *     Handle communication overhead
 *
 * 18. Library Integration:
 *     Compare your implementation with:
 *     - OpenCV GPU modules
 *     - NPP (NVIDIA Performance Primitives)
 *     How close can you get?
 *
 * KEY INSIGHTS:
 * - Shared memory eliminates redundant global memory accesses
 * - Separable filters reduce algorithmic complexity
 * - Coalesced access patterns are crucial
 * - Real applications need careful optimization
 * - Production code needs error handling
 *
 * Congratulations! You've built a real GPU application! 🎉
 * This is what GPU programming is all about - making the
 * impossible possible through massive parallelism.
 */