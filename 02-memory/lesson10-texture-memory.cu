/*
 * Lesson 10: Texture & Surface Memory
 * Specialized Memory for Spatial Locality
 *
 * Texture memory: A blast from the past that's still incredibly useful.
 * Originally for graphics, now perfect for any spatially-local data access.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>

// =====================================================
// PART 1: FIRST PRINCIPLES - Why Texture Memory?
// =====================================================

/*
 * TEXTURE MEMORY ADVANTAGES:
 * 
 * 1. Spatial Caching: 2D/3D locality (not just 1D like L1)
 * 2. Hardware Filtering: Free interpolation
 * 3. Boundary Handling: Automatic clamp/wrap
 * 4. Separate Cache: Doesn't pollute L1/L2
 * 
 * PERFECT FOR:
 * - Image processing
 * - Volume rendering  
 * - Lookup tables
 * - Any 2D/3D data access pattern
 * 
 * Real-world impact:
 * - Image filters: 2-3x speedup
 * - Ray tracing: Better cache hit rate
 * - Scientific visualization: Free trilinear filtering
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
// PART 2: TEXTURE OBJECTS (Modern Approach)
// =====================================================

// Simple 2D convolution using global memory
__global__ void convolutionGlobal(float *output, const float *input, 
                                 int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        // 3x3 box filter
        float sum = 0.0f;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int idx = (y + dy) * width + (x + dx);
                sum += input[idx];
            }
        }
        output[y * width + x] = sum / 9.0f;
    }
}

// Same convolution using texture memory
__global__ void convolutionTexture(float *output, cudaTextureObject_t texObj,
                                  int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        // Texture coordinates (normalized)
        float u = (x + 0.5f) / width;
        float v = (y + 0.5f) / height;
        
        float sum = 0.0f;
        float du = 1.0f / width;
        float dv = 1.0f / height;
        
        // 3x3 box filter with texture fetches
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                sum += tex2D<float>(texObj, u + dx * du, v + dy * dv);
            }
        }
        output[y * width + x] = sum / 9.0f;
    }
}

// =====================================================
// PART 3: HARDWARE INTERPOLATION
// =====================================================

// Bilinear interpolation - manual
__device__ float bilinearInterpolateGlobal(const float *data, 
                                          float x, float y, 
                                          int width, int height) {
    // Compute integer coordinates
    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    
    // Clamp to image bounds
    x0 = max(0, min(x0, width - 1));
    x1 = max(0, min(x1, width - 1));
    y0 = max(0, min(y0, height - 1));
    y1 = max(0, min(y1, height - 1));
    
    // Compute weights
    float wx = x - x0;
    float wy = y - y0;
    
    // Bilinear interpolation
    float v00 = data[y0 * width + x0];
    float v10 = data[y0 * width + x1];
    float v01 = data[y1 * width + x0];
    float v11 = data[y1 * width + x1];
    
    return (1-wx)*(1-wy)*v00 + wx*(1-wy)*v10 + 
           (1-wx)*wy*v01 + wx*wy*v11;
}

// Image scaling with manual interpolation
__global__ void scaleImageGlobal(float *output, const float *input,
                                int srcWidth, int srcHeight,
                                int dstWidth, int dstHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < dstWidth && y < dstHeight) {
        // Map to source coordinates
        float srcX = x * (float)srcWidth / dstWidth;
        float srcY = y * (float)srcHeight / dstHeight;
        
        // Manual bilinear interpolation
        output[y * dstWidth + x] = bilinearInterpolateGlobal(
            input, srcX, srcY, srcWidth, srcHeight);
    }
}

// Image scaling with texture (FREE interpolation!)
__global__ void scaleImageTexture(float *output, cudaTextureObject_t texObj,
                                 int dstWidth, int dstHeight,
                                 float scaleX, float scaleY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < dstWidth && y < dstHeight) {
        // Normalized coordinates
        float u = (x + 0.5f) / dstWidth * scaleX;
        float v = (y + 0.5f) / dstHeight * scaleY;
        
        // Hardware does the interpolation!
        output[y * dstWidth + x] = tex2D<float>(texObj, u, v);
    }
}

// =====================================================
// PART 4: 3D TEXTURE FOR VOLUME DATA
// =====================================================

// Volume rendering with 3D texture
__global__ void volumeRender(float *output, cudaTextureObject_t volTex,
                           int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Ray through volume
        float sum = 0.0f;
        int steps = 100;
        
        for (int i = 0; i < steps; i++) {
            float z = (float)i / steps;
            float u = (x + 0.5f) / width;
            float v = (y + 0.5f) / height;
            float w = z;
            
            // Sample 3D texture with trilinear filtering
            float val = tex3D<float>(volTex, u, v, w);
            sum += val * 0.01f;  // Simple accumulation
        }
        
        output[y * width + x] = sum;
    }
}

// =====================================================
// PART 5: SURFACE MEMORY (Read-Write Textures)
// =====================================================

// Surface write example
__global__ void surfaceWrite(cudaSurfaceObject_t surfObj, 
                           int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Compute some value
        float val = sinf(x * 0.1f) * cosf(y * 0.1f);
        
        // Write to surface (can also read)
        surf2Dwrite(val, surfObj, x * sizeof(float), y);
    }
}

// =====================================================
// PART 6: LAYERED TEXTURES (Arrays)
// =====================================================

// Process multiple images as layers
__global__ void processLayers(float *output, cudaTextureObject_t texArray,
                            int width, int height, int numLayers) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float sum = 0.0f;
        
        // Average across all layers
        for (int layer = 0; layer < numLayers; layer++) {
            sum += tex2DLayered<float>(texArray, 
                                       (x + 0.5f) / width, 
                                       (y + 0.5f) / height, 
                                       layer);
        }
        
        output[y * width + x] = sum / numLayers;
    }
}

// =====================================================
// PART 7: PERFORMANCE COMPARISON
// =====================================================

void compareTexturePerformance() {
    printf("\n=== Texture Memory Performance ===\n");
    
    const int width = 2048;
    const int height = 2048;
    const int iterations = 100;
    
    // Allocate memory
    size_t size = width * height * sizeof(float);
    float *h_input = new float[width * height];
    float *h_output = new float[width * height];
    
    // Initialize with pattern
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            h_input[y * width + x] = sinf(x * 0.01f) * cosf(y * 0.01f);
        }
    }
    
    // Device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    // Create texture object
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_input;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32;  // bits per component
    resDesc.res.linear.sizeInBytes = size;
    
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;
    
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    
    // Test 1: Convolution
    Timer global_timer;
    for (int i = 0; i < iterations; i++) {
        convolutionGlobal<<<grid, block>>>(d_output, d_input, width, height);
    }
    cudaDeviceSynchronize();
    float global_time = global_timer.elapsed();
    
    Timer texture_timer;
    for (int i = 0; i < iterations; i++) {
        convolutionTexture<<<grid, block>>>(d_output, texObj, width, height);
    }
    cudaDeviceSynchronize();
    float texture_time = texture_timer.elapsed();
    
    printf("3x3 Convolution (%dx%d):\n", width, height);
    printf("  Global memory: %.2f ms\n", global_time);
    printf("  Texture memory: %.2f ms (%.2fx speedup)\n", 
           texture_time, global_time / texture_time);
    
    // Test 2: Image scaling (2x upscale)
    int dstWidth = width * 2;
    int dstHeight = height * 2;
    float *d_scaled;
    cudaMalloc(&d_scaled, dstWidth * dstHeight * sizeof(float));
    
    dim3 scale_grid((dstWidth + block.x - 1) / block.x,
                    (dstHeight + block.y - 1) / block.y);
    
    Timer scale_global_timer;
    scaleImageGlobal<<<scale_grid, block>>>(
        d_scaled, d_input, width, height, dstWidth, dstHeight);
    cudaDeviceSynchronize();
    float scale_global_time = scale_global_timer.elapsed();
    
    Timer scale_texture_timer;
    scaleImageTexture<<<scale_grid, block>>>(
        d_scaled, texObj, dstWidth, dstHeight, 1.0f, 1.0f);
    cudaDeviceSynchronize();
    float scale_texture_time = scale_texture_timer.elapsed();
    
    printf("\nImage Scaling (2x):\n");
    printf("  Global memory (manual interp): %.2f ms\n", scale_global_time);
    printf("  Texture memory (HW interp): %.2f ms (%.2fx speedup)\n", 
           scale_texture_time, scale_global_time / scale_texture_time);
    
    // Cleanup
    cudaDestroyTextureObject(texObj);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_scaled);
    delete[] h_input;
    delete[] h_output;
}

// =====================================================
// PART 8: MAIN - COMPREHENSIVE EXAMPLES
// =====================================================

int main() {
    printf("==================================================\n");
    printf("TEXTURE & SURFACE MEMORY\n");
    printf("==================================================\n");
    
    // Check device capabilities
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("Device: %s\n", prop.name);
    printf("Max 1D texture size: %d\n", prop.maxTexture1D);
    printf("Max 2D texture size: %d x %d\n", 
           prop.maxTexture2D[0], prop.maxTexture2D[1]);
    printf("Max 3D texture size: %d x %d x %d\n",
           prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]);
    
    // Run performance comparison
    compareTexturePerformance();
    
    // Example: Create 2D texture array
    printf("\n=== 2D Texture Array Example ===\n");
    
    const int layerWidth = 512;
    const int layerHeight = 512;
    const int numLayers = 4;
    
    // Create CUDA array (special format for textures)
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t cuArray;
    cudaExtent extent = make_cudaExtent(layerWidth, layerHeight, numLayers);
    cudaMalloc3DArray(&cuArray, &channelDesc, extent, cudaArrayLayered);
    
    // Create texture object for array
    cudaResourceDesc resDesc2 = {};
    resDesc2.resType = cudaResourceTypeArray;
    resDesc2.res.array.array = cuArray;
    
    cudaTextureDesc texDesc2 = {};
    texDesc2.addressMode[0] = cudaAddressModeWrap;  // Wrap mode
    texDesc2.addressMode[1] = cudaAddressModeWrap;
    texDesc2.filterMode = cudaFilterModeLinear;
    texDesc2.readMode = cudaReadModeElementType;
    texDesc2.normalizedCoords = 1;
    
    cudaTextureObject_t layeredTex = 0;
    cudaCreateTextureObject(&layeredTex, &resDesc2, &texDesc2, nullptr);
    
    printf("Created %d-layer texture array (%dx%d per layer)\n",
           numLayers, layerWidth, layerHeight);
    
    // Cleanup
    cudaDestroyTextureObject(layeredTex);
    cudaFreeArray(cuArray);
    
    printf("\n==================================================\n");
    printf("KEY INSIGHTS\n");
    printf("==================================================\n");
    printf("1. Texture memory excels at spatially-local access\n");
    printf("2. Hardware filtering is essentially free\n");
    printf("3. Boundary handling (clamp/wrap) is automatic\n");
    printf("4. Separate cache doesn't compete with L1\n");
    printf("5. Great for image processing and lookups\n");
    printf("6. Surface memory allows read-write access\n");
    
    return 0;
}

/*
 * COMPREHENSIVE EXERCISES:
 *
 * === Understanding ===
 * 1. When is texture memory faster than global?
 * 2. Calculate cache line efficiency for 2D access
 * 3. How does texture cache differ from L1?
 * 4. What's the cost of normalized coordinates?
 * 5. When to use linear vs array textures?
 *
 * === Implementation ===
 * 6. Implement Gaussian blur with texture
 * 7. Create 3D volume renderer
 * 8. Build texture-based lookup tables
 * 9. Implement bicubic interpolation
 * 10. Create texture-based morphology ops
 *
 * === Optimization ===
 * 11. Compare texture formats (float vs int)
 * 12. Optimize for different filter sizes
 * 13. Minimize texture cache misses
 * 14. Combine multiple textures efficiently
 * 15. Profile texture vs shared memory
 *
 * === Advanced ===
 * 16. Implement texture-based ray marching
 * 17. Create mipmapped textures
 * 18. Build texture atlases
 * 19. Implement anisotropic filtering
 * 20. Create texture-based physics sim
 *
 * === Applications ===
 * 21. Medical imaging filters
 * 22. Video processing pipeline
 * 23. Game rendering effects
 * 24. Scientific visualization
 * 25. Computer vision operators
 */

/*
 * MENTAL MODELS:
 *
 * 1. "Smart Cache"
 *    - Knows about 2D/3D patterns
 *    - Prefetches intelligently
 *    - Handles boundaries gracefully
 *
 * 2. "Graphics Heritage"
 *    - Designed for image data
 *    - Filtering is native operation
 *    - Coordinates are normalized
 *
 * 3. "Spatial Locality"
 *    - Nearby pixels likely accessed together
 *    - Cache lines are 2D/3D aware
 *    - Different from linear memory
 *
 * 4. When to Use:
 *    - Spatial access patterns
 *    - Need interpolation
 *    - Boundary conditions matter
 *    - Read-mostly data
 */
