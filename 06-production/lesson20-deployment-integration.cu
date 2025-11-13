/*
 * Lesson 20: Deployment & Integration
 * Production-Ready GPU Applications
 *
 * Your amazing GPU code needs to work in the real world.
 * This lesson teaches you how to deploy, integrate, and scale.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <memory>
#include <vector>
#include <string>
#include <dlfcn.h>  // For dynamic loading

// =====================================================
// PART 1: FIRST PRINCIPLES - Production Requirements
// =====================================================

/*
 * DEVELOPMENT vs PRODUCTION:
 * 
 * Development:
 * - Single GPU, known hardware
 * - Controlled environment
 * - Focus on correctness
 * 
 * Production:
 * - Multiple GPU types
 * - Resource constraints
 * - Error recovery
 * - Monitoring
 * - Integration with existing systems
 * 
 * This lesson bridges that gap!
 */

// =====================================================
// PART 2: CUDA CONTEXT MANAGEMENT
// =====================================================

class CudaContext {
private:
    int device_id;
    cudaStream_t stream;
    bool initialized;
    
    // Device properties cache
    cudaDeviceProp device_props;
    size_t free_memory;
    size_t total_memory;
    
public:
    CudaContext(int device = 0) : device_id(device), initialized(false) {}
    
    bool initialize() {
        // Check device availability
        int device_count;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_id >= device_count) {
            fprintf(stderr, "Device %d not available\n", device_id);
            return false;
        }
        
        // Set device
        err = cudaSetDevice(device_id);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to set device %d: %s\n", 
                    device_id, cudaGetErrorString(err));
            return false;
        }
        
        // Get device properties
        cudaGetDeviceProperties(&device_props, device_id);
        cudaMemGetInfo(&free_memory, &total_memory);
        
        // Create stream
        cudaStreamCreate(&stream);
        
        initialized = true;
        return true;
    }
    
    ~CudaContext() {
        if (initialized) {
            cudaStreamDestroy(stream);
            cudaDeviceReset();
        }
    }
    
    // Getters
    cudaStream_t getStream() const { return stream; }
    const cudaDeviceProp& getDeviceProps() const { return device_props; }
    
    void printInfo() const {
        printf("Device %d: %s\n", device_id, device_props.name);
        printf("  Compute capability: %d.%d\n", 
               device_props.major, device_props.minor);
        printf("  Memory: %.1f GB free / %.1f GB total\n",
               free_memory / 1e9, total_memory / 1e9);
        printf("  SMs: %d\n", device_props.multiProcessorCount);
    }
};

// =====================================================
// PART 3: LIBRARY INTERFACE (C API)
// =====================================================

// C API for maximum compatibility
extern "C" {
    
    // Opaque handle
    typedef struct GpuProcessor_impl* GpuProcessor;
    
    // Status codes
    typedef enum {
        GPU_SUCCESS = 0,
        GPU_ERROR_INVALID_ARGUMENT = 1,
        GPU_ERROR_OUT_OF_MEMORY = 2,
        GPU_ERROR_DEVICE_NOT_AVAILABLE = 3,
        GPU_ERROR_KERNEL_LAUNCH_FAILED = 4,
        GPU_ERROR_UNKNOWN = 999
    } GpuStatus;
    
    // Version information
    typedef struct {
        int major;
        int minor;
        int patch;
        const char* build_info;
    } GpuVersion;
    
    // Create/destroy
    GpuStatus gpu_create(GpuProcessor* processor, int device_id);
    GpuStatus gpu_destroy(GpuProcessor processor);
    
    // Version
    GpuStatus gpu_get_version(GpuVersion* version);
    
    // Processing functions
    GpuStatus gpu_process_data(
        GpuProcessor processor,
        const float* input,
        float* output,
        size_t n,
        void* options
    );
}

// =====================================================
// PART 4: C++ WRAPPER WITH RAII
// =====================================================

namespace gpu {

// Exception class
class Exception : public std::exception {
private:
    std::string message;
    GpuStatus status;
    
public:
    Exception(GpuStatus status_, const std::string& msg) 
        : status(status_), message(msg) {}
    
    const char* what() const noexcept override {
        return message.c_str();
    }
    
    GpuStatus getStatus() const { return status; }
};

// RAII wrapper
class Processor {
private:
    GpuProcessor impl;
    
public:
    explicit Processor(int device_id = 0) {
        GpuStatus status = gpu_create(&impl, device_id);
        if (status != GPU_SUCCESS) {
            throw Exception(status, "Failed to create GPU processor");
        }
    }
    
    ~Processor() {
        if (impl) {
            gpu_destroy(impl);
        }
    }
    
    // Delete copy, allow move
    Processor(const Processor&) = delete;
    Processor& operator=(const Processor&) = delete;
    
    Processor(Processor&& other) noexcept : impl(other.impl) {
        other.impl = nullptr;
    }
    
    Processor& operator=(Processor&& other) noexcept {
        if (this != &other) {
            if (impl) gpu_destroy(impl);
            impl = other.impl;
            other.impl = nullptr;
        }
        return *this;
    }
    
    // Processing
    std::vector<float> process(const std::vector<float>& input) {
        std::vector<float> output(input.size());
        
        GpuStatus status = gpu_process_data(
            impl, 
            input.data(), 
            output.data(), 
            input.size(),
            nullptr
        );
        
        if (status != GPU_SUCCESS) {
            throw Exception(status, "Processing failed");
        }
        
        return output;
    }
};

} // namespace gpu

// =====================================================
// PART 5: PYTHON BINDINGS (STRUCTURE)
// =====================================================

// Example Python module structure (requires pybind11)
/*
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Python wrapper class
class PyGpuProcessor {
private:
    gpu::Processor processor;
    
public:
    PyGpuProcessor(int device_id = 0) : processor(device_id) {}
    
    py::array_t<float> process(py::array_t<float> input) {
        // Get input info
        auto buf = input.request();
        float* ptr = static_cast<float*>(buf.ptr);
        size_t size = buf.size;
        
        // Process
        std::vector<float> input_vec(ptr, ptr + size);
        auto output = processor.process(input_vec);
        
        // Return numpy array
        return py::array_t<float>(output.size(), output.data());
    }
};

// Module definition
PYBIND11_MODULE(gpu_processor, m) {
    m.doc() = "GPU Processing Module";
    
    py::class_<PyGpuProcessor>(m, "GpuProcessor")
        .def(py::init<int>(), py::arg("device_id") = 0)
        .def("process", &PyGpuProcessor::process);
}
*/

// =====================================================
// PART 6: DYNAMIC LIBRARY IMPLEMENTATION
// =====================================================

// Internal implementation
struct GpuProcessor_impl {
    std::unique_ptr<CudaContext> context;
    
    // Buffers
    float* d_buffer;
    size_t buffer_size;
    
    GpuProcessor_impl(int device_id) : d_buffer(nullptr), buffer_size(0) {
        context = std::make_unique<CudaContext>(device_id);
        if (!context->initialize()) {
            throw std::runtime_error("Failed to initialize CUDA context");
        }
    }
    
    ~GpuProcessor_impl() {
        if (d_buffer) {
            cudaFree(d_buffer);
        }
    }
};

// Simple processing kernel
__global__ void processKernel(float* output, const float* input, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Example: square each element
        output[idx] = input[idx] * input[idx];
    }
}

// C API implementation
GpuStatus gpu_create(GpuProcessor* processor, int device_id) {
    try {
        *processor = new GpuProcessor_impl(device_id);
        return GPU_SUCCESS;
    } catch (const std::exception& e) {
        return GPU_ERROR_DEVICE_NOT_AVAILABLE;
    }
}

GpuStatus gpu_destroy(GpuProcessor processor) {
    delete processor;
    return GPU_SUCCESS;
}

GpuStatus gpu_get_version(GpuVersion* version) {
    version->major = 1;
    version->minor = 0;
    version->patch = 0;
    version->build_info = __DATE__ " " __TIME__;
    return GPU_SUCCESS;
}

GpuStatus gpu_process_data(
    GpuProcessor processor,
    const float* input,
    float* output,
    size_t n,
    void* options) {
    
    // Validate arguments
    if (!processor || !input || !output || n == 0) {
        return GPU_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        // Allocate device memory if needed
        if (processor->buffer_size < n) {
            if (processor->d_buffer) {
                cudaFree(processor->d_buffer);
            }
            cudaMalloc(&processor->d_buffer, n * sizeof(float) * 2);
            processor->buffer_size = n;
        }
        
        float* d_input = processor->d_buffer;
        float* d_output = processor->d_buffer + n;
        
        // Copy input
        cudaMemcpyAsync(d_input, input, n * sizeof(float),
                       cudaMemcpyHostToDevice, 
                       processor->context->getStream());
        
        // Launch kernel
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        processKernel<<<blocks, threads, 0, 
                       processor->context->getStream()>>>(
            d_output, d_input, n);
        
        // Copy output
        cudaMemcpyAsync(output, d_output, n * sizeof(float),
                       cudaMemcpyDeviceToHost,
                       processor->context->getStream());
        
        // Synchronize
        cudaStreamSynchronize(processor->context->getStream());
        
        // Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            return GPU_ERROR_KERNEL_LAUNCH_FAILED;
        }
        
        return GPU_SUCCESS;
        
    } catch (...) {
        return GPU_ERROR_UNKNOWN;
    }
}

// =====================================================
// PART 7: DEPLOYMENT CONFIGURATIONS
// =====================================================

// Docker example
const char* dockerfile_example = R"(
# CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3-pip

# Copy library
COPY libgpu_processor.so /usr/local/lib/
COPY gpu_processor.h /usr/local/include/

# Python bindings
COPY gpu_processor.py /app/

# Set library path
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

WORKDIR /app
)";

// CMake example
const char* cmake_example = R"(
cmake_minimum_required(VERSION 3.18)
project(gpu_processor CUDA CXX)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Create library
add_library(gpu_processor SHARED
    gpu_processor.cu
)

target_compile_features(gpu_processor PUBLIC cxx_std_14)
target_link_libraries(gpu_processor PRIVATE CUDA::cudart)

# Set CUDA architectures
set_property(TARGET gpu_processor PROPERTY CUDA_ARCHITECTURES 60 70 80)

# Install
install(TARGETS gpu_processor
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
)";

// =====================================================
// PART 8: MONITORING & LOGGING
// =====================================================

class GpuMonitor {
private:
    FILE* log_file;
    bool verbose;
    
public:
    GpuMonitor(const char* log_path = nullptr, bool verbose_ = false) 
        : verbose(verbose_) {
        if (log_path) {
            log_file = fopen(log_path, "a");
        } else {
            log_file = stdout;
        }
    }
    
    ~GpuMonitor() {
        if (log_file && log_file != stdout) {
            fclose(log_file);
        }
    }
    
    void log(const char* level, const char* format, ...) {
        // Timestamp
        time_t now;
        time(&now);
        char timestamp[64];
        strftime(timestamp, sizeof(timestamp), 
                "%Y-%m-%d %H:%M:%S", localtime(&now));
        
        // Log entry
        fprintf(log_file, "[%s] [%s] ", timestamp, level);
        
        va_list args;
        va_start(args, format);
        vfprintf(log_file, format, args);
        va_end(args);
        
        fprintf(log_file, "\n");
        fflush(log_file);
    }
    
    void logGpuStatus() {
        int device;
        cudaGetDevice(&device);
        
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        
        float utilization = 100.0f * (1.0f - (float)free / total);
        
        log("INFO", "GPU %d memory: %.1f%% used (%.1f GB / %.1f GB)",
            device, utilization, (total - free) / 1e9, total / 1e9);
    }
};

// =====================================================
// PART 9: MAIN - DEPLOYMENT DEMO
// =====================================================

void demonstrateDeployment() {
    printf("\n=== Deployment Demo ===\n");
    
    // Test C API
    {
        printf("\nC API Test:\n");
        
        GpuProcessor processor;
        GpuStatus status = gpu_create(&processor, 0);
        
        if (status == GPU_SUCCESS) {
            float input[] = {1, 2, 3, 4, 5};
            float output[5];
            
            status = gpu_process_data(processor, input, output, 5, nullptr);
            
            if (status == GPU_SUCCESS) {
                printf("Results: ");
                for (int i = 0; i < 5; i++) {
                    printf("%.1f ", output[i]);
                }
                printf("\n");
            }
            
            gpu_destroy(processor);
        }
    }
    
    // Test C++ wrapper
    {
        printf("\nC++ Wrapper Test:\n");
        
        try {
            gpu::Processor processor(0);
            std::vector<float> input = {1, 2, 3, 4, 5};
            auto output = processor.process(input);
            
            printf("Results: ");
            for (float val : output) {
                printf("%.1f ", val);
            }
            printf("\n");
        } catch (const gpu::Exception& e) {
            fprintf(stderr, "Error: %s\n", e.what());
        }
    }
    
    // Monitor demo
    {
        printf("\nMonitoring Demo:\n");
        GpuMonitor monitor(nullptr, true);
        monitor.log("INFO", "Application started");
        monitor.logGpuStatus();
        monitor.log("INFO", "Processing complete");
    }
}

// Show deployment examples
void showDeploymentExamples() {
    printf("\n=== Deployment Examples ===\n");
    
    printf("\n1. Dockerfile example:\n");
    printf("%s\n", dockerfile_example);
    
    printf("\n2. CMakeLists.txt example:\n");
    printf("%s\n", cmake_example);
    
    printf("\n3. Python usage example:\n");
    printf(R"(
import numpy as np
from gpu_processor import GpuProcessor

# Create processor
processor = GpuProcessor(device_id=0)

# Process data
data = np.random.randn(1000000).astype(np.float32)
result = processor.process(data)
print(f"Processed {len(result)} elements")
)");
    
    printf("\n4. Cloud deployment (Kubernetes):\n");
    printf(R"(
apiVersion: v1
kind: Pod
metadata:
  name: gpu-processor
spec:
  containers:
  - name: processor
    image: myregistry/gpu-processor:latest
    resources:
      limits:
        nvidia.com/gpu: 1
)");
}

int main() {
    printf("==================================================\n");
    printf("DEPLOYMENT & INTEGRATION\n");
    printf("==================================================\n");
    
    // Check environment
    printf("\n=== Environment Check ===\n");
    
    int device_count;
    cudaGetDeviceCount(&device_count);
    printf("CUDA devices available: %d\n", device_count);
    
    int runtime_version, driver_version;
    cudaRuntimeGetVersion(&runtime_version);
    cudaDriverGetVersion(&driver_version);
    printf("CUDA runtime version: %d.%d\n", 
           runtime_version / 1000, (runtime_version % 100) / 10);
    printf("CUDA driver version: %d.%d\n",
           driver_version / 1000, (driver_version % 100) / 10);
    
    // Run demonstrations
    demonstrateDeployment();
    showDeploymentExamples();
    
    printf("\n==================================================\n");
    printf("DEPLOYMENT CHECKLIST\n");
    printf("==================================================\n");
    printf("✓ Error handling at every level\n");
    printf("✓ Resource management (RAII)\n");
    printf("✓ Version compatibility checks\n");
    printf("✓ Multiple language bindings\n");
    printf("✓ Monitoring and logging\n");
    printf("✓ Container support\n");
    printf("✓ Documentation\n");
    printf("✓ Testing framework\n");
    
    printf("\n==================================================\n");
    printf("KEY INSIGHTS\n");
    printf("==================================================\n");
    printf("1. Design API for stability, not just performance\n");
    printf("2. C API ensures maximum compatibility\n");
    printf("3. RAII prevents resource leaks\n");
    printf("4. Always check CUDA versions\n");
    printf("5. Monitor production deployments\n");
    printf("6. Container deployment simplifies dependencies\n");
    
    return 0;
}

/*
 * COMPREHENSIVE EXERCISES:
 *
 * === Understanding ===
 * 1. Why C API for library interface?
 * 2. How to handle GPU memory fragmentation?
 * 3. What happens with driver/runtime mismatch?
 * 4. How to ensure ABI compatibility?
 * 5. When to use static vs dynamic linking?
 *
 * === Implementation ===
 * 6. Add batch processing to API
 * 7. Implement async processing
 * 8. Create multi-GPU load balancing
 * 9. Add compression for data transfer
 * 10. Build plugin system
 *
 * === Integration ===
 * 11. Create Node.js bindings
 * 12. Build REST API server
 * 13. Implement gRPC service
 * 14. Add WebAssembly support
 * 15. Create Jupyter notebook interface
 *
 * === Deployment ===
 * 16. Setup CI/CD pipeline
 * 17. Create Helm charts
 * 18. Implement A/B testing
 * 19. Add telemetry collection
 * 20. Build auto-scaling system
 *
 * === Production ===
 * 21. Handle GPU failures gracefully
 * 22. Implement request queuing
 * 23. Add authentication/authorization
 * 24. Create performance SLAs
 * 25. Build disaster recovery
 */

/*
 * MENTAL MODELS:
 *
 * 1. "Onion Layers"
 *    - Core: CUDA kernels
 *    - Layer 1: C API
 *    - Layer 2: Language bindings
 *    - Layer 3: Applications
 *
 * 2. "Bridge Building"
 *    - GPU world ←→ CPU world
 *    - Low level ←→ High level
 *    - Performance ←→ Usability
 *
 * 3. "Production Pipeline"
 *    - Development → Testing
 *    - Testing → Staging
 *    - Staging → Production
 *    - Each step has gates
 *
 * 4. Deployment Reality:
 *    - It's not about the kernel
 *    - It's about the ecosystem
 *    - Documentation > clever code
 *    - Monitoring > assumptions
 */
