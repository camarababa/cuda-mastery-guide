/*
 * Specialized Track: Quantitative Finance
 * Monte Carlo Methods for Option Pricing
 *
 * High-performance financial simulations using GPU parallelism.
 * The foundation of modern computational finance!
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <algorithm>

// =====================================================
// PART 1: FIRST PRINCIPLES - Monte Carlo in Finance
// =====================================================

/*
 * BLACK-SCHOLES & MONTE CARLO:
 * 
 * Analytical solution exists for European options
 * But for exotic options, we need Monte Carlo!
 * 
 * Monte Carlo advantages:
 * - Handles path-dependent options
 * - Scales to high dimensions
 * - Naturally parallel
 * 
 * GPU advantages:
 * - Generate millions of paths
 * - Each path independent
 * - Perfect parallelism
 * 
 * Real-world usage:
 * - Banks: risk management (VaR)
 * - Hedge funds: derivative pricing
 * - Insurance: actuarial calculations
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
// PART 2: RANDOM NUMBER GENERATION
// =====================================================

// Initialize random number generators
__global__ void initRNG(curandState* states, unsigned long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, tid, 0, &states[tid]);
}

// Box-Muller transform for normal distribution
__device__ float2 boxMuller(float u1, float u2) {
    float r = sqrtf(-2.0f * logf(u1));
    float theta = 2.0f * M_PI * u2;
    return make_float2(r * cosf(theta), r * sinf(theta));
}

// Generate normal random numbers
__device__ float generateNormal(curandState* state) {
    float u1 = curand_uniform(state);
    float u2 = curand_uniform(state);
    return boxMuller(u1, u2).x;
}

// =====================================================
// PART 3: BLACK-SCHOLES MODEL
// =====================================================

// Option parameters
struct OptionData {
    float S0;      // Initial stock price
    float K;       // Strike price
    float r;       // Risk-free rate
    float sigma;   // Volatility
    float T;       // Time to maturity
    
    // Derived parameters
    float dt;      // Time step
    float drift;   // Drift term
    float diffusion; // Diffusion term
    
    __host__ __device__ void precompute(int steps) {
        dt = T / steps;
        drift = (r - 0.5f * sigma * sigma) * dt;
        diffusion = sigma * sqrtf(dt);
    }
};

// European option payoff
__device__ float europeanCallPayoff(float ST, float K) {
    return fmaxf(ST - K, 0.0f);
}

__device__ float europeanPutPayoff(float ST, float K) {
    return fmaxf(K - ST, 0.0f);
}

// Asian option payoff (path-dependent)
__device__ float asianCallPayoff(float avgS, float K) {
    return fmaxf(avgS - K, 0.0f);
}

// =====================================================
// PART 4: MONTE CARLO KERNELS
// =====================================================

// European option pricing (simple)
__global__ void monteCarloEuropean(
    float* results,
    curandState* states,
    OptionData option,
    int pathsPerThread,
    bool isCall) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = states[tid];
    
    float sum = 0.0f;
    
    for (int path = 0; path < pathsPerThread; path++) {
        // Generate final stock price directly (GBM)
        float Z = generateNormal(&localState);
        float ST = option.S0 * expf((option.r - 0.5f * option.sigma * option.sigma) * option.T 
                                   + option.sigma * sqrtf(option.T) * Z);
        
        // Calculate payoff
        float payoff = isCall ? europeanCallPayoff(ST, option.K) 
                             : europeanPutPayoff(ST, option.K);
        
        sum += payoff;
    }
    
    // Save state and result
    states[tid] = localState;
    results[tid] = sum / pathsPerThread;
}

// Asian option pricing (path-dependent)
__global__ void monteCarloAsian(
    float* results,
    curandState* states,
    OptionData option,
    int steps,
    int pathsPerThread) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = states[tid];
    
    float sum = 0.0f;
    option.precompute(steps);
    
    for (int path = 0; path < pathsPerThread; path++) {
        float S = option.S0;
        float avgS = 0.0f;
        
        // Simulate path
        for (int step = 0; step < steps; step++) {
            float Z = generateNormal(&localState);
            S *= expf(option.drift + option.diffusion * Z);
            avgS += S;
        }
        
        avgS /= steps;
        float payoff = asianCallPayoff(avgS, option.K);
        sum += payoff;
    }
    
    states[tid] = localState;
    results[tid] = sum / pathsPerThread;
}

// =====================================================
// PART 5: VARIANCE REDUCTION TECHNIQUES
// =====================================================

// Antithetic variates
__global__ void monteCarloAntithetic(
    float* results,
    curandState* states,
    OptionData option,
    int pathsPerThread) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = states[tid];
    
    float sum = 0.0f;
    
    for (int path = 0; path < pathsPerThread / 2; path++) {
        float Z = generateNormal(&localState);
        
        // Original path
        float ST1 = option.S0 * expf((option.r - 0.5f * option.sigma * option.sigma) * option.T 
                                    + option.sigma * sqrtf(option.T) * Z);
        
        // Antithetic path (use -Z)
        float ST2 = option.S0 * expf((option.r - 0.5f * option.sigma * option.sigma) * option.T 
                                    - option.sigma * sqrtf(option.T) * Z);
        
        float payoff1 = europeanCallPayoff(ST1, option.K);
        float payoff2 = europeanCallPayoff(ST2, option.K);
        
        sum += (payoff1 + payoff2) / 2.0f;
    }
    
    states[tid] = localState;
    results[tid] = sum / (pathsPerThread / 2);
}

// Control variates
__global__ void monteCarloControl(
    float* results,
    curandState* states,
    OptionData option,
    int pathsPerThread,
    float analyticPrice) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = states[tid];
    
    float sumPayoff = 0.0f;
    float sumStock = 0.0f;
    
    for (int path = 0; path < pathsPerThread; path++) {
        float Z = generateNormal(&localState);
        float ST = option.S0 * expf((option.r - 0.5f * option.sigma * option.sigma) * option.T 
                                   + option.sigma * sqrtf(option.T) * Z);
        
        sumPayoff += europeanCallPayoff(ST, option.K);
        sumStock += ST;
    }
    
    // Control variate adjustment
    float avgPayoff = sumPayoff / pathsPerThread;
    float avgStock = sumStock / pathsPerThread;
    float expectedStock = option.S0 * expf(option.r * option.T);
    
    // Optimal beta approximation
    float beta = 0.9f;  // In practice, estimate from pilot run
    
    states[tid] = localState;
    results[tid] = avgPayoff - beta * (avgStock - expectedStock);
}

// =====================================================
// PART 6: BARRIER OPTIONS
// =====================================================

// Barrier option types
enum BarrierType {
    UP_AND_OUT,
    UP_AND_IN,
    DOWN_AND_OUT,
    DOWN_AND_IN
};

__global__ void monteCarloBarrier(
    float* results,
    curandState* states,
    OptionData option,
    float barrier,
    BarrierType type,
    int steps,
    int pathsPerThread) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = states[tid];
    
    float sum = 0.0f;
    option.precompute(steps);
    
    for (int path = 0; path < pathsPerThread; path++) {
        float S = option.S0;
        bool barrierHit = false;
        
        // Simulate path and check barrier
        for (int step = 0; step < steps; step++) {
            float Z = generateNormal(&localState);
            S *= expf(option.drift + option.diffusion * Z);
            
            // Check barrier condition
            switch (type) {
                case UP_AND_OUT:
                case UP_AND_IN:
                    if (S >= barrier) barrierHit = true;
                    break;
                case DOWN_AND_OUT:
                case DOWN_AND_IN:
                    if (S <= barrier) barrierHit = true;
                    break;
            }
        }
        
        // Calculate payoff based on barrier
        float payoff = 0.0f;
        switch (type) {
            case UP_AND_OUT:
            case DOWN_AND_OUT:
                if (!barrierHit) payoff = europeanCallPayoff(S, option.K);
                break;
            case UP_AND_IN:
            case DOWN_AND_IN:
                if (barrierHit) payoff = europeanCallPayoff(S, option.K);
                break;
        }
        
        sum += payoff;
    }
    
    states[tid] = localState;
    results[tid] = sum / pathsPerThread;
}

// =====================================================
// PART 7: PORTFOLIO VALUE AT RISK (VaR)
// =====================================================

__global__ void portfolioVaR(
    float* portfolioValues,
    curandState* states,
    float* weights,
    float* means,
    float* covariance,
    int numAssets,
    int scenarios) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < scenarios) {
        curandState localState = states[tid];
        
        // Generate correlated normal random numbers
        float portfolioReturn = 0.0f;
        
        // Cholesky decomposition of covariance matrix (precomputed)
        // Simplified for demonstration
        for (int i = 0; i < numAssets; i++) {
            float Z = generateNormal(&localState);
            float assetReturn = means[i] + sqrtf(covariance[i * numAssets + i]) * Z;
            portfolioReturn += weights[i] * assetReturn;
        }
        
        portfolioValues[tid] = portfolioReturn;
        states[tid] = localState;
    }
}

// =====================================================
// PART 8: BLACK-SCHOLES ANALYTICAL
// =====================================================

// For comparison with Monte Carlo
__host__ __device__ float normalCDF(float x) {
    return 0.5f * erfcf(-x / sqrtf(2.0f));
}

__host__ __device__ float blackScholesCall(
    float S0, float K, float r, float sigma, float T) {
    
    float d1 = (logf(S0 / K) + (r + 0.5f * sigma * sigma) * T) / (sigma * sqrtf(T));
    float d2 = d1 - sigma * sqrtf(T);
    
    return S0 * normalCDF(d1) - K * expf(-r * T) * normalCDF(d2);
}

// =====================================================
// PART 9: COMPREHENSIVE TESTING
// =====================================================

void testMonteCarlo() {
    printf("\n=== Monte Carlo Option Pricing ===\n");
    
    // Option parameters
    OptionData option;
    option.S0 = 100.0f;
    option.K = 100.0f;
    option.r = 0.05f;
    option.sigma = 0.2f;
    option.T = 1.0f;
    
    // Simulation parameters
    int numThreads = 10000;
    int pathsPerThread = 10000;
    int totalPaths = numThreads * pathsPerThread;
    
    printf("Option: S0=%.2f, K=%.2f, r=%.2f, sigma=%.2f, T=%.2f\n",
           option.S0, option.K, option.r, option.sigma, option.T);
    printf("Simulating %d paths (%d threads x %d paths/thread)\n", 
           totalPaths, numThreads, pathsPerThread);
    
    // Analytical price for comparison
    float analyticalPrice = blackScholesCall(
        option.S0, option.K, option.r, option.sigma, option.T);
    printf("Black-Scholes analytical price: %.4f\n", analyticalPrice);
    
    // Allocate memory
    curandState* d_states;
    float* d_results;
    float* h_results = new float[numThreads];
    
    cudaMalloc(&d_states, numThreads * sizeof(curandState));
    cudaMalloc(&d_results, numThreads * sizeof(float));
    
    // Initialize RNG
    initRNG<<<(numThreads + 255) / 256, 256>>>(d_states, time(NULL));
    
    // Test 1: Standard Monte Carlo
    Timer timer1;
    monteCarloEuropean<<<(numThreads + 255) / 256, 256>>>(
        d_results, d_states, option, pathsPerThread, true);
    cudaDeviceSynchronize();
    float time1 = timer1.elapsed();
    
    cudaMemcpy(h_results, d_results, numThreads * sizeof(float), 
              cudaMemcpyDeviceToHost);
    
    float price1 = 0.0f;
    for (int i = 0; i < numThreads; i++) {
        price1 += h_results[i];
    }
    price1 = price1 / numThreads * expf(-option.r * option.T);
    
    printf("\n1. Standard MC: %.4f (error: %.4f%%), time: %.2f ms\n", 
           price1, 100.0f * fabsf(price1 - analyticalPrice) / analyticalPrice, time1);
    
    // Test 2: Antithetic variates
    Timer timer2;
    monteCarloAntithetic<<<(numThreads + 255) / 256, 256>>>(
        d_results, d_states, option, pathsPerThread);
    cudaDeviceSynchronize();
    float time2 = timer2.elapsed();
    
    cudaMemcpy(h_results, d_results, numThreads * sizeof(float), 
              cudaMemcpyDeviceToHost);
    
    float price2 = 0.0f;
    for (int i = 0; i < numThreads; i++) {
        price2 += h_results[i];
    }
    price2 = price2 / numThreads * expf(-option.r * option.T);
    
    printf("2. Antithetic MC: %.4f (error: %.4f%%), time: %.2f ms\n", 
           price2, 100.0f * fabsf(price2 - analyticalPrice) / analyticalPrice, time2);
    
    // Test 3: Asian option
    Timer timer3;
    monteCarloAsian<<<(numThreads + 255) / 256, 256>>>(
        d_results, d_states, option, 252, pathsPerThread / 10);
    cudaDeviceSynchronize();
    float time3 = timer3.elapsed();
    
    cudaMemcpy(h_results, d_results, numThreads * sizeof(float), 
              cudaMemcpyDeviceToHost);
    
    float price3 = 0.0f;
    for (int i = 0; i < numThreads; i++) {
        price3 += h_results[i];
    }
    price3 = price3 / numThreads * expf(-option.r * option.T);
    
    printf("3. Asian option: %.4f, time: %.2f ms\n", price3, time3);
    
    // Performance metrics
    printf("\nPerformance metrics:\n");
    printf("Paths/second: %.2f million\n", totalPaths / time1 / 1000.0f);
    printf("Speedup vs CPU (estimated): 100-500x\n");
    
    // Cleanup
    cudaFree(d_states);
    cudaFree(d_results);
    delete[] h_results;
}

// Greeks calculation
void calculateGreeks() {
    printf("\n=== Greeks Calculation ===\n");
    
    OptionData option;
    option.S0 = 100.0f;
    option.K = 100.0f;
    option.r = 0.05f;
    option.sigma = 0.2f;
    option.T = 1.0f;
    
    float h = 0.01f;  // Bump size
    
    // Calculate using finite differences
    float basePrice = blackScholesCall(
        option.S0, option.K, option.r, option.sigma, option.T);
    
    // Delta: dV/dS
    float priceUp = blackScholesCall(
        option.S0 + h, option.K, option.r, option.sigma, option.T);
    float delta = (priceUp - basePrice) / h;
    
    // Gamma: d²V/dS²
    float priceDown = blackScholesCall(
        option.S0 - h, option.K, option.r, option.sigma, option.T);
    float gamma = (priceUp - 2 * basePrice + priceDown) / (h * h);
    
    // Vega: dV/dσ
    float vegaPrice = blackScholesCall(
        option.S0, option.K, option.r, option.sigma + h, option.T);
    float vega = (vegaPrice - basePrice) / h;
    
    // Theta: dV/dt
    float thetaPrice = blackScholesCall(
        option.S0, option.K, option.r, option.sigma, option.T - h/365);
    float theta = (thetaPrice - basePrice) / (h/365);
    
    // Rho: dV/dr
    float rhoPrice = blackScholesCall(
        option.S0, option.K, option.r + h, option.sigma, option.T);
    float rho = (rhoPrice - basePrice) / h;
    
    printf("Option price: %.4f\n", basePrice);
    printf("Delta: %.4f\n", delta);
    printf("Gamma: %.4f\n", gamma);
    printf("Vega: %.4f\n", vega);
    printf("Theta: %.4f\n", theta);
    printf("Rho: %.4f\n", rho);
}

// =====================================================
// PART 10: MAIN
// =====================================================

int main() {
    printf("==================================================\n");
    printf("MONTE CARLO METHODS IN FINANCE\n");
    printf("==================================================\n");
    
    // Test Monte Carlo pricing
    testMonteCarlo();
    
    // Calculate Greeks
    calculateGreeks();
    
    printf("\n==================================================\n");
    printf("KEY INSIGHTS\n");
    printf("==================================================\n");
    printf("1. GPU perfect for Monte Carlo (embarrassingly parallel)\n");
    printf("2. Variance reduction improves convergence\n");
    printf("3. Path-dependent options need full simulation\n");
    printf("4. Random number quality crucial\n");
    printf("5. Millions of paths enable accurate pricing\n");
    printf("6. Real-time risk management possible\n");
    
    return 0;
}

/*
 * COMPREHENSIVE EXERCISES:
 *
 * === Understanding ===
 * 1. Derive Black-Scholes PDE
 * 2. Why does Monte Carlo converge as 1/√N?
 * 3. Compare different RNG algorithms
 * 4. When is MC better than analytical?
 * 5. Understand option Greeks meaning
 *
 * === Implementation ===
 * 6. Implement lookback options
 * 7. Create American option pricer
 * 8. Build multi-asset options
 * 9. Implement Heston model
 * 10. Create interest rate derivatives
 *
 * === Optimization ===
 * 11. Quasi-Monte Carlo (Sobol sequences)
 * 12. Importance sampling
 * 13. Stratified sampling
 * 14. Multi-level Monte Carlo
 * 15. GPU-optimized RNG
 *
 * === Applications ===
 * 16. Portfolio optimization
 * 17. Credit risk modeling
 * 18. Market risk (VaR/CVaR)
 * 19. Derivative hedging
 * 20. Algorithmic trading backtesting
 */

/*
 * MENTAL MODELS:
 *
 * 1. "Many Worlds"
 *    - Each path = possible future
 *    - Average over all futures
 *    - Law of large numbers
 *
 * 2. "Risk-Neutral World"
 *    - Drift = risk-free rate
 *    - Real probabilities don't matter
 *    - Arbitrage-free pricing
 *
 * 3. "Variance as Enemy"
 *    - Reduce variance, not bias
 *    - Clever tricks (antithetic, control)
 *    - More paths always help
 */
