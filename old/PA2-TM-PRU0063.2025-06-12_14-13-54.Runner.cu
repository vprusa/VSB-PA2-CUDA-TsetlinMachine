#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <cmath>
#include <limits>
#include <set>
#include <float.h>
#include <limits.h>
#include <unordered_map>
#include <iomanip>
#include <unordered_set>
#include <chrono>

//#include "Tests1.h"
//#include "Headers1.h"

/*
Device 0: "Quadro P3200 with Max-Q Design"
CUDA Driver Version / Runtime Version       12.8 / 12.8
CUDA Capability Major/Minor version number: 6.1
Total amount of global memory:              6144 MBytes (6442319872 bytes)
(14) Multiprocessors, (128) CUDA Cores/MP:  1792 CUDA Cores
GPU Max Clock rate:                         1404 MHz (1.40 GHz)
Memory Clock rate:                          3505 Mhz
Memory Bus Width:                           192-bit
L2 Cache Size:                              1572864 bytes => 1.536 MB
Maximum Texture Dimension Size (x,y,z)      1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
Total amount of constant memory:            zu bytes : 65,536 bytes (64 KB)
Total amount of shared memory per block:    zu bytes : 49,152 bytes (48 KB)
Total number of registers available per block: 65536
Warp size:                                  32
Maximum number of threads per multiprocessor:  2048
Maximum number of threads per block:        1024
Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
Max dimension size of a grid size   (x,y,z): (2147483647, 65535, 65535)
Maximum memory pitch:                       zu bytes : 2,147,483,648 bytes (2 GB)
Texture alignment:                          zu bytes : 512 bytes
Concurrent copy and kernel execution:       Yes with 5 copy engine(s)
Run time limit on kernels:                  Yes
Integrated GPU sharing Host Memory:         No
Support host page-locked memory mapping:    Yes
Alignment requirement for Surfaces:         Yes
Device has ECC support:                     Disabled
CUDA Device Driver Mode (TCC or WDDM):      WDDM (Windows Display Driver Model)
Device supports Unified Addressing (UVA):   Yes
Device supports Compute Preemption:         Yes
Supports Cooperative Kernel Launch:         Yes
Supports MultiDevice Co-op Kernel Launch:   No
Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
Compute Mode:
< Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 12.8, CUDA Runtime Version = 12.8, NumDevs = 1, Device0 = Quadro P3200 with Max-Q Design
Result = PASS

*/

#define COLOR_RESET   "\033[0m"
#define COLOR_RED     "\033[31m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_BLUE    "\033[34m"
#define COLOR_MAGENTA "\033[35m"
#define COLOR_CYAN    "\033[36m"
#define COLOR_WHITE   "\033[37m"

// nVidia

#define CUDA_CHECK(msg)                                                        \
    do {                                                                       \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            std::cerr << COLOR_RED << "[CUDA ERROR] " << msg << ": "           \
                      << cudaGetErrorString(err) << COLOR_RESET << std::endl;  \
        }                                                                      \
    } while (0)


using namespace std;

// Column metainfo storage structure
struct DatasetStats {

    float float_min, float_max, float_min_diff;
    int int_min, int_max, int_min_diff;
    int bool_true_count, bool_false_count;
    int float_expected_categories;
    map<string, int> category_values;
    int num_categories;
    bool should_use;
    DatasetStats()
        : float_min(FLT_MAX), float_max(-FLT_MAX), float_min_diff(FLT_MAX),
        int_min(INT_MAX), int_max(INT_MIN), int_min_diff(INT_MAX),
        bool_true_count(0), bool_false_count(0), float_expected_categories(0), should_use(true) {
    }
};


// Dataset schema used for holding information about input data file columns, mappings and status
struct DatasetSchema {
    enum ColumnType { BOOL, CATEGORY, INTEGER, FLOAT };
    vector<pair<int, ColumnType>> column_mapping;
    bool should_shuffle;
    vector<DatasetStats> stats;
    DatasetSchema(size_t count) : stats(count) {}
};


void computeStatistics2(
    const vector<vector<string>>&raw_data,
    DatasetSchema & schema
);

void printStats(const DatasetSchema & schema);



void printCudaLaunchConfig(
    const dim3& grid, const dim3& threads, size_t shared_mem_size, int num_samples = -1
) {
    cout << "[CUDA LAUNCH CONFIGURATION]" << endl;
    cout << "num_samples: " << num_samples << endl;
    cout << "Grid dimensions: ("
        << grid.x << ", " << grid.y << ", " << grid.z << ")" << endl;
    cout << "Threads per block: ("
        << threads.x << ", " << threads.y << ", " << threads.z << ")" << endl;
    cout << "Shared memory size per block: "
        << shared_mem_size << " bytes" << endl;
}

void printCudaLaunchConfig(
    const int grid, const int threads, size_t shared_mem_size, int num_samples = -1
) {
    cout << "[CUDA LAUNCH CONFIGURATION]" << endl;
    cout << "num_samples: " << num_samples << endl;
    cout << "Grid dimensions: ("
        << grid << ", " << ", " << ")" << endl;
    cout << "Threads per block: ("
        << threads << ", " << ", " << ")" << endl;
    cout << "Shared memory size per block: "
        << shared_mem_size << " bytes" << endl;
}

void printPredictions(const std::vector<int>& predictions_host) {
    cout << "Predictions:" << endl;
    for (size_t i = 0; i < predictions_host.size(); ++i) {
        if (predictions_host[i] != 0) {
            cout << "Sample " << i << ": " << predictions_host[i] << endl;
        }
    }
}

float safeStof(const string& str) {
    try {
        return stof(str);
    }
    catch (...) {
        return 0.0f;
    }
}


void loadData(
    vector<vector<string>>& data,
    vector<string>& labels,
    const string& filename
) {
    ifstream file(filename);
    string line, cell;
    if (!getline(file, line)) {
        cout << "Error: empty file or cannot read header\n";
        return;
    }
    while (getline(file, line)) {
        vector<string> row;
        stringstream lineStream(line);
        if (getline(lineStream, cell, ';')) labels.push_back(cell);
        while (getline(lineStream, cell, ';')) {
            row.push_back(cell);
        }
        data.push_back(row);
    }
}


inline string trim(const string& s) {
    auto start = find_if_not(s.begin(), s.end(), ::isspace);
    auto end = find_if_not(s.rbegin(), s.rend(), ::isspace).base();
    return (start < end ? string(start, end) : string());
}

void loadDataFromString(const string& data_str, vector<vector<string>>& data, vector<string>& labels) {
    istringstream data_stream(data_str);
    string line;

    getline(data_stream, line);

    while (getline(data_stream, line)) {
        istringstream line_stream(line);
        string token;
        vector<string> row;

        getline(line_stream, token, ';');
        labels.push_back(trim(token));

        while (getline(line_stream, token, ';')) {
            row.push_back(trim(token));
        }

        data.push_back(row);
    }
}

vector<int> convertLabels(const vector<string>& raw_labels) {
    vector<int> labels(raw_labels.size());
    for (size_t i = 0; i < raw_labels.size(); ++i) {
        labels[i] = (raw_labels[i] == "p") ? 1 : 0;
    }
    return labels;
}

__device__ float atomicMinFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(
            address_as_int,
            assumed,
            __float_as_int(fminf(val, __int_as_float(assumed)))
        );
    } while (assumed != old);
}

__device__ void atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(
            address_as_int,
            assumed,
            __float_as_int(fmaxf(val, __int_as_float(assumed)))
        );
    } while (assumed != old);
}


// statistics are necessary to update schema with min and max values for each category
//   TODO This could be used to store prepare the data in GPU RAM right away, but...
void computeStatisticsPlain(
    const vector<vector<string>>& raw_data, // input data
    DatasetSchema& schema // schema to be updated
) {
    cout << "[LOG] Running computeStatistics on GPU...\n";
    int num_rows = raw_data.size();
    int num_cols = schema.column_mapping.size();
    // Initialize statistics
    for (auto& stat : schema.stats) {
        stat.bool_true_count = stat.bool_false_count = 0;
    }
    cout << "[LOG] Temporary CPU-side arrays to transfer data to GPU...\n";
    for (int col = 0; col < num_cols; ++col) {
        int col_idx = schema.column_mapping[col].first;
        DatasetSchema::ColumnType col_type = schema.column_mapping[col].second;
        switch (col_type) {
        case DatasetSchema::FLOAT: {
            //continue;
            break;
        }
        case DatasetSchema::INTEGER: {
            //continue;
            break;
        }

        case DatasetSchema::BOOL: {
            int true_count = 0, false_count = 0;
            for (int i = 0; i < num_rows; ++i) {
                bool val = (raw_data[i][col_idx] == "t" || raw_data[i][col_idx] == "1");
                if (val) true_count++;
                else false_count++;
            }
            schema.stats[col].bool_true_count = true_count;
            schema.stats[col].bool_false_count = false_count;
            schema.stats[col].num_categories = 2;
            break;
        }

        case DatasetSchema::CATEGORY: {
            set<string> unique_categories;
            for (int i = 0; i < num_rows; ++i) {
                unique_categories.insert(raw_data[i][col_idx]);
            }
            schema.stats[col].category_values.clear();
            int index = 0;
            for (const auto& cat : unique_categories) {
                schema.stats[col].category_values[cat] = index++;
            }
            schema.stats[col].num_categories = schema.stats[col].category_values.size();

            break;
        }
        }
    }
    cout << "[LOG] Temporary CPU-side arrays to transfer data to GPU 2...\n";

    for (DatasetStats stats : schema.stats) {
        //int num_categories = 0;
        if (!stats.category_values.empty()) {
            stats.num_categories = stats.category_values.size();
        }
        else if (stats.bool_true_count + stats.bool_false_count > 0) {
            //num_categories = ; // Boolean values have two categories
            stats.num_categories = 2;
        }
        //else if (stats.int_max != INT_MIN && stats.int_min != INT_MAX) {
            //num_categories = stats.int_max - stats.int_min + 1;  // TODO may be done better
        //}
        //else if (stats.float_max != -FLT_MAX && stats.float_min != FLT_MAX) {
            //num_categories = static_cast<int>((stats.float_max - stats.float_min) / stats.float_min_diff) + 1;
        //}
        //num_categories_per_feature.push_back(num_categories);
    }
    cout << "[LOG] CPU/GPU-based statistics computed successfully.\n";
}



__global__ void computeMinNonzeroDiffKernel(const float* data, int size, float* min_nonzero_diff) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float local_min = FLT_MAX;
    if (idx < size - 1) {
        float diff = fabsf(data[idx + 1] - data[idx]);
        if (diff > 0.0f) {
            local_min = diff;
        }
    }

    sdata[tid] = local_min;
    __syncthreads();

    // shared-memory reduction until warp-size (32 threads)
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // warp-level reduction -> no __syncthreads
    if (tid < 32) {
        volatile float* vsmem = sdata;
        vsmem[tid] = fminf(vsmem[tid], vsmem[tid + 32]);
        vsmem[tid] = fminf(vsmem[tid], vsmem[tid + 16]);
        vsmem[tid] = fminf(vsmem[tid], vsmem[tid + 8]);
        vsmem[tid] = fminf(vsmem[tid], vsmem[tid + 4]);
        vsmem[tid] = fminf(vsmem[tid], vsmem[tid + 2]);
        vsmem[tid] = fminf(vsmem[tid], vsmem[tid + 1]);
    }

    // Final atomic update by thread 0
    if (tid == 0) {
        atomicMinFloat(min_nonzero_diff, sdata[0]);
    }
}


__global__ void gpuReductionMinMaxKernel(float* input, float* output_min, float* output_max, int size) {
    extern __shared__ float sdata[];
    float* smin = sdata;
    float* smax = sdata + blockDim.x;

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + tid;

    // init sh.mem
    float val1 = (idx < size) ? input[idx] : FLT_MAX;
    float val2 = (idx + blockDim.x < size) ? input[idx + blockDim.x] : FLT_MAX;

    smin[tid] = min(val1, val2);

    val1 = (idx < size) ? input[idx] : -FLT_MAX;
    val2 = (idx + blockDim.x < size) ? input[idx + blockDim.x] : -FLT_MAX;

    smax[tid] = max(val1, val2);

    __syncthreads();

    // red for both min and max
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smin[tid] = min(smin[tid], smin[tid + s]);
            smax[tid] = max(smax[tid], smax[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output_min[blockIdx.x] = smin[0];
        output_max[blockIdx.x] = smax[0];
    }
}


template <typename T>
__global__ void reduceMinMaxKernel(const T* data, int size, T* min_val, T* max_val) {
    extern __shared__ char sdata4[];  // Shared memory as bytes
    T* smin = reinterpret_cast<T*>(sdata4);
    T* smax = reinterpret_cast<T*>(&smin[blockDim.x]);

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + tid;

    T max_init = is_same<T, float>::value ? FLT_MAX : INT_MAX;
    T min_init = is_same<T, float>::value ? -FLT_MAX : INT_MIN;

    T val1 = (idx < size) ? data[idx] : max_init;
    T val2 = (idx + blockDim.x < size) ? data[idx + blockDim.x] : max_init;

    smin[tid] = min(val1, val2);
    smax[tid] = max((idx < size) ? data[idx] : min_init,
        (idx + blockDim.x < size) ? data[idx + blockDim.x] : min_init);
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smin[tid] = min(smin[tid], smin[tid + s]);
            smax[tid] = max(smax[tid], smax[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        if constexpr (is_same<T, int>::value) {
            atomicMin((int*)min_val, smin[0]);
            atomicMax((int*)max_val, smax[0]);
        }
        else if constexpr (is_same<T, float>::value) {
            atomicMinFloat(min_val, smin[0]);
            atomicMaxFloat(max_val, smax[0]);
        }
    }
}

template<typename T>
int calculateNumCategories(const T& min_val, const T& max_val, const T& min_nonzero_diff) {
    if (min_nonzero_diff <= 0) return 1;

    return static_cast<int>(ceil((max_val - min_val) / min_nonzero_diff));
}

template <typename T>
__global__ void reduceMinMaxKernel_Global(const T* input, int size, T* global_min, T* global_max) {
    extern __shared__ char shared_mem[];
    T* smin = reinterpret_cast<T*>(shared_mem);
    T* smax = reinterpret_cast<T*>(&smin[blockDim.x]);

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + tid;

    T min_init = is_same<T, float>::value ? FLT_MAX : INT_MAX;
    T max_init = is_same<T, float>::value ? -FLT_MAX : INT_MIN;

    T min_val = (idx < size) ? input[idx] : min_init;
    T max_val = (idx < size) ? input[idx] : max_init;

    smin[tid] = min_val;
    smax[tid] = max_val;

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            smin[tid] = min(smin[tid], smin[tid + s]);
            smax[tid] = max(smax[tid], smax[tid + s]);
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile T* vmin = smin;
        volatile T* vmax = smax;
        vmin[tid] = min(vmin[tid], vmin[tid + 32]);
        vmax[tid] = max(vmax[tid], vmax[tid + 32]);
        vmin[tid] = min(vmin[tid], vmin[tid + 16]);
        vmax[tid] = max(vmax[tid], vmax[tid + 16]);
        vmin[tid] = min(vmin[tid], vmin[tid + 8]);
        vmax[tid] = max(vmax[tid], vmax[tid + 8]);
        vmin[tid] = min(vmin[tid], vmin[tid + 4]);
        vmax[tid] = max(vmax[tid], vmax[tid + 4]);
        vmin[tid] = min(vmin[tid], vmin[tid + 2]);
        vmax[tid] = max(vmax[tid], vmax[tid + 2]);
        vmin[tid] = min(vmin[tid], vmin[tid + 1]);
        vmax[tid] = max(vmax[tid], vmax[tid + 1]);
    }

    if (tid == 0) {
        if constexpr (is_same<T, float>::value) {
            atomicMinFloat(global_min, smin[0]);
            atomicMaxFloat(global_max, smax[0]);
        }
        else {
            atomicMin(global_min, smin[0]);
            atomicMax(global_max, smax[0]);
        }
    }
}


template<typename T>
void computeColumnStatistics(const vector<T>& column_data, DatasetStats& stats, cudaStream_t stream) {
    int num_rows = column_data.size();
    T* d_data;
    cudaMalloc(&d_data, num_rows * sizeof(T));
    cudaMemcpyAsync(d_data, column_data.data(), num_rows * sizeof(T), cudaMemcpyHostToDevice, stream);

    int threadsPerBlock = 1024; // max 2k per SM ...
    int blocksPerGrid = (num_rows + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);
    size_t sharedMemSize = 2 * threadsPerBlock * sizeof(T);

    T init_min = is_same<T, float>::value ? FLT_MAX : INT_MAX;
    T init_max = is_same<T, float>::value ? -FLT_MAX : INT_MIN;

    T* d_min;
    T* d_max;
    cudaMalloc(&d_min, sizeof(T));
    cudaMalloc(&d_max, sizeof(T));
    cudaMemcpyAsync(d_min, &init_min, sizeof(T), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_max, &init_max, sizeof(T), cudaMemcpyHostToDevice, stream);

    reduceMinMaxKernel_Global << <blocksPerGrid, threadsPerBlock, sharedMemSize, stream >> > (
        d_data, num_rows, d_min, d_max
    );

    cudaMemcpyAsync(&stats.float_min, d_min, sizeof(T), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&stats.float_max, d_max, sizeof(T), cudaMemcpyDeviceToHost, stream);

    if constexpr (is_same<T, float>::value) {
        
        float* d_min_diff;
        cudaMalloc(&d_min_diff, sizeof(float));
        float init_diff = FLT_MAX;
        cudaMemcpyAsync(d_min_diff, &init_diff, sizeof(float), cudaMemcpyHostToDevice, stream);

        computeMinNonzeroDiffKernel << <blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float), stream >> > (
            d_data, num_rows, d_min_diff
        );

        cudaMemcpyAsync(&stats.float_min_diff, d_min_diff, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaFree(d_min_diff);

        stats.num_categories = calculateNumCategories(stats.float_min, stats.float_max, stats.float_min_diff);

        //stats.float_min_diff = (stats.float_max - stats.float_min) / 10.0f;
        stats.float_min_diff = (stats.float_max - stats.float_min) / 10.0f;
        //stats.float_min_diff = (stats.float_max - stats.float_min) / 4.0f;
        //stats.float_min_diff = (stats.float_max - stats.float_min) / 100.0f;
        stats.num_categories = calculateNumCategories(stats.float_min, stats.float_max, stats.float_min_diff);

        /*
        // Use midpoint to binarize the feature later
        float float_min = stats.float_min;
        float float_max = stats.float_max;
        float midpoint = (float_max - float_min) /10.0f;

        stats.float_min_diff = stats.float_min_diff * 10;  // reuse the field to store midpoint for binarization
        stats.num_categories = stats.num_categories / 10;         // since it's binary
      
        float float_min = stats.float_min;
        float float_max = stats.float_max;
        float midpoint = (float_max - float_min) / 2.0f;
        
        stats.float_min_diff = midpoint - float_min;
        stats.num_categories = calculateNumCategories(stats.float_min, stats.float_max, stats.float_min_diff);
        */
    }

    cudaStreamSynchronize(stream);

    cudaFree(d_data);
    cudaFree(d_min);
    cudaFree(d_max);
}



void computeStatisticsReduction(const vector<vector<float>>& raw_data, DatasetSchema& schema) {
    int num_rows = raw_data.size();
    int num_cols = schema.column_mapping.size();

    vector<cudaStream_t> streams(num_cols);
    for (int col = 0; col < num_cols; ++col) {
        cudaStreamCreate(&streams[col]);
        int col_idx = schema.column_mapping[col].first;

        switch (schema.column_mapping[col].second) {
        case DatasetSchema::FLOAT: {
            vector<float> col_data(num_rows);
            for (int i = 0; i < num_rows; ++i) {
                col_data[i] = raw_data[i][col_idx];
            }
            computeColumnStatistics<float>(col_data, schema.stats[col], streams[col]);
            //computeColumnStatistics2<float>(col_data, schema.stats[col], streams[col]);
            break;
        }
        case DatasetSchema::INTEGER: {
            vector<int> col_data(num_rows);
            for (int i = 0; i < num_rows; ++i) {
                col_data[i] = static_cast<int>(raw_data[i][col_idx]);
            }
            computeColumnStatistics<int>(col_data, schema.stats[col], streams[col]);
            //computeColumnStatistics2<int>(col_data, schema.stats[col], streams[col]);
            schema.stats[col].num_categories = schema.stats[col].int_max - schema.stats[col].int_min;
            break;
        }
        default:
            break;
        }
    }

    // Cleanup streams
    for (int col = 0; col < num_cols; ++col)
        cudaStreamDestroy(streams[col]);
}

// same as a `computeStatistics` but on CPU
void computeStatisticsCPU(const vector<vector<string>>& raw_data, DatasetSchema& schema) {
    size_t num_samples = raw_data.size();
    for (size_t j = 0; j < schema.column_mapping.size(); ++j) {
        auto col_type = schema.column_mapping[j].second;
        int col_idx = schema.column_mapping[j].first;
        DatasetStats& stats = schema.stats[j];
        switch (col_type) {
        case DatasetSchema::BOOL: {
            stats.bool_true_count = 0;
            stats.bool_false_count = 0;
            for (size_t i = 0; i < num_samples; ++i) {
                string val = raw_data[i][col_idx];
                if (val == "t" || val == "1" || val == "true")
                    stats.bool_true_count++;
                else
                    stats.bool_false_count++;
            }
            break;
        }
        case DatasetSchema::CATEGORY: {
            stats.category_values.clear();
            int category_index = 0;
            for (size_t i = 0; i < num_samples; ++i) {
                const auto& val = raw_data[i][col_idx];
                if (stats.category_values.find(val) == stats.category_values.end()) {
                    stats.category_values[val] = category_index++;
                }
            }
            break;
        }
        case DatasetSchema::INTEGER: {
            stats.int_min = INT_MAX;
            stats.int_max = INT_MIN;
            set<int> unique_vals;
            for (size_t i = 0; i < num_samples; ++i) {
                int val = stoi(raw_data[i][col_idx]);
                stats.int_min = min(stats.int_min, val);
                stats.int_max = max(stats.int_max, val);
                unique_vals.insert(val);
            }
            stats.int_min_diff = INT_MAX;
            int prev = *unique_vals.begin();
            for (auto it = ++unique_vals.begin(); it != unique_vals.end(); ++it) {
                int diff = *it - prev;
                if (diff > 0)
                    stats.int_min_diff = min(stats.int_min_diff, diff);
                prev = *it;
            }
            if (stats.int_min_diff == INT_MAX) stats.int_min_diff = 1; // Fallback
            break;
        }
        case DatasetSchema::FLOAT: {
            // skipping, already calc via GPU
            // can be used for double-check calc correctness or debugging
            /*
            stats.float_min = numeric_limits<float>::max();
            stats.float_max = numeric_limits<float>::lowest();
            set<float> unique_vals;
            for (size_t i = 0; i < num_samples; ++i) {
                float val = stof(raw_data[i][col_idx]);
                stats.float_min = min(stats.float_min, val);
                stats.float_max = max(stats.float_max, val);
                unique_vals.insert(val);
            }
            stats.float_min_diff = numeric_limits<float>::max();
            float prev = *unique_vals.begin();
            for (auto it = ++unique_vals.begin(); it != unique_vals.end(); ++it) {
                float diff = fabs(*it - prev);
                if (diff > 0)
                    stats.float_min_diff = min(stats.float_min_diff, diff);
                prev = *it;
            }
            if (stats.float_min_diff == numeric_limits<float>::max()) stats.float_min_diff = 0.001f; // Fallback
            stats.float_expected_categories = static_cast<int>(
                ((stats.float_max - stats.float_min) / stats.float_min_diff) + 1);
            */
            break;
        }
        }
    }
}


// CPU categorization
void categorizeData(
    const vector<vector<string>>& raw_data, // input data
    const DatasetSchema& schema, // schema used for categorization
    vector<vector<int>>& categorized_data // resulting categorized data
) {
    size_t num_samples = raw_data.size();
    size_t num_features = schema.column_mapping.size();
    categorized_data.resize(num_samples, vector<int>(num_features));
    for (size_t i = 0; i < num_samples; ++i) {
        for (size_t j = 0; j < num_features; ++j) {
            int col_idx = schema.column_mapping[j].first;
            const auto& stats = schema.stats[j];
            switch (schema.column_mapping[j].second) {
            case DatasetSchema::BOOL:
                categorized_data[i][j] = (raw_data[i][col_idx] == "t" || raw_data[i][col_idx] == "1") ? 1 : 0;
                break;
            case DatasetSchema::CATEGORY:
                if (stats.category_values.count(raw_data[i][col_idx]))
                    categorized_data[i][j] = stats.category_values.at(raw_data[i][col_idx]);
                else
                    categorized_data[i][j] = 0; // default/fallback category
                break;
            case DatasetSchema::INTEGER:
                categorized_data[i][j] = (stoi(raw_data[i][col_idx]) - stats.int_min) / stats.int_min_diff;
                break;
            case DatasetSchema::FLOAT:
                categorized_data[i][j] = (int)((stof(raw_data[i][col_idx]) - stats.float_min) / stats.float_min_diff);
                break;
            }
        }
    }
}

// Helper macro with function
#define cudaCheck(err) { cudaAssert((err), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        cout << "CUDA error: " << cudaGetErrorString(code) << " (" << file << ":" << line << ")\n";
        if (abort) exit(code);
    }
}


void splitDataset(
    vector<vector<string>> raw_data,
    vector<string> raw_labels,
    float train_ratio,
    vector<vector<string>>* train_raw,
    vector<string>* train_labels_str,
    vector<vector<string>>* valid_raw,
    vector<string>* valid_labels_str,
    DatasetSchema& schema,
    bool shoudShffle = true
) {

    int total_samples = raw_data.size();
    int train_size = static_cast<int>(total_samples * train_ratio);
    int valid_size = total_samples - train_size;


    vector<int> indices(total_samples);
    iota(indices.begin(), indices.end(), 0);

    if (shoudShffle) {
        //if (schema.should_shuffle) {
        shuffle(indices.begin(), indices.end(), mt19937(random_device{}()));
        //}
    }

    auto copySubset = [&](int start, int size, auto& dst_data, auto& dst_labels) {
        dst_data.resize(size);
        dst_labels.resize(size);
        for (int i = 0; i < size; ++i) {
            dst_data[i] = raw_data[indices[start + i]];
            dst_labels[i] = raw_labels[indices[start + i]];
        }
        };


    copySubset(0, train_size, *train_raw, *train_labels_str);
    copySubset(train_size, valid_size, *valid_raw, *valid_labels_str);
}

void convertDataToNumericTable(
    const vector<vector<string>>& data,
    const DatasetSchema& schema,
    vector<vector<float>>* numeric_data
) {
    numeric_data->resize(data.size(), vector<float>(schema.column_mapping.size()));
    vector<unordered_map<string, int>> category_mappings(schema.column_mapping.size());

    for (size_t col = 0; col < schema.column_mapping.size(); ++col) {
        DatasetSchema::ColumnType type = schema.column_mapping[col].second;

        if (type == DatasetSchema::CATEGORY) {
            const map<string, int>& known_categories = schema.stats[col].category_values;
            for (size_t row = 0; row < data.size(); ++row) {
                const string& val = data[row][col];
                map<string, int>::const_iterator it = known_categories.find(val);

                if (it != known_categories.end()) {
                    (*numeric_data)[row][col] = static_cast<float>(it->second);
                } else {
                    (*numeric_data)[row][col] = static_cast<float>(known_categories.size()); 
                }
            }
        }
        else if (type == DatasetSchema::INTEGER) {
            for (size_t row = 0; row < data.size(); ++row) {
                (*numeric_data)[row][col] = static_cast<float>(stoi(data[row][col]));
            }
        }
        else if (type == DatasetSchema::FLOAT) {
            for (size_t row = 0; row < data.size(); ++row) {
                (*numeric_data)[row][col] = stof(data[row][col]);
            }
        }
        else if (type == DatasetSchema::BOOL) {
            for (size_t row = 0; row < data.size(); ++row) {
                const string& val = data[row][col];
                if (val == "1" || val == "t" || val == "T" || val == "true" || val == "True") {
                    (*numeric_data)[row][col] = 1.0f;
                }
                else {
                    (*numeric_data)[row][col] = 0.0f;
                }
            }
        }
    }
}

void convertDataToNumericTable2(
    const std::vector<std::vector<std::string>>& data,
    const DatasetSchema& schema,
    std::vector<std::vector<float>>* numeric_data
) {
    if (data.empty() || schema.column_mapping.empty()) return;

    size_t num_rows = data.size();
    size_t num_cols = schema.column_mapping.size();

    numeric_data->resize(num_rows, std::vector<float>(num_cols, 0.0f));
    std::vector<std::unordered_map<std::string, int>> category_mappings(num_cols);
    std::vector<int> category_counters(num_cols, 0);

    for (size_t col = 0; col < num_cols; ++col) {
        // Manual lookup for column type
        DatasetSchema::ColumnType col_type = DatasetSchema::FLOAT;
        bool found = false;
        for (size_t i = 0; i < schema.column_mapping.size(); ++i) {
            if (schema.column_mapping[i].first == static_cast<int>(col)) {
                col_type = schema.column_mapping[i].second;
                found = true;
                break;
            }
        }
        if (!found) continue;

        for (size_t row = 0; row < num_rows; ++row) {
            if (col >= data[row].size()) continue;  // safeguard
            const std::string& value = data[row][col];

            if (col_type == DatasetSchema::CATEGORY) {
                if (category_mappings[col].count(value) == 0) {
                    category_mappings[col][value] = category_counters[col]++;
                }
                (*numeric_data)[row][col] = static_cast<float>(category_mappings[col][value]);
            }
            else if (col_type == DatasetSchema::INTEGER || col_type == DatasetSchema::BOOL) {
                try {
                    (*numeric_data)[row][col] = static_cast<float>(std::stoi(value));
                }
                catch (...) {
                    (*numeric_data)[row][col] = 0.0f;
                }
            }
            else if (col_type == DatasetSchema::FLOAT) {
                try {
                    (*numeric_data)[row][col] = value.empty() ? 0.0f : std::stof(value);
                }
                catch (...) {
                    (*numeric_data)[row][col] = 0.0f;
                }
            }
        }
    }
}


void convertNumericFloatsToInts(
    const vector<vector<float>>& numeric_data_float,
    const DatasetSchema& schema,
    vector<vector<int>>* numeric_data_int
) {
    numeric_data_int->resize(numeric_data_float.size(), vector<int>(schema.column_mapping.size()));


    for (size_t col = 0; col < schema.column_mapping.size(); ++col) {
        if (schema.column_mapping[col].second == DatasetSchema::FLOAT) {
            float float_min = schema.stats[col].float_min;
            float float_min_diff = schema.stats[col].float_min_diff;


            for (size_t row = 0; row < numeric_data_float.size(); ++row) {
                (*numeric_data_int)[row][col] = static_cast<int>(round((numeric_data_float[row][col] - float_min) / float_min_diff));
            }
        }
        else {
            for (size_t row = 0; row < numeric_data_float.size(); ++row) {
                (*numeric_data_int)[row][col] = static_cast<int>(numeric_data_float[row][col]);
            }
        }
    }
}


void testConvertNumericFloatsToInts() {
    void (*runTest)(void (*)(), const string&) = [](void (*testFn)(), const string& testName) {
        try {
            testFn();
            cout << testName << " passed." << endl;
        }
        catch (const exception& e) {
            cout << testName << " failed: " << e.what() << endl;
        }
        };


    void (*testFloatConversion)() = []() {
        DatasetSchema schema(1);
        schema.column_mapping = { {0, DatasetSchema::FLOAT} };
        schema.stats[0].float_min = 0.0f;
        schema.stats[0].float_min_diff = 0.1f;
        vector<vector<float>> numeric_data_float = { {0.0f}, {0.1f}, {0.2f}, {0.4f}, {0.5f} };
        vector<vector<int>> numeric_data_int;


        convertNumericFloatsToInts(numeric_data_float, schema, &numeric_data_int);


        vector<int> expected_values = { 0, 1, 2, 4, 5 };
        for (size_t i = 0; i < numeric_data_int.size(); ++i) {
            if (numeric_data_int[i][0] != expected_values[i])
                throw runtime_error("Incorrect value at row " + to_string(i));
        }
        };


    runTest(testFloatConversion, "Test FLOAT conversion case");
}

void convertLabelsToInt(
    const vector<string>& labels_str,
    vector<int>& labels_int,
    map<string, int>& label_mapping
) {
    int label_index = 0;
    for (const auto& label : labels_str) {
        if (label_mapping.find(label) == label_mapping.end()) {
            label_mapping[label] = label_index++;
        }
        labels_int.push_back(label_mapping[label]);
    }
}
/*
__global__ void validateClausesKernel2(
    const int* d_numeric_data,
    const int* d_clause_states,
    int num_samples,
    int num_features,
    const int* d_num_categories,
    const int* d_num_categories_offsets,
    int num_literals,
    int num_clauses,
    int* d_predictions,
    int samples_per_thread = 1
) {
    int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_id >= num_samples) return;

    int vote_sum = 0;

    for (int feature_id = 0; feature_id < num_features; ++feature_id) {
        int category = d_numeric_data[sample_id * num_features + feature_id];
        int literal_start_index = d_num_categories_offsets[feature_id];

        int pos_clause_index = literal_start_index + category * 2;
        int neg_clause_index = pos_clause_index + 1;

        int pos_val = d_clause_states[pos_clause_index];
        int neg_val = d_clause_states[neg_clause_index];

        if (pos_val > 0 && pos_val >= neg_val) {
            vote_sum += 1;
        }
    }

    d_predictions[sample_id] = (vote_sum > (num_features / 2)) ? 1 : 0;
}
*/
__global__ void validateClausesKernel(
    const int* d_numeric_data,
    const int* d_clause_states,
    int num_samples,
    int num_features,
    const int* d_num_categories,
    const int* d_num_categories_offsets,
    int num_literals,
    int num_clauses,
    int* d_predictions,
    int samples_per_thread = 1
) {
    int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_id >= num_samples) return;

    int vote_sum = 0;

    for (int feature_id = 0; feature_id < num_features; ++feature_id) {
        int observed_category = d_numeric_data[sample_id * num_features + feature_id];
        int num_categories = d_num_categories[feature_id];
        int base_offset = d_num_categories_offsets[feature_id];

        for (int cat = 0; cat < num_categories; ++cat) {
            int pos_clause_index = (base_offset + cat) * 2;
            int neg_clause_index = pos_clause_index + 1;

            int pos_val = d_clause_states[pos_clause_index];
            int neg_val = d_clause_states[neg_clause_index];

            // Positive literal matches if category equals observed
            if (cat == observed_category && pos_val > 0 && pos_val >= neg_val) {
                vote_sum += 1;
            }
            // Negative literal matches if category != observed
            else if (cat != observed_category && neg_val > 0 && neg_val >= pos_val) {
                vote_sum += 1;
            }
        }
    }

    d_predictions[sample_id] = (vote_sum >= (num_features / 2)) ? 1 : 0;
    d_predictions[sample_id] = vote_sum; // (vote_sum > (num_features / 2)) ? 1 : 0;
    d_predictions[sample_id] = vote_sum; // (vote_sum > (num_features / 2)) ? 1 : 0;
}


void printStats(const DatasetSchema& schema) {
    cout << left << setw(10) << "Column"
        << setw(12) << "Type"
        << setw(15) << "Float Min"
        << setw(15) << "Float Max"
        << setw(15) << "Float Min Diff"
        << setw(15) << "Int Min"
        << setw(15) << "Int Min Diff"
        << setw(15) << "Int Max"
        << setw(15) << "Bool True"
        << setw(15) << "Bool False"
        << setw(20) << "Categories" << endl;

    cout << string(127, '-') << endl;
    for (size_t i = 0; i < schema.stats.size(); ++i) {
        const auto& stat = schema.stats[i];
        const auto& colType = schema.column_mapping[i].second;

        //string typeStr;
        string typeStr = "UNKNOWN";
        switch (colType) {
        case DatasetSchema::BOOL: typeStr = "BOOL"; break;
        case DatasetSchema::CATEGORY: typeStr = "CATEGORY"; break;
        case DatasetSchema::INTEGER: typeStr = "INTEGER"; break;
        case DatasetSchema::FLOAT: typeStr = "FLOAT"; break;
        }

        cout << left << setw(10) << i
            << setw(12) << typeStr
            << setw(15) << (colType == DatasetSchema::FLOAT ? to_string(stat.float_min) : "N/A")
            << setw(15) << (colType == DatasetSchema::FLOAT ? to_string(stat.float_max) : "N/A")
            << setw(15) << (colType == DatasetSchema::FLOAT ? to_string(stat.float_min_diff) : "N/A")
            << setw(15) << (colType == DatasetSchema::INTEGER ? to_string(stat.int_min) : "N/A")
            << setw(15) << (colType == DatasetSchema::INTEGER ? to_string(stat.int_max) : "N/A")
            << setw(15) << (colType == DatasetSchema::INTEGER ? to_string(stat.int_min_diff) : "N/A")
            << setw(15) << (colType == DatasetSchema::BOOL ? to_string(stat.bool_true_count) : "N/A")
            << setw(15) << (colType == DatasetSchema::BOOL ? to_string(stat.bool_false_count) : "N/A");

        if (colType == DatasetSchema::CATEGORY) {
            string categories;
            for (const auto& kv : stat.category_values) {
                categories += kv.first + "(" + to_string(kv.second) + ") ";
            }
            cout << setw(20) << categories;
        }
        else if (colType == DatasetSchema::FLOAT) {
            cout << setw(20) << stat.num_categories;
        }
        else if (colType == DatasetSchema::BOOL) {
            cout << setw(20) << stat.num_categories;
        }
        else {
            cout << setw(20) << "N/A";
        }
        cout << endl;
    }
}


void retrieveDeviceData(
    int num_clauses, int num_literals, int num_samples,
    int* d_clause_states,
    int* d_debug_shared_literals,
    int* d_numeric_data,
    int* d_labels,
    int* d_num_categories,
    vector<int>& clause_states,
    vector<int>& shared_literals_host,
    vector<int>& numeric_data_host,
    vector<int>& labels_host,
    vector<int>& num_categories_host
) {
    clause_states.resize(num_clauses);
    cudaMemcpy(clause_states.data(), d_clause_states, clause_states.size() * sizeof(int), cudaMemcpyDeviceToHost);

    shared_literals_host.resize(num_samples * num_literals);
    cudaMemcpy(shared_literals_host.data(), d_debug_shared_literals, shared_literals_host.size() * sizeof(int), cudaMemcpyDeviceToHost);

    numeric_data_host.resize(num_samples * num_literals);
    cudaMemcpy(numeric_data_host.data(), d_numeric_data, numeric_data_host.size() * sizeof(int), cudaMemcpyDeviceToHost);

    labels_host.resize(num_samples);
    cudaMemcpy(labels_host.data(), d_labels, labels_host.size() * sizeof(int), cudaMemcpyDeviceToHost);

    num_categories_host.resize(num_literals);
    cudaMemcpy(num_categories_host.data(), d_num_categories, num_categories_host.size() * sizeof(int), cudaMemcpyDeviceToHost);
}


void printTestData(
    const vector<int>& clause_states,
    const vector<int>& shared_literals_host,
    const vector<int>& numeric_data_host,
    const vector<int>& labels_host,
    const vector<int>& num_categories_host,
    size_t num_samples,
    int num_literals
) {
    cout << "Clause States:" << endl;
    for (size_t i = 0; i < clause_states.size(); ++i) {
        cout << "clause_states[" << i << "]: " << clause_states[i] << endl;
    }

    //cout << "Shared Literals:" << endl;
    //for (size_t sample = 0; sample < num_samples; ++sample) {
    //    cout << "Sample " << sample << ": ";
    //    for (int lit = 0; lit < num_literals; ++lit) {
    //        cout << shared_literals_host[sample * num_literals + lit] << " ";
    //    }
    //    cout << endl;
    //}

    cout << "Numeric Data:" << endl;
    for (size_t i = 0; i < numeric_data_host.size(); ++i) {
        cout << "numeric_data[" << i << "]: " << numeric_data_host[i] << endl;
    }

    cout << "Labels:" << endl;
    for (size_t i = 0; i < labels_host.size(); ++i) {
        cout << "labels[" << i << "]: " << labels_host[i] << endl;
    }

    cout << "Num Categories:" << endl;
    for (size_t i = 0; i < num_categories_host.size(); ++i) {
        cout << "num_categories[" << i << "]: " << num_categories_host[i] << endl;
    }
}

bool verifyTestData(
    const vector<int>& clause_states,
    const vector<int>& shared_literals_host,
    const vector<vector<int>>& expected_clause_states,
    const vector<vector<int>>& expected_shared_literals
) {
    bool verification_passed = true;

    for (size_t i = 0; i < expected_clause_states.size(); ++i) {
        for (size_t j = 0; j < expected_clause_states[i].size(); ++j) {
            if (clause_states[i * expected_clause_states[i].size() + j] != expected_clause_states[i][j]) {
                cout << "Mismatch in clause_states at clause " << i << ", literal " << j << " " << clause_states[i * expected_clause_states[i].size() + j] << " != " << expected_clause_states[i][j] << endl;
                verification_passed = false;
            }
        }
    }
    for
        (size_t sample = 0; sample < expected_shared_literals.size(); ++sample) {
        for (size_t lit = 0; lit < expected_shared_literals[sample].size(); ++lit) {
            if (shared_literals_host[sample * expected_shared_literals[sample].size() + lit] != expected_shared_literals[sample][lit]) {
                cout << COLOR_RED << "Mismatch in shared_literals at sample " << sample << ", literal " << lit << " "
                    << shared_literals_host[sample * expected_shared_literals[sample].size() + lit] << " != " << expected_shared_literals[sample][lit] << COLOR_RESET << endl;
                verification_passed = false;
            }
        }
    }
    return verification_passed;
}

void printDatasetSplits(
    const vector<vector<string>>& train_raw,
    const vector<string>& train_labels_str,
    const vector<vector<string>>& valid_raw,
    const vector<string>& valid_labels_str) {

    cout << "Training Data:" << endl;
    for (size_t i = 0; i < train_raw.size(); ++i) {
        cout << "Sample " << i << ": ";
        for (const auto& val : train_raw[i]) {
            cout << val << " ";
        }
        cout << " | Label: " << train_labels_str[i] << endl;
    }

    cout << "\nValidation Data:" << endl;
    for (size_t i = 0; i < valid_raw.size(); ++i) {
        cout << "Sample " << i << ": ";
        for (const auto& val : valid_raw[i]) {
            cout << val << " ";
        }
        cout << " | Label: " << valid_labels_str[i] << endl;
    }
}

void printNumericDataAndLabels(
    const vector<vector<float>>& train_numeric_float, const vector<int>& train_labels_int,
    const vector<vector<float>>& valid_numeric_float, const vector<int>& valid_labels_int,
    const vector<vector<int>>& train_numeric_int, const vector<vector<int>>& valid_numeric_int
) {
    cout << "Train Numeric Float Data:" << endl;
    for (size_t i = 0; i < train_numeric_float.size(); ++i) {
        cout << "Sample " << i << ": ";
        for (const auto& val : train_numeric_float[i]) cout << val << " ";
        cout << "| Label (int): " << train_labels_int[i] << endl;
    }

    cout << "\nValidation Numeric Float Data:" << endl;
    for (size_t i = 0; i < valid_numeric_float.size(); ++i) {
        cout << "Sample " << i << ": ";
        for (const auto& val : valid_numeric_float[i]) cout << val << " ";
        cout << "| Label (int): " << valid_labels_int[i] << endl;
    }

    cout << "\nTrain Numeric Int Data:" << endl;
    for (size_t i = 0; i < train_numeric_int.size(); ++i) {
        cout << "Sample " << i << ": ";
        for (const auto& val : train_numeric_int[i]) cout << val << " ";
        cout << endl;
    }

    cout << "\nValidation Numeric Int Data:" << endl;
    for (size_t i = 0; i < valid_numeric_int.size(); ++i) {
        cout << "Sample " << i << ": ";
        for (const auto& val : valid_numeric_int[i]) cout << val << " ";
        cout << endl;
    }
}


void printNumericIntData(
    const std::vector<std::vector<int>>& train_numeric_int,
    const std::vector<std::vector<int>>& valid_numeric_int
) {
    std::cout << "\nTrain Numeric Int Data:\n";
    for (size_t i = 0; i < train_numeric_int.size(); ++i) {
        std::cout << "Sample " << i << ": ";
        for (int val : train_numeric_int[i]) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\nValidation Numeric Int Data:\n";
    for (size_t i = 0; i < valid_numeric_int.size(); ++i) {
        std::cout << "Sample " << i << ": ";
        for (int val : valid_numeric_int[i]) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }
}


void printNumericIntDataS(
    const std::vector<std::vector<int>>& train_numeric_int,
    const std::vector<std::vector<int>>& valid_numeric_int
) {
    std::cout << "\nTrain Numeric Int Data:\n";
    for (size_t i = 0; i < train_numeric_int.size(); ++i) {
        //std::cout << "Sample " << i << ": ";
        for (int val : train_numeric_int[i]) {
            //std::cout << val << " ";
        }
        //std::cout << "\n";
    }

    std::cout << "\nValidation Numeric Int Data:\n";
    for (size_t i = 0; i < valid_numeric_int.size(); ++i) {
        std::cout << "Sample " << i << ": ";
        for (int val : valid_numeric_int[i]) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }
}

__device__ inline int clamp(int val, int min_val, int max_val) {
    return max(min(val, max_val), min_val);
}

__device__ inline void atomicClampAdd(int* addr, int value, int min_val, int max_val) {
    int old = atomicAdd(addr, value);
    int new_val = clamp(old + value, min_val, max_val);
    atomicExch(addr, new_val);
}

__global__ void trainClausesKernel(
    const int* d_numeric_data,
    const int* d_labels,
    int num_samples,
    int num_features,
    const int* d_num_categories,
    const int* d_num_categories_offsets,
    int num_literals,
    int num_clauses,
    int* d_clause_states,
    int* d_debug_shared_literals,
    int samples_per_thread = 1
) {
    //int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int block_id = blockIdx.z * gridDim.y * gridDim.x +
        blockIdx.y * gridDim.x +
        blockIdx.x;

    int thread_id_in_block = threadIdx.y * blockDim.x + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;

    int thread_id = block_id * threads_per_block + thread_id_in_block;
    
    int total_threads = num_samples * num_features;
    if (thread_id >= total_threads) { return; }

    int sample_id = thread_id / num_features;
    int feature_id = thread_id % num_features;

    if (sample_id >= num_samples) { return; }
    if (feature_id >= num_features) { return; }

    int observed_category = d_numeric_data[sample_id * num_features + feature_id];
    int num_categories = d_num_categories[feature_id];
    int base_offset = d_num_categories_offsets[feature_id];

    int label = d_labels[sample_id];

    //if (sample_id < 5) {
    if (feature_id < 1) {
        //printf("Thread %d → Sample %d, Label = %d, num_samples: %d, feature_id: %d\n", thread_id, sample_id, d_labels[sample_id], num_samples, feature_id);
    }

    int pos_clause_index = (base_offset + observed_category) * 2;
    int neg_clause_index = pos_clause_index + 1;
    /*
    if (label == 1) {
        atomicAdd(&d_clause_states[pos_clause_index], 4);
        atomicSub(&d_clause_states[neg_clause_index], 3);
    } else {
        atomicSub(&d_clause_states[pos_clause_index], 1);
        atomicAdd(&d_clause_states[neg_clause_index], 2);
    }
    */
    
    if (label == 1) {
        atomicAdd(&d_clause_states[pos_clause_index], 5);
        atomicSub(&d_clause_states[neg_clause_index], 3);
    }
    else {
        atomicSub(&d_clause_states[pos_clause_index], 1);
        atomicAdd(&d_clause_states[neg_clause_index], 2);
    }
  
    /*
    //int pos_clause_index = (base_offset + observed_category) * 2;
    //int neg_clause_index = pos_clause_index + 1;

    // Clamp clause state values to prevent unbounded growth
    //const int max_val = 256;
    //const int min_val = -256;
    const int max_val = 1024;
    const int min_val = -1024;

    if (label == 1) {
        atomicClampAdd(&d_clause_states[pos_clause_index], 5, min_val, max_val);
        atomicClampAdd(&d_clause_states[neg_clause_index], -3, min_val, max_val);
    }
    else {
        atomicClampAdd(&d_clause_states[pos_clause_index], -1, min_val, max_val);
        atomicClampAdd(&d_clause_states[neg_clause_index], 2, min_val, max_val);
    }
    */
}

__global__ void validateClausesKernel_withVotes(
    const int* d_numeric_data,
    const int* d_clause_states,
    int num_samples,
    int num_features,
    const int* d_num_categories,
    const int* d_num_categories_offsets,
    int num_literals,
    int num_clauses,
    int* d_predictions,
    int* d_votes_per_feature_per_sample
) {
    int block_id = blockIdx.z * gridDim.y * gridDim.x +
        blockIdx.y * gridDim.x +
        blockIdx.x;

    int thread_id_in_block = threadIdx.y * blockDim.x + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;

    int sample_id = block_id * threads_per_block + thread_id_in_block;
    if (sample_id >= num_samples) return;

    int total_vote_sum = 0;

    for (int feature_id = 0; feature_id < num_features; ++feature_id) {
        int feature_vote = 0;

        int observed_category = d_numeric_data[sample_id * num_features + feature_id];
        int num_categories = d_num_categories[feature_id];
        int base_offset = d_num_categories_offsets[feature_id];

        for (int cat = 0; cat < num_categories; ++cat) {
            int pos_clause_index = (base_offset + cat) * 2;
            int neg_clause_index = pos_clause_index + 1;

            int pos_val = d_clause_states[pos_clause_index];
            int neg_val = d_clause_states[neg_clause_index];

            if (cat == observed_category) {
                if (pos_val > 0) { feature_vote += pos_val; }
                if (neg_val > 0) { feature_vote -= neg_val; }
            } else {
                if (neg_val > 0) { feature_vote += neg_val; }
                if (pos_val > 0) { feature_vote -= pos_val; }
            }
        }

        total_vote_sum += feature_vote;
        d_votes_per_feature_per_sample[sample_id * num_features + feature_id] = feature_vote;
    }

    d_predictions[sample_id] = (total_vote_sum > 0) ? 1 : 0;

    if (sample_id < 20) {
        printf("Sample %d: total_vote_sum=%d → prediction=%d\n",
            sample_id, total_vote_sum, d_predictions[sample_id]);
    }
}
__global__ void validateClausesKernel_withVotes_parallel(
    const int* d_numeric_data,
    const int* d_clause_states,
    int num_samples,
    int num_features,
    const int* d_num_categories,
    const int* d_num_categories_offsets,
    int num_literals,
    int num_clauses,
    int* d_predictions,
    int* d_votes_per_feature_per_sample
) {
    int block_id = blockIdx.z * gridDim.y * gridDim.x +
        blockIdx.y * gridDim.x +
        blockIdx.x;

    int thread_id_in_block = threadIdx.y * blockDim.x + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;
    int thread_id = block_id * threads_per_block + thread_id_in_block;

    int total_threads = num_samples * num_features;
    if (thread_id >= total_threads) return;

    int sample_id = thread_id / num_features;
    int feature_id = thread_id % num_features;

    //if (sample_id < 5 && feature_id < 5) {
        //printf("[Thread %d] sample_id: %d, feature_id: %d\n", thread_id, sample_id, feature_id);
    //}

    int observed_category = d_numeric_data[sample_id * num_features + feature_id];
    int num_categories = d_num_categories[feature_id];
    int base_offset = d_num_categories_offsets[feature_id];

    //if (sample_id < 5 && feature_id < 5) {
        //printf(" Sample %d, Feature %d: observed_category=%d, num_categories=%d, base_offset=%d\n",
            //sample_id, feature_id, observed_category, num_categories, base_offset);
    //}

    int feature_vote = 0;

    for (int cat = 0; cat < num_categories; ++cat) {
        int pos_clause_index = (base_offset + cat) * 2;
        int neg_clause_index = pos_clause_index + 1;

        int pos_val = d_clause_states[pos_clause_index];
        int neg_val = d_clause_states[neg_clause_index];

        /*
        if (sample_id < 1 && feature_id < 1 && cat < 3) {
            printf("   Cat %d: POS[%d]=%d, NEG[%d]=%d\n",
                cat, pos_clause_index, pos_val, neg_clause_index, neg_val);
        }*/

        /*
        if (cat == observed_category) {
            if (pos_val > 0) { feature_vote += pos_val; }
            if (neg_val > 0) { feature_vote -= neg_val; }
        } else {
            if (neg_val > 0) { feature_vote += neg_val; }
            if (pos_val > 0) { feature_vote -= pos_val; }
        }
        */
        if (cat == observed_category) {
            if (pos_val > 0) { feature_vote += pos_val; }
            //if (neg_val > 0) { feature_vote -= neg_val; }
        }
        else {
            //if (neg_val > 0) { feature_vote += neg_val; }
            if (pos_val > 0) { feature_vote -= pos_val; }
        }
    }

    //d_votes_per_feature_per_sample[sample_id * num_features + feature_id] = feature_vote;

    atomicAdd(&d_predictions[sample_id], feature_vote);

}

__global__ void computeAccuracyKernel(
    const int* d_predictions,
    const int* d_labels,
    int num_samples,
    int* d_correct_count
) {
    extern __shared__ int shared_correct[];

    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    int is_correct = 0;
    if (global_id < num_samples) {
        int pred = (d_predictions[global_id] > 0) ? 1 : 0;
        if (pred == d_labels[global_id]) {
            is_correct = 1;
        }
    }

    // Store local result in shared memory
    shared_correct[tid] = is_correct;
    __syncthreads();

    // In-block reduction
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared_correct[tid] += shared_correct[tid + offset];
        }
        __syncthreads();
    }

    // Block master adds result to global counter
    if (tid == 0) {
        atomicAdd(d_correct_count, shared_correct[0]);
    }
}




void retrieveDeviceData2(
    int num_clauses, int num_literals, int num_samples, int num_features,
    int* d_clause_states,
    int* d_debug_shared_literals,
    int* d_numeric_data,
    int* d_labels,
    int* d_num_categories,
    vector<int>& clause_states,
    vector<int>& shared_literals_host,
    vector<int>& numeric_data_host,
    vector<int>& labels_host,
    vector<int>& num_categories_host
) {
    clause_states.resize(num_clauses);
    cudaMemcpy(clause_states.data(), d_clause_states, clause_states.size() * sizeof(int), cudaMemcpyDeviceToHost);

    shared_literals_host.resize(num_samples * num_literals);
    cudaMemcpy(shared_literals_host.data(), d_debug_shared_literals, shared_literals_host.size() * sizeof(int), cudaMemcpyDeviceToHost);

    numeric_data_host.resize(num_samples * num_features);
    cudaMemcpy(numeric_data_host.data(), d_numeric_data, numeric_data_host.size() * sizeof(int), cudaMemcpyDeviceToHost);

    labels_host.resize(num_samples);
    cudaMemcpy(labels_host.data(), d_labels, labels_host.size() * sizeof(int), cudaMemcpyDeviceToHost);

    num_categories_host.resize(num_features);  // Only for real features
    cudaMemcpy(num_categories_host.data(), d_num_categories, num_features * sizeof(int), cudaMemcpyDeviceToHost);
}

void printClauseStates(const vector<int>& clause_states) {
    cout << "Clause States: " << clause_states.size() << endl;
    for (size_t i = 0; i < clause_states.size(); ++i) {
        cout << "clause_states[" << i << "]: " << clause_states[i] << endl;
    }
}

void printNumericData(const vector<int>& numeric_data_host, int num_features, size_t num_samples) {
    cout << "Numeric Data:" << endl;
    for (size_t sample = 0; sample < num_samples; ++sample) {
        cout << "Sample " << sample << ": ";
        for (int feat = 0; feat < num_features; ++feat) {
            cout << numeric_data_host[sample * num_features + feat] << " ";
        }
        cout << endl;
    }
}

void printLabels(const vector<int>& labels_host) {
    cout << "Labels:" << endl;
    for (size_t i = 0; i < labels_host.size(); ++i) {
        cout << "labels[" << i << "]: " << labels_host[i] << endl;
    }
}

void printNumCategories(const vector<int>& num_categories_host) {
    cout << "Num Categories:" << endl;
    for (size_t i = 0; i < num_categories_host.size(); ++i) {
        cout << "num_categories[" << i << "]: " << num_categories_host[i] << endl;
    }
}


void printTestData2(
    const vector<int>& clause_states,
    const vector<int>& shared_literals_host,
    const vector<int>& numeric_data_host,
    const vector<int>& labels_host,
    const vector<int>& num_categories_host,
    int num_features,
    size_t num_samples,
    int num_literals
) {
    cout << "Clause States:" << endl;
    for (size_t i = 0; i < clause_states.size(); ++i) {
        cout << "clause_states[" << i << "]: " << clause_states[i] << endl;
    }

    cout << "Numeric Data:" << endl;
    for (size_t sample = 0; sample < num_samples; ++sample) {
        cout << "Sample " << sample << ": ";
        for (int feat = 0; feat < num_features; ++feat) {
            cout << numeric_data_host[sample * num_features + feat] << " ";
        }
        cout << endl;
    }

    cout << "Labels:" << endl;
    for (size_t i = 0; i < labels_host.size(); ++i) {
        cout << "labels[" << i << "]: " << labels_host[i] << endl;
    }

    cout << "Num Categories:" << endl;
    for (size_t i = 0; i < num_categories_host.size(); ++i) {
        cout << "num_categories[" << i << "]: " << num_categories_host[i] << endl;
    }
}


void computeLaunchConfig(int total_threads, int max_threads_per_block, dim3& grid, dim3& block) {
    
    int block_x = 32;
    int block_y = 32;

    while (block_x * block_y > 1024) {
        if (block_y > block_x) block_y--;
        else block_x--;
    }

    int threads_per_block = block_x * block_y;
    int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    block = dim3(block_x, block_y, 1);
    grid = dim3(num_blocks, 1, 1);
}


void createFilteredCSV(
    const string& original_path,
    const DatasetSchema& original_schema,
    const vector<int>& include_column_indices
) {
    // Step 1: Build set for fast lookup
    unordered_set<int> included(include_column_indices.begin(), include_column_indices.end());

    // Step 2: Open original file
    ifstream infile(original_path);
    if (!infile.is_open()) {
        cerr << "Failed to open file: " << original_path << endl;
        return;
    }

    // Step 3: Prepare output file path
    string modified_path = original_path;
    size_t dot_pos = modified_path.find_last_of('.');
    if (dot_pos != string::npos)
        modified_path.insert(dot_pos, "_modified");
    else
        modified_path += "_modified";

    ofstream outfile(modified_path);
    if (!outfile.is_open()) {
        cerr << "Failed to open output file: " << modified_path << endl;
        return;
    }

    string line;
    while (getline(infile, line)) {
        stringstream ss(line);
        string token;
        stringstream out_line;
        int col_idx = 0;
        int output_col_count = 0;

        getline(ss, token, ';');
        out_line << token;
        out_line << ";";
        while (getline(ss, token, ';')) {
            if (included.count(col_idx)) {
                if (output_col_count++ > 0) out_line << ";";
                out_line << token;
            }
            ++col_idx;
        }

        outfile << out_line.str() << "\n";
    }

    cout << "Filtered file written to: " << modified_path << endl;
}

void test_training_on_input_data_real(string& data_path_external) {
    cout << COLOR_YELLOW << "test_training_on_input_data_real" << COLOR_RESET << endl;

    try {
        float training_time_ms = 0, validation_time_ms = 0, accuracy_time_ms, accuracy = 0;
        
        DatasetSchema schema_in(20);
        schema_in.column_mapping = {
                {0, DatasetSchema::FLOAT}, {1, DatasetSchema::CATEGORY}, {2, DatasetSchema::CATEGORY},
                {3, DatasetSchema::CATEGORY}, {4, DatasetSchema::BOOL}, {5, DatasetSchema::CATEGORY},
                {6, DatasetSchema::CATEGORY}, {7, DatasetSchema::CATEGORY}, {8, DatasetSchema::FLOAT},
                {9, DatasetSchema::FLOAT}, {10, DatasetSchema::CATEGORY}, {11, DatasetSchema::CATEGORY},
                {12, DatasetSchema::CATEGORY}, {13, DatasetSchema::CATEGORY}, {14, DatasetSchema::CATEGORY},
                {15, DatasetSchema::BOOL}, {16, DatasetSchema::CATEGORY}, {17, DatasetSchema::CATEGORY},
                {18, DatasetSchema::CATEGORY}, {19, DatasetSchema::CATEGORY}
        };
        string data_path_in = "C:\\Users\\jdoe\\Downloads\\secondary_data.csv";
        DatasetSchema schema = schema_in;
        // string data_path = data_path_in;
        string data_path = data_path_external;
        //string data_path = "C:\\Users\\jdoe\\Downloads\\secondary_data_part.csv";
        //string data_path = "C:\\Users\\jdoe\\Downloads\\sample-data_2_rows_all.csv";
        //string data_path = "C:\\Users\\jdoe\\Downloads\\sample-data_2_rows_all_2.csv"; // 28.5714, rand 0
        //string data_path = "C:\\Users\\jdoe\\Downloads\\sample-data_2_rows_all_2.csv";
        //string data_path = "C:\\Users\\jdoe\\Downloads\\secondary_data_4k.csv"; // 
        //string data_path = "C:\\Users\\jdoe\\Downloads\\secondary_data_1k.csv"; // 
        //string data_path = "C:\\Users\\jdoe\\Downloads\\secondary_data_10k.csv"; //
        //string data_path = "C:\\Users\\jdoe\\Downloads\\secondary_data_4k.csv"; //
        //string data_path_in = "C:\\Users\\jdoe\\Downloads\\secondary_data_last_true.csv"; //
        //string data_path = "C:\\Users\\jdoe\\Downloads\\secondary_data_last_true_modified.csv";
        //string data_path = "C:\\Users\\jdoe\\Downloads\\secondary_data_last_true.csv"; //

        // just some columns...
        /*
        createFilteredCSV(data_path_in, schema_in, { 1,2,3 });
        DatasetSchema schema(3);
        schema.column_mapping = {
                {0, DatasetSchema::CATEGORY}, {1, DatasetSchema::CATEGORY},
                {2, DatasetSchema::CATEGORY}
        };
        */
        /*
        createFilteredCSV(data_path_in, schema_in, { 0, 8, 9});
        DatasetSchema schema(3);
        schema.column_mapping = {
                {0, DatasetSchema::FLOAT}, {1, DatasetSchema::FLOAT},
                {2, DatasetSchema::FLOAT}
        };
        */
        /*
        createFilteredCSV(data_path_in, schema_in, { 17, 18, 19 });
        DatasetSchema schema(3);
        schema.column_mapping = {
                {0, DatasetSchema::CATEGORY}, {1, DatasetSchema::CATEGORY},
                {2, DatasetSchema::CATEGORY}
        };
        */
        //createFilteredCSV(data_path_in, schema_in, { 19 });
        
        /*
        DatasetSchema schema(1);
        schema.column_mapping = {
                {0, DatasetSchema::CATEGORY}
        };
        */

        vector<vector<string>> raw_data;
        vector<string> raw_labels;
        loadData(raw_data, raw_labels, data_path);

        bool should_print_clauses = true;
        //bool should_print_clauses = false;
        //bool should_print_votes = true;
        bool should_print_votes = false;
        //bool should_print_data = true; 
        bool should_print_data = false;
        //bool should_print_labels = true; 
        bool should_print_labels = false;
        //bool should_print_results = true;
        bool should_print_results = false;
        //bool should_print = true;
        bool should_print = false;
        bool randomize = true;
        //bool randomize = false;



        float train_ratio = 0.8f;

        vector<vector<string>> train_raw, valid_raw;
        vector<string> train_labels_str, valid_labels_str;
        splitDataset(raw_data, raw_labels, train_ratio, &train_raw, &train_labels_str, &valid_raw, &valid_labels_str, schema, randomize);
        cout << "printDatasetSplits..." << endl;
        if (should_print_labels) {
            printDatasetSplits(train_raw, train_labels_str, valid_raw, valid_labels_str);
        }
        cout << "printDatasetSplits...train_labels_str: " << train_labels_str.size() << endl;

       
        computeStatisticsPlain(raw_data, schema);
        printStats(schema);

        vector<vector<float>> train_numeric_float, valid_numeric_float;
        convertDataToNumericTable(train_raw, schema, &train_numeric_float);
        convertDataToNumericTable(valid_raw, schema, &valid_numeric_float);
        computeStatisticsReduction(train_numeric_float, schema);
        printStats(schema);
        computeStatisticsCPU(raw_data, schema);
        printStats(schema);

        vector<vector<int>> train_numeric_int, valid_numeric_int;
        convertNumericFloatsToInts(train_numeric_float, schema, &train_numeric_int);
        convertNumericFloatsToInts(valid_numeric_float, schema, &valid_numeric_int);
        if (should_print) { printNumericIntData(train_numeric_int, valid_numeric_int); }

        vector<int> train_labels_int, valid_labels_int;
        map<string, int> label_mapping;
        convertLabelsToInt(train_labels_str, train_labels_int, label_mapping);
        convertLabelsToInt(valid_labels_str, valid_labels_int, label_mapping);

        int total_categories = 0;
        for (const auto& stat : schema.stats) {
            total_categories += stat.num_categories;
        }

        int num_literals = total_categories * 2;
        int num_clauses = num_literals;
        cout << "train_numeric_int: " << train_numeric_int.size() << endl;
        cout << "total_categories: " << total_categories << endl;

        int* d_numeric_data = nullptr;
        int* d_labels = nullptr;
        int* d_clause_states = nullptr;
        int* d_num_categories = nullptr;
        int* d_num_categories_offsets = nullptr;
        int* d_debug_shared_literals = nullptr;

        int num_samples = train_raw.size();
        int num_features = schema.column_mapping.size();

        vector<int> numeric_data_flat(num_samples * num_features);
        for (int i = 0; i < num_samples; ++i) {
            for (int j = 0; j < num_features; ++j) {
                numeric_data_flat[i * num_features + j] = train_numeric_int[i][j];
            }
        }

        cudaMalloc(&d_numeric_data, numeric_data_flat.size() * sizeof(int));
        cudaMemcpy(d_numeric_data, numeric_data_flat.data(), numeric_data_flat.size() * sizeof(int), cudaMemcpyHostToDevice);

        //std::cout << "train_labels_int: ";
        //for (int label : train_labels_int) {
            //std::cout << label <<  " ";
        //}
        //std::cout << std::endl;

        cudaMalloc(&d_labels, train_labels_int.size() * sizeof(int));
        cudaError_t err;
        CUDA_CHECK("cudaMalloc d_labels, train_labels_int");
        cudaMemcpy(d_labels, train_labels_int.data(), train_labels_int.size() * sizeof(int), cudaMemcpyHostToDevice);
        CUDA_CHECK("cudaMemcpy d_labels");

        std::cout << "train_labels_int.size(): " << train_labels_int.size() << std::endl;
        std::cout << "num_samples: " << num_samples << std::endl;
        vector<int> num_categories_host(schema.stats.size());
        for (size_t i = 0; i < schema.stats.size(); ++i) {
            num_categories_host[i] = schema.stats[i].num_categories;
        }
        cudaMalloc(&d_num_categories, num_categories_host.size() * sizeof(int));
        cudaMemcpy(d_num_categories, num_categories_host.data(), num_categories_host.size() * sizeof(int), cudaMemcpyHostToDevice);

        vector<int> d_num_categories_offsets_host(schema.stats.size());
        int tmp = 0;
        cout << "Tmp: " << tmp;
        for (size_t i = 0; i < schema.stats.size(); ++i) {
            d_num_categories_offsets_host[i] = tmp;
            tmp += schema.stats[i].num_categories;
            cout << " " << tmp << "(" << schema.stats[i].num_categories << ")";
        }
        cout << " " << endl;
        cudaMalloc(&d_num_categories_offsets, d_num_categories_offsets_host.size() * sizeof(int));
        cudaMemcpy(d_num_categories_offsets, d_num_categories_offsets_host.data(), d_num_categories_offsets_host.size() * sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc(&d_clause_states, num_literals * sizeof(int));
        cudaMemset(d_clause_states, 0, num_literals * sizeof(int));
        cudaMalloc(&d_debug_shared_literals, num_samples * num_literals * sizeof(int));

        int max_thread_per_SM = 1024; // 2048;

        dim3 grid_train, threads_train;
        int total_threads = num_samples * num_features;
        computeLaunchConfig(total_threads, max_thread_per_SM, grid_train, threads_train);
        cout << "num_clauses: " << num_clauses << endl;
        cout << "num_samples: " << num_samples << endl;
        cout << "num_features: " << num_features << endl;
        cout << "total_threads: " << total_threads << endl;
        cout << "num_literals: " << num_literals << endl;
        //size_t shared_mem_size = num_literals * sizeof(int);
        size_t shared_mem_size = 0;
        printCudaLaunchConfig(grid_train, threads_train, shared_mem_size, num_samples);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        trainClausesKernel << <grid_train, threads_train >> > (
            d_numeric_data, d_labels,
            num_samples, num_features,
            d_num_categories, d_num_categories_offsets,
            num_literals, num_clauses,
            d_clause_states,
            d_debug_shared_literals
        );
        cudaEventRecord(stop);

        CUDA_CHECK("Kernel failed");

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&training_time_ms, start, stop);
        cout << "Training completed in " << training_time_ms << " ms." << endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cout << "num_clauses: " << num_clauses << endl;

        vector<int> clause_states, shared_literals_host, numeric_data_host, labels_host;


        retrieveDeviceData2(
            num_clauses, num_literals, train_numeric_int.size(), schema.column_mapping.size(),
            d_clause_states, d_debug_shared_literals, d_numeric_data, d_labels, d_num_categories,
            clause_states, shared_literals_host, numeric_data_host, labels_host, num_categories_host
        );

        if (should_print || should_print_data) {
            printTestData2(clause_states, shared_literals_host, numeric_data_host, labels_host, num_categories_host, schema.column_mapping.size(), train_numeric_int.size(), num_literals);
        }
        if (should_print || should_print_clauses) {
            printClauseStates(clause_states);
            //printNumericData(numeric_data_host, num_features, num_samples);
            //printLabels(labels_host);
            //printNumCategories(num_categories_host);
        }

        int num_samples_valid = valid_numeric_int.size();
        vector<int> numeric_data_flat_valid(num_samples_valid * num_features);
        cout << "mapping valid data, num_samples_valid: " << num_samples_valid << endl;

        for (int i = 0; i < num_samples_valid; ++i) {
            for (int j = 0; j < num_features; ++j) {
                numeric_data_flat_valid[i * num_features + j] = valid_numeric_int[i][j];
            }
        }
        if (should_print) {
            for (int i = 0; i < num_samples_valid; ++i) {
                for (int j = 0; j < num_features; ++j) {
                    //numeric_data_flat_valid[i * num_features + j] = valid_numeric_int[i][j];
                    cout << valid_numeric_int[i][j] << " ";
                }
                cout << endl;
            }
        }


        int* d_numeric_data_valid = nullptr;
        cudaMalloc(&d_numeric_data_valid, numeric_data_flat_valid.size() * sizeof(int));
        
        CUDA_CHECK("malloc d_numeric_data_valid");
        cudaMemcpy(d_numeric_data_valid, numeric_data_flat_valid.data(), numeric_data_flat_valid.size() * sizeof(int), cudaMemcpyHostToDevice);

        CUDA_CHECK("malloc d_numeric_data_valid");

        //cudaFree(d_labels); // freed later
        int* d_predictions = nullptr;
        cudaMalloc(&d_predictions, num_samples_valid * sizeof(int));
        cudaMemset(d_predictions, 0, num_samples_valid * sizeof(int));

        CUDA_CHECK("malloc d_predictions");

        int* d_votes_per_feature_per_sample = nullptr;
        size_t d_votes_per_feature_per_sample_size = num_samples_valid * num_features * sizeof(int);
        cudaMalloc(&d_votes_per_feature_per_sample, d_votes_per_feature_per_sample_size);

        CUDA_CHECK("malloc d_votes_per_feature_per_sample");
        int total_threads_valid = num_samples_valid * num_features;

        cout << "num_samples_valid: " << num_samples_valid << endl;
        cout << "num_clauses: " << num_clauses << endl;
        cout << "total_threads_valid: " << total_threads_valid << endl;
        dim3 grid_valid, threads_valid;
      
        bool use_acc_kernel = true;
        //bool use_acc_kernel = false;
        if (use_acc_kernel) {
            cout << "validateClausesKernel_withVotes" << endl;

            int* d_correct_count;
            cudaMalloc(&d_correct_count, sizeof(int));
            cudaMemset(d_correct_count, 0, sizeof(int));
            int* d_labels_valid;

            cudaMalloc(&d_labels_valid, valid_labels_int.size() * sizeof(int));
            cudaMemcpy(d_labels_valid, valid_labels_int.data(), valid_labels_int.size() * sizeof(int), cudaMemcpyHostToDevice);

            computeLaunchConfig(total_threads_valid, max_thread_per_SM, grid_valid, threads_valid);
            printCudaLaunchConfig(grid_valid, threads_valid, shared_mem_size, total_threads);

            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            validateClausesKernel_withVotes_parallel << <grid_valid, threads_valid, 0 >> > (
                d_numeric_data_valid,
                d_clause_states,
                num_samples_valid,
                num_features,
                d_num_categories,
                d_num_categories_offsets,
                num_literals,
                num_clauses,
                d_predictions,
                d_votes_per_feature_per_sample
            );
            
            cudaEventRecord(stop);
            
            CUDA_CHECK("Kernel failed");

            
            std::vector<int> votes_per_feature_per_sample_host(0);
            if (should_print || should_print_votes) {
                votes_per_feature_per_sample_host.resize(num_samples_valid * num_features);
                cudaMemcpy(votes_per_feature_per_sample_host.data(), d_votes_per_feature_per_sample, votes_per_feature_per_sample_host.size() * sizeof(int), cudaMemcpyDeviceToHost);
            }

            dim3 grid_acc, threads_acc;
            int threads_per_block = max_thread_per_SM; // 1024;
            int blocks = (num_samples_valid + threads_per_block - 1) / threads_per_block;
            size_t shared_mem_size = threads_per_block * sizeof(int);
            cout << "Acc threads_per_block: " << threads_per_block << endl;
            cout << "Acc blocks: " << blocks << endl;
            cout << "Acc shared_mem_size: " << shared_mem_size << endl;

            computeAccuracyKernel << <blocks, threads_per_block, shared_mem_size >> > (
                d_predictions, d_labels_valid, num_samples_valid, d_correct_count
            );
            
            cudaEventSynchronize(stop);

            cudaEventElapsedTime(&validation_time_ms, start, stop);
            //cout << "Validation completed in " << validation_time_ms << " ms." << endl; // 2 for loops 12.8704 ms.

            int correct_gpu = 0;
            cudaMemcpy(&correct_gpu, d_correct_count, sizeof(int), cudaMemcpyDeviceToHost);

            if (should_print || should_print_votes) {
                for (int i = 0; i < num_samples_valid; ++i) {
                    std::cout << "Sample " << i << " votes per feature: ";
                    for (int f = 0; f < num_features; ++f) {
                        std::cout << votes_per_feature_per_sample_host[i * num_features + f] << " ";
                    }
                    //std::cout << " - pred: " << predictions_host[i];
                    std::cout << "\n";
                }
            }

            cout << "Validation and Accuracy completed in " << validation_time_ms << " ms." << endl;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            accuracy = (static_cast<float>(correct_gpu) / num_samples_valid) * 100.0f;

        } else {

            cout << "validateClausesKernel_withVotes" << endl;

            computeLaunchConfig(num_samples_valid, 1024, grid_valid, threads_valid);
            printCudaLaunchConfig(grid_valid, threads_valid, shared_mem_size, num_samples_valid);

            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            validateClausesKernel_withVotes << <grid_valid, threads_valid, 0 >> > (
                d_numeric_data_valid,
                d_clause_states,
                num_samples_valid,
                num_features,
                d_num_categories,
                d_num_categories_offsets,
                num_literals,
                num_clauses,
                d_predictions,
                d_votes_per_feature_per_sample
            );
            cudaDeviceSynchronize(); // printf debugging ...
            cout << "cudaDeviceSynchronize - done" << endl;

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&validation_time_ms, start, stop);
            cout << "Validation completed in " << validation_time_ms << " ms." << endl; // 2 for loops 12.8704 ms.
            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            CUDA_CHECK("Kernel failed");

            cudaEventSynchronize(stop);

            std::vector<int> votes_per_feature_per_sample_host(num_samples_valid * num_features);
            cudaMemcpy(votes_per_feature_per_sample_host.data(), d_votes_per_feature_per_sample, votes_per_feature_per_sample_host.size() * sizeof(int), cudaMemcpyDeviceToHost);

            if (should_print || should_print_votes) {
                for (int i = 0; i < num_samples_valid; ++i) {
                    std::cout << "Sample " << i << " votes per feature: ";
                    for (int f = 0; f < num_features; ++f) {
                        std::cout << votes_per_feature_per_sample_host[i * num_features + f] << " ";
                    }
                    std::cout << "\n";
                }
            }
        
            vector<int> predictions_host(num_samples_valid);
            cudaMemcpy(predictions_host.data(), d_predictions, num_samples_valid * sizeof(int), cudaMemcpyDeviceToHost);

            int correct = 0;
            for (int i = 0; i < num_samples_valid; ++i) {
                if (predictions_host[i] == labels_host[i]) {
                    correct++;
                }
            }

            accuracy = (static_cast<float>(correct) / num_samples_valid) * 100.0f;


            cout << "Validation accuracy: " << accuracy << "%\n";
            if (should_print || should_print_results) {
                for (int i = 0; i < num_samples_valid; ++i) {
                    std::cout << "Sample " << i << ": Prediction = " << predictions_host[i]
                        << ", Label = " << labels_host[i] << "\n";
                }
            }
        
        }


        cudaFree(d_predictions);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaFree(d_numeric_data);
        cudaFree(d_labels);
        cudaFree(d_num_categories);
        cudaFree(d_clause_states);
        //cudaFree(d_shared_clauses); // d_debug_shared_literals

        //performTraining_clean(
            //d_numeric_data, d_labels, d_clause_states, d_num_categories, d_debug_shared_literals
        //);

        if (accuracy >= 0.0f && accuracy <= 100.0f) {
            cout << "Test passed with valid accuracy." << endl;
            cout << "Validation accuracy: " << accuracy << "%\n";
        }
        else {
            cout << "Test failed, accuracy out of expected bounds." << endl;
        }
    }
    catch (const exception& e) {
        cout << "Test failed: " << e.what() << endl;
    }
    cout << COLOR_YELLOW << "test_training_on_input_data_real - done" << COLOR_RESET << endl;

}

int main(int argc, char* argv[]) {
    string data_path;
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <data_path>" << endl;
        //return 1;
        data_path = "C:\\Users\\jdoe\\Downloads\\secondary_data.csv";
        cout << "Using file from default path: " << data_path << endl;
    } else {
        data_path = argv[1];
    }

    test_training_on_input_data_real(data_path);

    return 0;
}

/**
Usage: C:\Users\jdoe\Downloads\pa2\out\sample\CUDA_12_8_VS2022\CUDA_12_8_VS2022\bin\x64\Release\TemplateProject.exe <data_path>
Using file from default path: C:\Users\jdoe\Downloads\secondary_data.csv
test_training_on_input_data_real
printDatasetSplits...
printDatasetSplits...train_labels_str: 48855
[LOG] Running computeStatistics on GPU...
[LOG] Temporary CPU-side arrays to transfer data to GPU...
[LOG] Temporary CPU-side arrays to transfer data to GPU 2...
[LOG] CPU/GPU-based statistics computed successfully.
Column    Type        Float Min      Float Max      Float Min Diff Int Min        Int Min Diff   Int Max        Bool True      Bool False     Categories
-------------------------------------------------------------------------------------------------------------------------------
0         FLOAT       340282346638528859811704183484516925440.000000-340282346638528859811704183484516925440.000000340282346638528859811704183484516925440.000000N/A            N/A            N/A            N/A            N/A            0
1         CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            b(0) c(1) f(2) o(3) p(4) s(5) x(6)
2         CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            (0) d(1) e(2) g(3) h(4) i(5) k(6) l(7) s(8) t(9) w(10) y(11)
3         CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            b(0) e(1) g(2) k(3) l(4) n(5) o(6) p(7) r(8) u(9) w(10) y(11)
4         BOOL        N/A            N/A            N/A            N/A            N/A            N/A            10590          50479          2
5         CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            (0) a(1) d(2) e(3) f(4) p(5) s(6) x(7)
6         CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            (0) c(1) d(2) f(3)
7         CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            b(0) e(1) f(2) g(3) k(4) n(5) o(6) p(7) r(8) u(9) w(10) y(11)
8         FLOAT       340282346638528859811704183484516925440.000000-340282346638528859811704183484516925440.000000340282346638528859811704183484516925440.000000N/A            N/A            N/A            N/A            N/A            0
9         FLOAT       340282346638528859811704183484516925440.000000-340282346638528859811704183484516925440.000000340282346638528859811704183484516925440.000000N/A            N/A            N/A            N/A            N/A            0
10        CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            (0) b(1) c(2) f(3) r(4) s(5)
11        CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            (0) f(1) g(2) h(3) i(4) k(5) s(6) t(7) y(8)
12        CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            b(0) e(1) f(2) g(3) k(4) l(5) n(6) o(7) p(8) r(9) u(10) w(11) y(12)
13        CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            (0) u(1)
14        CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            (0) e(1) k(2) n(3) u(4) w(5) y(6)
15        BOOL        N/A            N/A            N/A            N/A            N/A            N/A            15179          45890          2
16        CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            (0) e(1) f(2) g(3) l(4) m(5) p(6) r(7) z(8)
17        CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            (0) g(1) k(2) n(3) p(4) r(5) u(6) w(7)
18        CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            d(0) g(1) h(2) l(3) m(4) p(5) u(6) w(7)
19        CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            a(0) s(1) u(2) w(3)
Column    Type        Float Min      Float Max      Float Min Diff Int Min        Int Min Diff   Int Max        Bool True      Bool False     Categories
-------------------------------------------------------------------------------------------------------------------------------
0         FLOAT       0.440000       62.340000      6.190000       N/A            N/A            N/A            N/A            N/A            10
1         CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            b(0) c(1) f(2) o(3) p(4) s(5) x(6)
2         CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            (0) d(1) e(2) g(3) h(4) i(5) k(6) l(7) s(8) t(9) w(10) y(11)
3         CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            b(0) e(1) g(2) k(3) l(4) n(5) o(6) p(7) r(8) u(9) w(10) y(11)
4         BOOL        N/A            N/A            N/A            N/A            N/A            N/A            10590          50479          2
5         CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            (0) a(1) d(2) e(3) f(4) p(5) s(6) x(7)
6         CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            (0) c(1) d(2) f(3)
7         CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            b(0) e(1) f(2) g(3) k(4) n(5) o(6) p(7) r(8) u(9) w(10) y(11)
8         FLOAT       0.000000       31.799999      3.180000       N/A            N/A            N/A            N/A            N/A            10
9         FLOAT       0.000000       103.910004     10.391001      N/A            N/A            N/A            N/A            N/A            10
10        CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            (0) b(1) c(2) f(3) r(4) s(5)
11        CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            (0) f(1) g(2) h(3) i(4) k(5) s(6) t(7) y(8)
12        CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            b(0) e(1) f(2) g(3) k(4) l(5) n(6) o(7) p(8) r(9) u(10) w(11) y(12)
13        CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            (0) u(1)
14        CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            (0) e(1) k(2) n(3) u(4) w(5) y(6)
15        BOOL        N/A            N/A            N/A            N/A            N/A            N/A            15179          45890          2
16        CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            (0) e(1) f(2) g(3) l(4) m(5) p(6) r(7) z(8)
17        CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            (0) g(1) k(2) n(3) p(4) r(5) u(6) w(7)
18        CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            d(0) g(1) h(2) l(3) m(4) p(5) u(6) w(7)
19        CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            a(0) s(1) u(2) w(3)
Column    Type        Float Min      Float Max      Float Min Diff Int Min        Int Min Diff   Int Max        Bool True      Bool False     Categories
-------------------------------------------------------------------------------------------------------------------------------
0         FLOAT       0.440000       62.340000      6.190000       N/A            N/A            N/A            N/A            N/A            10
1         CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            b(3) c(4) f(1) o(6) p(2) s(5) x(0)
2         CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            (2) d(8) e(5) g(0) h(1) i(10) k(11) l(7) s(6) t(3) w(9) y(4)
3         CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            b(9) e(1) g(3) k(11) l(10) n(2) o(0) p(7) r(4) u(8) w(5) y(6)
4         BOOL        N/A            N/A            N/A            N/A            N/A            N/A            10590          50479          2
5         CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            (1) a(2) d(3) e(0) f(7) p(6) s(4) x(5)
6         CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            (0) c(1) d(2) f(3)
7         CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            b(4) e(8) f(11) g(5) k(10) n(1) o(9) p(2) r(7) u(3) w(0) y(6)
8         FLOAT       0.000000       31.799999      3.180000       N/A            N/A            N/A            N/A            N/A            10
9         FLOAT       0.000000       103.910004     10.391001      N/A            N/A            N/A            N/A            N/A            10
10        CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            (1) b(2) c(4) f(5) r(3) s(0)
11        CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            (1) f(8) g(7) h(5) i(4) k(3) s(2) t(6) y(0)
12        CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            b(4) e(8) f(12) g(10) k(9) l(5) n(2) o(11) p(7) r(6) u(3) w(0) y(1)
13        CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            (1) u(0)
14        CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            (2) e(4) k(6) n(3) u(5) w(0) y(1)
15        BOOL        N/A            N/A            N/A            N/A            N/A            N/A            15179          45890          2
16        CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            (6) e(2) f(4) g(0) l(3) m(5) p(1) r(7) z(8)
17        CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            (0) g(7) k(3) n(6) p(2) r(4) u(5) w(1)
18        CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            d(0) g(2) h(3) l(4) m(1) p(5) u(7) w(6)
19        CATEGORY    N/A            N/A            N/A            N/A            N/A            N/A            N/A            N/A            a(2) s(3) u(1) w(0)
train_numeric_int: 48855
total_categories: 155
train_labels_int.size(): 48855
num_samples: 48855
Tmp: 0 10(10) 17(7) 29(12) 41(12) 43(2) 51(8) 55(4) 67(12) 77(10) 87(10) 93(6) 102(9) 115(13) 117(2) 124(7) 126(2) 135(9) 143(8) 151(8) 155(4)
num_clauses: 310
num_samples: 48855
num_features: 20
total_threads: 977100
num_literals: 310
[CUDA LAUNCH CONFIGURATION]
num_samples: 48855
Grid dimensions: (955, 1, 1)
Threads per block: (32, 32, 1)
Shared memory size per block: 0 bytes
Training completed in 1.1017 ms.
num_clauses: 310
Clause States: 310
clause_states[0]: 8842
clause_states[1]: 7313
clause_states[2]: 51806
clause_states[3]: -11030
clause_states[4]: 15817
clause_states[5]: -5370
clause_states[6]: 3796
clause_states[7]: -1313
clause_states[8]: 288
clause_states[9]: -58
clause_states[10]: 17
clause_states[11]: -6
clause_states[12]: 25
clause_states[13]: -15
clause_states[14]: 215
clause_states[15]: -129
clause_states[16]: 775
clause_states[17]: -465
clause_states[18]: 360
clause_states[19]: -216
clause_states[20]: 1529
clause_states[21]: 4061
clause_states[22]: 2301
clause_states[23]: -213
clause_states[24]: 20509
clause_states[25]: -4548
clause_states[26]: 1109
clause_states[27]: 2262
clause_states[28]: 5456
clause_states[29]: -2127
clause_states[30]: 10109
clause_states[31]: -1696
clause_states[32]: 40948
clause_states[33]: -9040
clause_states[34]: 20828
clause_states[35]: -4318
clause_states[36]: 6319
clause_states[37]: -1123
clause_states[38]: 3866
clause_states[39]: -837
clause_states[40]: 5672
clause_states[41]: -298
clause_states[42]: 8817
clause_states[43]: -2675
clause_states[44]: 75
clause_states[45]: 2020
clause_states[46]: -1003
clause_states[47]: 3014
clause_states[48]: 2269
clause_states[49]: -576
clause_states[50]: 14876
clause_states[51]: -5259
clause_states[52]: 6948
clause_states[53]: 1924
clause_states[54]: 2383
clause_states[55]: 8
clause_states[56]: 10901
clause_states[57]: -3175
clause_states[58]: 3705
clause_states[59]: -1929
clause_states[60]: 1012
clause_states[61]: 2932
clause_states[62]: 7449
clause_states[63]: -2123
clause_states[64]: 1404
clause_states[65]: 27
clause_states[66]: 1537
clause_states[67]: -498
clause_states[68]: 40195
clause_states[69]: -10964
clause_states[70]: 2280
clause_states[71]: 1516
clause_states[72]: 858
clause_states[73]: 895
clause_states[74]: -480
clause_states[75]: 2080
clause_states[76]: 1600
clause_states[77]: 251
clause_states[78]: 11931
clause_states[79]: -2750
clause_states[80]: 10460
clause_states[81]: -732
clause_states[82]: 66555
clause_states[83]: -8412
clause_states[84]: 15396
clause_states[85]: -2883
clause_states[86]: 9148
clause_states[87]: 1605
clause_states[88]: 12724
clause_states[89]: 1243
clause_states[90]: 11566
clause_states[91]: 3
clause_states[92]: 12425
clause_states[93]: -5082
clause_states[94]: 3837
clause_states[95]: 47
clause_states[96]: 15510
clause_states[97]: -7353
clause_states[98]: 7418
clause_states[99]: -920
clause_states[100]: 9323
clause_states[101]: -838
clause_states[102]: 32668
clause_states[103]: -3869
clause_states[104]: 29602
clause_states[105]: -1538
clause_states[106]: 15844
clause_states[107]: -5935
clause_states[108]: 3837
clause_states[109]: 47
clause_states[110]: 2598
clause_states[111]: -1283
clause_states[112]: 547
clause_states[113]: 530
clause_states[114]: 3837
clause_states[115]: 47
clause_states[116]: 7156
clause_states[117]: -2132
clause_states[118]: 2611
clause_states[119]: 35
clause_states[120]: 6645
clause_states[121]: 3454
clause_states[122]: 3659
clause_states[123]: -318
clause_states[124]: 7178
clause_states[125]: -356
clause_states[126]: 1344
clause_states[127]: 168
clause_states[128]: 1318
clause_states[129]: -137
clause_states[130]: 33751
clause_states[131]: -10816
clause_states[132]: 11307
clause_states[133]: -487
clause_states[134]: -1050
clause_states[135]: 2100
clause_states[136]: 12261
clause_states[137]: 4164
clause_states[138]: 52249
clause_states[139]: -15248
clause_states[140]: 11515
clause_states[141]: -924
clause_states[142]: 2499
clause_states[143]: 427
clause_states[144]: 1663
clause_states[145]: -372
clause_states[146]: 989
clause_states[147]: -347
clause_states[148]: 835
clause_states[149]: -501
clause_states[150]: 605
clause_states[151]: -363
clause_states[152]: 305
clause_states[153]: -183
clause_states[154]: 8645
clause_states[155]: 7021
clause_states[156]: 39389
clause_states[157]: -6006
clause_states[158]: 27662
clause_states[159]: -11532
clause_states[160]: 3186
clause_states[161]: 565
clause_states[162]: 959
clause_states[163]: -154
clause_states[164]: 651
clause_states[165]: -315
clause_states[166]: 629
clause_states[167]: -376
clause_states[168]: 535
clause_states[169]: -321
clause_states[170]: 330
clause_states[171]: -198
clause_states[172]: 35
clause_states[173]: -21
clause_states[174]: 72506
clause_states[175]: -12278
clause_states[176]: 7551
clause_states[177]: -3349
clause_states[178]: -545
clause_states[179]: 1090
clause_states[180]: -829
clause_states[181]: 1658
clause_states[182]: -1128
clause_states[183]: 2256
clause_states[184]: 4406
clause_states[185]: -678
clause_states[186]: 59305
clause_states[187]: -13848
clause_states[188]: -829
clause_states[189]: 1658
clause_states[190]: -1392
clause_states[191]: 2784
clause_states[192]: -419
clause_states[193]: 838
clause_states[194]: 4278
clause_states[195]: 551
clause_states[196]: 1171
clause_states[197]: 493
clause_states[198]: 12309
clause_states[199]: -4619
clause_states[200]: 3833
clause_states[201]: -708
clause_states[202]: 3695
clause_states[203]: 1556
clause_states[204]: 705
clause_states[205]: -423
clause_states[206]: 967
clause_states[207]: 1104
clause_states[208]: -829
clause_states[209]: 1658
clause_states[210]: 5559
clause_states[211]: -2172
clause_states[212]: 250
clause_states[213]: 585
clause_states[214]: 342
clause_states[215]: -75
clause_states[216]: 20741
clause_states[217]: -434
clause_states[218]: 2350
clause_states[219]: 67
clause_states[220]: -152
clause_states[221]: 1088
clause_states[222]: 59
clause_states[223]: 463
clause_states[224]: 1397
clause_states[225]: 230
clause_states[226]: 44952
clause_states[227]: -16012
clause_states[228]: 5610
clause_states[229]: 2626
clause_states[230]: 79406
clause_states[231]: -12127
clause_states[232]: 2545
clause_states[233]: 832
clause_states[234]: 72580
clause_states[235]: -10368
clause_states[236]: -139
clause_states[237]: 278
clause_states[238]: -278
clause_states[239]: 556
clause_states[240]: -409
clause_states[241]: 818
clause_states[242]: -286
clause_states[243]: 572
clause_states[244]: 8323
clause_states[245]: -1855
clause_states[246]: 2160
clause_states[247]: -1296
clause_states[248]: 65183
clause_states[249]: -11499
clause_states[250]: 16768
clause_states[251]: 204
clause_states[252]: 4789
clause_states[253]: -1682
clause_states[254]: 3045
clause_states[255]: -287
clause_states[256]: 64916
clause_states[257]: -8970
clause_states[258]: 2353
clause_states[259]: -828
clause_states[260]: 3150
clause_states[261]: -1253
clause_states[262]: 1490
clause_states[263]: -894
clause_states[264]: 1662
clause_states[265]: -202
clause_states[266]: 2226
clause_states[267]: -539
clause_states[268]: -1680
clause_states[269]: 3360
clause_states[270]: 78469
clause_states[271]: -14320
clause_states[272]: 1360
clause_states[273]: -816
clause_states[274]: 75
clause_states[275]: 1894
clause_states[276]: -848
clause_states[277]: 1696
clause_states[278]: 712
clause_states[279]: 606
clause_states[280]: -131
clause_states[281]: 262
clause_states[282]: -147
clause_states[283]: 294
clause_states[284]: 2461
clause_states[285]: -911
clause_states[286]: 61586
clause_states[287]: -10080
clause_states[288]: 5521
clause_states[289]: 2804
clause_states[290]: 1851
clause_states[291]: 330
clause_states[292]: 6812
clause_states[293]: -2725
clause_states[294]: 4602
clause_states[295]: -1084
clause_states[296]: -291
clause_states[297]: 582
clause_states[298]: 440
clause_states[299]: -264
clause_states[300]: 1430
clause_states[301]: -858
clause_states[302]: 37492
clause_states[303]: -3129
clause_states[304]: 5142
clause_states[305]: -1772
clause_states[306]: 28032
clause_states[307]: -1926
clause_states[308]: 11285
clause_states[309]: -4468
mapping valid data, num_samples_valid: 12214
num_samples_valid: 12214
num_clauses: 310
total_threads_valid: 244280
validateClausesKernel_withVotes
[CUDA LAUNCH CONFIGURATION]
num_samples: 977100
Grid dimensions: (239, 1, 1)
Threads per block: (32, 32, 1)
Shared memory size per block: 0 bytes
Acc threads_per_block: 1024
Acc blocks: 12
Acc shared_mem_size: 4096 B
Validation and Accuracy completed in 0.475136 ms.
Test passed with valid accuracy.
Validation accuracy: 63.4682%
test_training_on_input_data_real - done

*/
