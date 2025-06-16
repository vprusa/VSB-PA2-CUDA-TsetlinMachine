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


// these libraries were not working for finding min/max of float values on GPU, but they shuold
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>


#include <chrono>


using namespace std;


// Column metainfo storage structure
struct DatasetStats {
    float float_min, float_max, float_min_diff;
    int int_min, int_max, int_min_diff;
    int bool_true_count, bool_false_count;
    int float_expected_categories;
    map<string, int> category_values;
    DatasetStats()
            : float_min(FLT_MAX), float_max(-FLT_MAX), float_min_diff(FLT_MAX),
              int_min(INT_MAX), int_max(INT_MIN), int_min_diff(INT_MAX),
              bool_true_count(0), bool_false_count(0), float_expected_categories(0) {
    }
};


// Dataset schema used for holding information about input data file columns, mappings and status
struct DatasetSchema {
    enum ColumnType { BOOL, CATEGORY, INTEGER, FLOAT };
    vector<pair<int, ColumnType>> column_mapping;
    vector<DatasetStats> stats;
    DatasetSchema(size_t count) : stats(count) {}
};


// Note. used for debugging, try-catche was perf unfreindly.
float safeStof(const string& str) {
    try {
        return stof(str);
    }
    catch (...) {
        return 0.0f;
    }
}


// loads data from CSV file
void loadData(
        vector<vector<string>>& data,
        vector<string>& labels,
        const string& filename
) {
    ifstream file(filename);
    string line, cell;
    if (!getline(file, line)) {
        cerr << "Error: empty file or cannot read header\n";
        return;
    }
    while (getline(file, line)) {
        vector<string> row;
        stringstream lineStream(line);
        // first column is label (target feature)
        if (getline(lineStream, cell, ';')) labels.push_back(cell);
        // rest is data to be train on...s
        while (getline(lineStream, cell, ';')) {
            row.push_back(cell);
        }
        data.push_back(row);
    }
}


// CPU for converting target features to binary labels
vector<int> convertLabels(const vector<string>& raw_labels) {
    vector<int> labels(raw_labels.size());
    for (size_t i = 0; i < raw_labels.size(); ++i) {
        labels[i] = (raw_labels[i] == "p") ? 1 : 0;
    }
    return labels;
}
/*
// Kernel: Compute statistics for dataset (2D Parallelism: columns × samples)
__global__ void computeStatisticsKernel(
   	const float* data, // input data
   	int num_samples, // number of samples
   	int num_features, // number of features
   	float* min_vals, // output min values
   	float* max_vals, // output max values
   	float* min_diffs // output min differences
) {
   // First dimension parallelism: columns/features
   int col = blockIdx.x * blockDim.x + threadIdx.x;
   if (col >= num_features) return;
   extern __shared__ float shared_data[];
   // Load column data to shared memory
   for (int row = threadIdx.y; row < num_samples; row += blockDim.y) {
   	shared_data[row] = data[row * num_features + col];
   }
   __syncthreads();
   // Initialize min and max values per feature
   float min_val = shared_data[0];
   float max_val = shared_data[0];
   // Compute min/max per feature
   for (int row = 1; row < num_samples; ++row) {
   	min_val = fminf(min_val, shared_data[row]);
   	max_val = fmaxf(max_val, shared_data[row]);
   }
   min_vals[col] = min_val;
   max_vals[col] = max_val;
   // Compute minimal non-zero difference
   float min_diff = max_val - min_val;
   for (int i = 0; i < num_samples; ++i) {
   	for (int j = i + 1; j < num_samples; ++j) {
       	float diff = fabsf(shared_data[i] - shared_data[j]);
       	if (diff > 0 && diff < min_diff) min_diff = diff;
   	}
   }
   min_diffs[col] = min_diff;
}*/


// kernel: self explanatory function name
__device__ float atomicMinFloat(
        float* address, // address of the float to be modified
        float val
) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}


// kernel: self explanatory function name
__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}


// kernel: calcs min and max values of input data
__global__ void reduceMinMax(
        const float* data, // input data
        int size, // size of data
        float* min_val, // resulting min
        float* max_val // resulting max
) {
    extern __shared__ float shared[];
    float* shared_min = shared;
    float* shared_max = shared + blockDim.x;


    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;


    // Initialize shared memory
    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;


    // Step 1: Each thread finds min/max within its assigned data segment
    if (idx < size) {
        float val = data[idx];
        local_min = val;
        local_max = val;
    }


    shared_min[tid] = local_min;
    shared_max[tid] = local_max;
    __syncthreads();


    // Step 2: Reduce within block to find block-level min/max... could use parallelization.
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride && idx + stride < size) {
            shared_min[tid] = fminf(shared_min[tid], shared_min[tid + stride]);
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }


    // Step 3: Write results from each block to global memory
    if (tid == 0) {
        atomicMinFloat(min_val, shared_min[0]);
        atomicMaxFloat(max_val, shared_max[0]);
    }
}


// statistics are necessary to update schema with min and max values for each category
//   TODO This could be used to store prepare the data in GPU RAM right away, but...
void computeStatistics(
        const vector<vector<string>>& raw_data, // input data
        DatasetSchema& schema // schema to be updated
) {
    cout << "[LOG] Running computeStatistics on GPU...\n";


    int num_rows = raw_data.size();
    int num_cols = schema.column_mapping.size();


    // Initialize statistics
    for (auto& stat : schema.stats) {
        stat.bool_true_count = stat.bool_false_count = 0;
        stat.category_values.clear();
        stat.int_min = INT_MAX; stat.int_max = INT_MIN; stat.int_min_diff = INT_MAX;
        stat.float_min = FLT_MAX; stat.float_max = -FLT_MAX; stat.float_min_diff = FLT_MAX;
    }


    cout << "[LOG] Temporary CPU-side arrays to transfer data to GPU...\n";
    vector<float> h_float_data(num_rows);
    vector<int> h_int_data(num_rows);


    for (int col = 0; col < num_cols; ++col) {
        int col_idx = schema.column_mapping[col].first;
        DatasetSchema::ColumnType col_type = schema.column_mapping[col].second;


        switch (col_type) {
            case DatasetSchema::FLOAT: {
                for (int i = 0; i < num_rows; ++i) {
                    h_float_data[i] = stof(raw_data[i][col_idx]);
                }


                float* d_float_data;
                cudaMalloc(&d_float_data, num_rows * sizeof(float));
                cudaMemcpy(d_float_data, h_float_data.data(), num_rows * sizeof(float), cudaMemcpyHostToDevice);


                float* d_min, * d_max;
                cudaMalloc(&d_min, sizeof(float));
                cudaMalloc(&d_max, sizeof(float));


                float init_min = FLT_MAX, init_max = -FLT_MAX;
                cudaMemcpy(d_min, &init_min, sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_max, &init_max, sizeof(float), cudaMemcpyHostToDevice);


                int threadsPerBlock = 256;
                int blocksPerGrid = (num_rows + threadsPerBlock - 1) / threadsPerBlock;
                size_t sharedMemSize = 2 * threadsPerBlock * sizeof(float);


                reduceMinMax << <blocksPerGrid, threadsPerBlock, sharedMemSize >> > (d_float_data, num_rows, d_min, d_max);
                cudaDeviceSynchronize();


                cudaMemcpy(&schema.stats[col].float_min, d_min, sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(&schema.stats[col].float_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);


                cudaFree(d_min);
                cudaFree(d_max);
                cudaFree(d_float_data);
                break;
            }


            case DatasetSchema::INTEGER: {
                for (int i = 0; i < num_rows; ++i) {
                    h_int_data[i] = stoi(raw_data[i][col_idx]);
                }


                int* d_int_data;
                cudaMalloc(&d_int_data, num_rows * sizeof(int));
                cudaMemcpy(d_int_data, h_int_data.data(), num_rows * sizeof(int), cudaMemcpyHostToDevice);


                int h_min, h_max;
                thrust::device_ptr<int> dev_ptr(d_int_data);
                h_min = *(thrust::min_element(dev_ptr, dev_ptr + num_rows));
                h_max = *(thrust::max_element(dev_ptr, dev_ptr + num_rows));


                schema.stats[col].int_min = h_min;
                schema.stats[col].int_max = h_max;


                cudaFree(d_int_data);
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
                break;
            }


            case DatasetSchema::CATEGORY: {
                set<string> unique_categories;
                for (int i = 0; i < num_rows; ++i) {
                    unique_categories.insert(raw_data[i][col_idx]);
                }
                //schema.stats[col].category_values.assign(unique_categoris.begin(), unique_categories.end());
                schema.stats[col].category_values.clear();
                int index = 0;
                for (const auto& cat : unique_categories) {
                    schema.stats[col].category_values[cat] = index++;
                }
                break;
            }
        }
    }


    cout << "[LOG] GPU-based statistics computed successfully.\n";
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


// Kernel: Training the Tsetlin Machine (2D Parallelism: samples × literals)
__global__ void trainTM_2D(
        const int* data, // Flattened binary dataset (samples × literals)
        const int* labels, // Array of labels for each sample
        int num_samples, // Total number of samples
        int num_literals, // Number of literals (features)
        int vote_margin, // Maximum/minimum allowed state value
        int* states // Current state of each literal
) {
    // First dimension parallelism: Each thread processes one data sample
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Second dimension parallelism: Each thread processes one literal
    int literal_idx = blockIdx.y * blockDim.y + threadIdx.y;
    // Boundary check for parallel dimensions
    if (sample_idx >= num_samples || literal_idx >= num_literals) return;
    int label = labels[sample_idx];
    int literal_value = data[sample_idx * num_literals + literal_idx];
    // Update TM states based on labels and literal values (Recognize and Erase feedback)
    if (label == 1) {
        if (literal_value) // if literal is correct
            atomicAdd(&states[literal_idx], 1);   // inc state
        else
            atomicSub(&states[literal_idx], 1);   // dec otherwise
    }
    else { // Negative feedback for negative class
        if (literal_value) // if negative label
            atomicSub(&states[literal_idx], 1);   // dec state
    }


    // Vote margin logic (simplified demonstration)
    // (Adjuststable strength of memorization according to vote margin)
    if (states[literal_idx] > vote_margin) states[literal_idx] = vote_margin;
    if (states[literal_idx] < -vote_margin) states[literal_idx] = -vote_margin;
}


// kernel: sketch 3D
__global__ void trainTM_3D(
        const int* data, // input data
        const int* labels, // input labels
        int num_samples, // number of samples
        int num_literals, // number of literals
        int num_clauses, // number of clauses
        int vote_margin, // vote margin
        int* states // current state of each literal
) {
    // 3D indexing...
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int literal_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int clause_idx = blockIdx.z; // One clause per block in z-dim


    if (sample_idx >= num_samples || literal_idx >= num_literals || clause_idx >= num_clauses)
        return;


    // calc index for current clause's literal state
    int state_idx = clause_idx * num_literals + literal_idx;


    int literal_value = data[sample_idx * num_literals + literal_idx];
    int label = labels[sample_idx];


    // ex. state update logic per clause:
    if (label == 1) {
        if (literal_value) atomicAdd(&states[state_idx], 1);
        else atomicSub(&states[state_idx], 1);
    }
    else {
        if (literal_value) atomicSub(&states[state_idx], 1);
    }


    // Enforce vote_margin bounds, TODO needs doublecheck
    states[state_idx] = max(min(states[state_idx], vote_margin), -vote_margin);
}


// kernel for GPU binarization the previously categorized
__global__ void binarizeDataKernel(
        const int* numeric_data, // input data
        int num_samples, // number of samples
        int num_features, // number of features
        const float* thresholds, // thresholds for binarization
        int* binarized_data // output binarized data
) {
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int feature_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (sample_idx >= num_samples || feature_idx >= num_features)
        return;
    int value = numeric_data[sample_idx * num_features + feature_idx];
    float threshold = thresholds[feature_idx];
    // GPU-based binarization logic
    binarized_data[sample_idx * num_features + feature_idx] = (value >= threshold) ? 1 : 0;
}


// kernel for GPU flattening the previously binarized data
__global__ void flattenBinarizedDataKernel(
        const int* binarized_data, // input binarized data
        int num_samples, // number of samples
        int num_features, // number of features
        int* flat_data // output flattened data
) {
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int feature_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (sample_idx >= num_samples || feature_idx >= num_features)
        return;
    int bin_value = binarized_data[sample_idx * num_features + feature_idx];
    // Assign literals: feature value and its negation
    flat_data[sample_idx * num_features * 2 + feature_idx * 2] = bin_value;
    flat_data[sample_idx * num_features * 2 + feature_idx * 2 + 1] = 1 - bin_value;
}


// kernel, old, previous version of validateTM_2D_old
__global__ void validateTM_2D_old(
        const int* data, // Flattened binary dataset (samples × literals)
        int num_samples, // Total number of samples
        int num_literals, // Number of literals (features)
        const int* states, // Current state of each literal
        int* predictions  // Output array storing predictions per sample
) {
    // First dimension parallelism: One thread per sample
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_idx >= num_samples) return;
    // Calculate votes by summing across literals (second dimension loop)
    int vote = 0;
    for (int literal_idx = 0; literal_idx < num_literals; ++literal_idx) {
        int literal_value = data[sample_idx * num_literals + literal_idx];
        // Positive state contributes positively if literal matches
        if ((states[literal_idx] > 0 && literal_value) || (states[literal_idx] < 0 && !literal_value))
            vote += (states[literal_idx] > 0) ? 1 : -1;
    }
    // Final prediction based on accumulated vote
    predictions[sample_idx] = (vote >= 0) ? 1 : 0;
}


// Kernel: Validation of the Tsetlin Machine (2D Parallelism: samples × literals)
__global__ void validateTM_2D(
        const int* data, // Flattened binary dataset (samples × literals)
        int num_samples, // Total number of samples
        int num_literals, // Number of literals (features)
        const int* states, // Current state of each literal
        int* predictions  // Output array storing predictions per sample
) {
    // First dimension parallelism: One thread per sample
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;


    // Determine current literal index based on block and thread indices
    int literal_idx = blockIdx.y * blockDim.y + threadIdx.y;


    // Ensure indices are within bounds of samples and literals
    if (sample_idx >= num_samples || literal_idx >= num_literals) return;


    // Fetch literal value for the current sample and literal
    int literal_value = data[sample_idx * num_literals + literal_idx];


    // Calculate votes for active literals
    if (literal_value && states[literal_idx] > 0) {
        atomicAdd(&predictions[sample_idx], 1);  // Increment prediction count if literal is active and positive
    }
}


// kernel: categorization of input data
__global__ void categorizeKernel(
        int* d_raw_numeric, // Input numeric categories (from CPU)
        int* d_numeric_data, // Output categorized data
        int num_samples,  // number of samples
        int num_features, // number of features
        int* d_feature_offsets, // offsets for each feature
        int* d_category_maps, // mapping of raw values to categories
        int max_categories // maximum number of categories
) {
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int feature_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (sample_idx >= num_samples || feature_idx >= num_features) return;
    int raw_value = d_raw_numeric[sample_idx * num_features + feature_idx];
    int offset = d_feature_offsets[feature_idx];
    int category_numeric = d_category_maps[offset + raw_value % max_categories];
    d_numeric_data[sample_idx * num_features + feature_idx] = category_numeric;
}


// kernel: binarization of input data
__global__ void binarizeFlattenKernel(
        int* numeric_data, // Input numeric data
        int num_samples, // number of samples
        int num_features, // number of features
        float* thresholds, // thresholds for binarization
        int* flat_data // output flattened data
) {
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int feature_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (sample_idx >= num_samples || feature_idx >= num_features) return;
    int numeric_val = numeric_data[sample_idx * num_features + feature_idx];
    float threshold = thresholds[feature_idx];
    int binary_val = (numeric_val >= threshold) ? 1 : 0;
    // Flatten into literals and their negations
    int num_literals = num_features * 2;
    flat_data[sample_idx * num_literals + 2 * feature_idx] = binary_val;
    flat_data[sample_idx * num_literals + 2 * feature_idx + 1] = 1 - binary_val;
}


// Helper macro with function
#define cudaCheck(err) { cudaAssert((err), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        cerr << "CUDA error: " << cudaGetErrorString(code) << " (" << file << ":" << line << ")\n";
        if (abort) exit(code);
    }
}


int randomizer = 1; // this varable could be replaced with time generated random number... or not.


// main funciton that runs the TM on GPU with given configuratiopn
void cuda_1(
        dim3 threads, // threads config
        int num_literals, // number of literals
        int num_clauses, // number of clauses
        DatasetSchema& schema, // input schema
        const string& data_path, // input file path
        float train_ratio, // ration for splitting input data for training and validation
        float& binarization_ms, // result binarization time
        float& flattening_ms, // result flattening time
        float& training_ms, // result training time
        float& validation_ms, // result validation time
        float& accuracy // resulting accuracy
) {
    cout << "[LOG] Loading raw data...\n";
    vector<vector<string>> raw_data;
    vector<string> raw_labels;
    loadData(raw_data, raw_labels, data_path);
    cout << "[LOG] Computing statistics...\n";
    computeStatistics(raw_data, schema);
    //	computeStatisticsCPU(raw_data, schema);
    cout << "[LOG] Splitting dataset...\n";
    int total_samples = raw_data.size();
    int train_size = static_cast<int>(total_samples * train_ratio);
    int valid_size = total_samples - train_size;
    vector<int> indices(total_samples);
    iota(indices.begin(), indices.end(), 0);
    //shuffle(indices.begin(), indices.end(), mt19937(12345));
    shuffle(indices.begin(), indices.end(), mt19937(12345 + (10 * ++randomizer)));
    auto copySubset = [&](int start, int size, auto& dst_data, auto& dst_labels) {
        dst_data.resize(size);
        dst_labels.resize(size);
        for (int i = 0; i < size; ++i) {
            dst_data[i] = raw_data[indices[start + i]];
            dst_labels[i] = raw_labels[indices[start + i]];
        }
    };
    vector<vector<string>> train_raw, valid_raw;
    vector<string> train_labels_str, valid_labels_str;
    copySubset(0, train_size, train_raw, train_labels_str);
    copySubset(train_size, valid_size, valid_raw, valid_labels_str);


    cout << "[LOG] Preparing data for GPU categorization...\n";
    int num_features = schema.column_mapping.size();
    vector<unordered_map<string, int>> feature_category_maps(num_features);
    vector<int> max_categories_per_feature(num_features, 0);
    cout << "[LOG] Create numeric mappings from strings...\n";
    for (int f = 0; f < num_features; ++f) {
        int category_index = 0;
        for (int i = 0; i < total_samples; ++i) {
            const string& cat = raw_data[i][f];
            auto& map = feature_category_maps[f];
            if (map.find(cat) == map.end()) {
                map[cat] = category_index++;
            }
        }
        max_categories_per_feature[f] = category_index;
    }
    cout << "[LOG] Flatten the mappings for GPU use...\n";
    vector<int> h_feature_offsets(num_features, 0);
    int total_categories = 0;
    for (int f = 0; f < num_features; ++f) {
        h_feature_offsets[f] = total_categories;
        total_categories += max_categories_per_feature[f];
    }
    vector<int> h_category_maps(total_categories, 0);
    for (int f = 0; f < num_features; ++f) {
        int offset = h_feature_offsets[f];
        for (const auto& item : feature_category_maps[f]) {
            const string& cat_str = item.first;
            int cat_idx = item.second;
            h_category_maps[offset + cat_idx] = cat_idx;
        }
    }
    cout << "[LOG] Flatten raw data into numeric indices for GPU categorization...\n";
    auto flattenData = [&](const vector<vector<string>>& raw_subset, vector<int>& flat_numeric) {
        int samples = raw_subset.size();
        flat_numeric.resize(samples * num_features);
        for (int i = 0; i < samples; ++i) {
            for (int f = 0; f < num_features; ++f) {
                flat_numeric[i * num_features + f] = feature_category_maps[f][raw_subset[i][f]];
            }
        }
    };
    vector<int> train_raw_numeric, valid_raw_numeric;
    flattenData(train_raw, train_raw_numeric);
    flattenData(valid_raw, valid_raw_numeric);
    cout << "[LOG] Allocating GPU memory for categorization...\n";
    int* d_train_binarized, * d_valid_binarized;
    cudaMalloc(&d_train_binarized, train_size * num_features * sizeof(int));
    cudaMalloc(&d_valid_binarized, valid_size * num_features * sizeof(int));


    cout << "[LOG] Allocate GPU memory for numeric input...\n";
    int* d_train_numeric, * d_valid_numeric;
    cudaMalloc(&d_train_numeric, train_size * num_features * sizeof(int));
    cudaMalloc(&d_valid_numeric, valid_size * num_features * sizeof(int));


    cout << "[LOG] Allocate GPU memory for category mappings...\n";
    int* d_feature_offsets, * d_category_maps;
    cudaMalloc(&d_feature_offsets, num_features * sizeof(int));
    cudaMalloc(&d_category_maps, total_categories * sizeof(int));


    cout << "[LOG] Kernel launch configuration...\n";


    dim3 blocksTrainCat((train_size + threads.x - 1) / threads.x,
                        (num_features + threads.y - 1) / threads.y);
    dim3 blocksValidCat((valid_size + threads.x - 1) / threads.x,
                        (num_features + threads.y - 1) / threads.y);
    cout << "[LOG] Categorizing training data on GPU...\n";
    categorizeKernel << <blocksTrainCat, threads >> > (
            reinterpret_cast<int*>(d_train_numeric),
                    d_train_numeric,
                    train_size,
                    num_features,
                    d_feature_offsets,
                    d_category_maps,
                    *max_element(max_categories_per_feature.begin(), max_categories_per_feature.end())
    );
    cudaDeviceSynchronize();
    cout << "[LOG] Categorizing validation data on GPU...\n";
    categorizeKernel << <blocksValidCat, threads >> > (
            reinterpret_cast<int*>(d_valid_numeric),
                    d_valid_numeric,
                    valid_size,
                    num_features,
                    d_feature_offsets,
                    d_category_maps,
                    *max_element(max_categories_per_feature.begin(), max_categories_per_feature.end())
    );
    cudaDeviceSynchronize();


    cout << "[LOG] Allocating GPU memory for binarization and flattening...\n";
    int* d_train_flat, * d_valid_flat;
    cudaMalloc(&d_train_flat, train_size * num_literals * sizeof(int));
    cudaMalloc(&d_valid_flat, valid_size * num_literals * sizeof(int));
    dim3 blocksTrainBin((train_size + threads.x - 1) / threads.x, (schema.column_mapping.size() + threads.y - 1) / threads.y);
    dim3 blocksValidBin((valid_size + threads.x - 1) / threads.x, (schema.column_mapping.size() + threads.y - 1) / threads.y);
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    // Calculate thresholds (CPU side)
    vector<float> thresholds(num_features);
    for (int f = 0; f < num_features; ++f) {
        thresholds[f] = (schema.stats[f].float_min + schema.stats[f].float_max) / 2.0f;
    }


    cout << "[LOG] Allocate GPU memory for thresholds and transfer data...\n";
    float* d_thresholds;
    cudaMalloc(&d_thresholds, num_features * sizeof(float));
    cudaMemcpy(d_thresholds, thresholds.data(), num_features * sizeof(float), cudaMemcpyHostToDevice);


    cout << "[LOG] Starting binarization and flattening (train)...\n";
    cudaEventRecord(start);
    binarizeFlattenKernel << <blocksTrainBin, threads >> > (d_train_numeric, train_size, schema.column_mapping.size(), d_thresholds, d_train_flat);
    cudaDeviceSynchronize();
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&binarization_ms, start, stop);


    cout << "[LOG] Starting binarization and flattening (validation)...\n";
    cudaEventRecord(start);
    binarizeFlattenKernel << <blocksValidBin, threads >> > (d_valid_numeric, valid_size, schema.column_mapping.size(), d_thresholds, d_valid_flat);
    cudaDeviceSynchronize();
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&flattening_ms, start, stop);


    cout << "[LOG] Allocating TM states and labels...\n";
    int* d_states, * d_train_labels, * d_valid_labels, * d_predictions;
    cudaMalloc(&d_states, num_clauses * num_literals * sizeof(int));
    cudaMalloc(&d_train_labels, train_size * sizeof(int));
    cudaMalloc(&d_valid_labels, valid_size * sizeof(int));
    cudaMalloc(&d_predictions, valid_size * sizeof(int));
    cudaMemset(d_states, 0, num_clauses * num_literals * sizeof(int));
    vector<int> train_labels = convertLabels(train_labels_str);
    vector<int> valid_labels = convertLabels(valid_labels_str);
    cudaMemcpy(d_train_labels, train_labels.data(), train_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_valid_labels, valid_labels.data(), valid_size * sizeof(int), cudaMemcpyHostToDevice);


    // variables chosen to cover dimensions (dim3):
    //  1. cover the sample dimension
    //	e.g. |samples| = 1025, thread.x = 32 -> (1025 + 32 - 1) / 32 = 33
    //  2. cover all literals (simillar equation as above)
    //  3. per clause
    dim3 blocksTrain((train_size + threads.x - 1) / threads.x, (num_literals + threads.y - 1) / threads.y, num_clauses);
    dim3 blocksValid((valid_size + threads.x - 1) / threads.x, (num_literals + threads.y - 1) / threads.y, num_clauses);


    cout << "[LOG] Running training kernel...\n";
    cudaEventRecord(start);
    trainTM_2D << <blocksTrain, threads >> > (d_train_flat, d_train_labels, train_size, num_literals, 2, d_states);
    cudaDeviceSynchronize();
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&training_ms, start, stop);


    cout << "[LOG] Preparing validation data...\n";
    dim3 blocksValidFlat((valid_size + threads.x - 1) / threads.x, (num_features + threads.y - 1) / threads.y);


    binarizeDataKernel << <blocksValidBin, threads >> > (d_valid_numeric, valid_size, num_features, d_thresholds, d_valid_binarized);
    cudaDeviceSynchronize();
    flattenBinarizedDataKernel << <blocksValidFlat, threads >> > (d_valid_binarized, valid_size, num_features, d_valid_flat);
    cudaDeviceSynchronize();


    cout << "[LOG] Running validation kernel...\n";
    cudaEventRecord(start);
    validateTM_2D << <blocksValid, threads >> > (d_valid_flat, valid_size, num_literals, d_states, d_predictions);
    cudaDeviceSynchronize();
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&validation_ms, start, stop);


    cout << "[LOG] Copying predictions and calculating accuracy...\n";
    vector<int> predictions(valid_size);
    cudaMemcpy(predictions.data(), d_predictions, valid_size * sizeof(int), cudaMemcpyDeviceToHost);
    int correct = 0;
    for (int i = 0; i < valid_size; ++i)
        correct += (predictions[i] == valid_labels[i]);


    accuracy = 100.0f * correct / valid_size;


    cout << "[LOG] Freeing GPU resources...\n";
    cudaFree(d_train_numeric); cudaFree(d_valid_numeric);
    cudaFree(d_train_binarized); cudaFree(d_valid_binarized);
    cudaFree(d_train_flat); cudaFree(d_valid_flat);
    cudaFree(d_train_labels); cudaFree(d_valid_labels);
    cudaFree(d_states); cudaFree(d_predictions); cudaFree(d_thresholds);
    cudaEventDestroy(start); cudaEventDestroy(stop);


    cout << "[LOG] Cleaning GPU memory...\n";
    cudaFree(d_train_numeric); cudaFree(d_valid_numeric); cudaFree(d_train_flat); cudaFree(d_valid_flat);
    cudaFree(d_states); cudaFree(d_train_labels); cudaFree(d_valid_labels); cudaFree(d_predictions);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}


// evauation function,
int cuda_1_eval(
        int num_clauses, // number of clauses
        dim3 threads,  // threads configuration
        int cuda_iterations, // number of iterations of TM train and eval to have more precise timing information
        float* bin_timing_avg, // binarization time
        float* flat_timing_avg, // flattening time
        float* train_timing_avg, // training time
        float* val_timing_avg, // validation time
        float* total_timing_avg, // total time
        float* total_timing_cpu_avg, // total time on CPU
        const string& data_path
) {


    // Configuration
    int num_features = 20;
    int num_literals = num_features * 2;
    float train_ratio = 2.0f / 3.0f;
    DatasetSchema schema(20);
    schema.column_mapping = {
            {0, DatasetSchema::FLOAT}, {1, DatasetSchema::CATEGORY}, {2, DatasetSchema::CATEGORY},
            {3, DatasetSchema::CATEGORY}, {4, DatasetSchema::BOOL}, {5, DatasetSchema::CATEGORY},
            {6, DatasetSchema::CATEGORY}, {7, DatasetSchema::CATEGORY}, {8, DatasetSchema::FLOAT},
            {9, DatasetSchema::FLOAT}, {10, DatasetSchema::CATEGORY}, {11, DatasetSchema::CATEGORY},
            {12, DatasetSchema::CATEGORY}, {13, DatasetSchema::CATEGORY}, {14, DatasetSchema::CATEGORY},
            {15, DatasetSchema::BOOL}, {16, DatasetSchema::CATEGORY}, {17, DatasetSchema::CATEGORY},
            {18, DatasetSchema::CATEGORY}, {19, DatasetSchema::CATEGORY}
    };
    // Vectors to store results from each run
    vector<float> binarization_timings;
    vector<float> flattening_timings;
    vector<float> training_timings;
    vector<float> validation_timings;
    vector<float> total_timing;
    vector<float> accuracies;
    // Run the evaluation cuda_iterations times
    for (int run = 1; run <= cuda_iterations; ++run) {
        cout << "\n--- [RUN " << run << "] ---\n";


        float bin_ms, flat_ms, train_ms, val_ms, accuracy;


        auto cpu_start_time = std::chrono::high_resolution_clock::now();


        cuda_1(threads, num_literals, num_clauses, schema, data_path, train_ratio,
               bin_ms, flat_ms, train_ms, val_ms, accuracy);


        auto cpu_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> cpu_elapsed = cpu_end_time - cpu_start_time;


        *total_timing_cpu_avg = cpu_elapsed.count();


        binarization_timings.push_back(bin_ms);
        flattening_timings.push_back(flat_ms);
        training_timings.push_back(train_ms);
        validation_timings.push_back(val_ms);
        accuracies.push_back(accuracy);
        total_timing.push_back(bin_ms + flat_ms + train_ms + val_ms);
        // Log per-run results
        cout << "[RESULTS - RUN " << run << "]\n";
        cout << "Binarization kernel: " << bin_ms << " ms\n";
        cout << "Flattening kernel:   " << flat_ms << " ms\n";
        cout << "Training kernel:   " << train_ms << " ms\n";
        cout << "Validation kernel:   " << val_ms << " ms\n";
        cout << "Validation accuracy: " << accuracy << "%\n";
    }
    // Print all accumulated results per iteration
    cout << "\n=== [ACCUMULATED RESULTS FROM ALL RUNS] ===\n";
    for (size_t i = 0; i < accuracies.size(); ++i) {
        cout << "[Run " << (i + 1) << "] "
             << "Bin: " << binarization_timings[i] << " ms, "
             << "Flat: " << flattening_timings[i] << " ms, "
             << "Train: " << training_timings[i] << " ms, "
             << "Valid: " << validation_timings[i] << " ms, "
             << "Total: " << total_timing[i] << " ms, "
             << "Accuracy: " << accuracies[i] << "%\n";
    }
    // calc avgs
    auto average = [](const vector<float>& v) {
        return accumulate(v.begin(), v.end(), 0.0f) / v.size();
    };


    *bin_timing_avg = average(binarization_timings);
    *flat_timing_avg = average(flattening_timings);
    *train_timing_avg = average(training_timings);
    *val_timing_avg = average(validation_timings);
    *total_timing_avg = average(total_timing);
    cout << "\n=== [AVERAGE RESULTS OVER 10 RUNS] ===\n";
    cout << "Average Binarization kernel: " << *bin_timing_avg << " ms\n";
    cout << "Average Flattening kernel:   " << *flat_timing_avg << " ms\n";
    cout << "Average Training kernel:   " << *train_timing_avg << " ms\n";
    cout << "Average Validation kernel:   " << *val_timing_avg << " ms\n";
    cout << "Average Total time:   " << *total_timing_avg << " ms\n";
    cout << "Average Validation accuracy: " << average(accuracies) << "%\n";
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <data_path>" << endl;
        return 1;
    }

    string data_path = argv[1];
//    string data_path = "C:\\Users\\jdoe\\Downloads\\secondary_data.csv";
//    string data_path = "secondary_data.csv";

    cout << "Data path set to: " << data_path << endl;

    int cuda_iterations = 5;
    //vector<int> num_clauses_set = { 10, 20, 50, 100, 200, 500, 1000, 5000, 10000, 100000, 1000000 }; // log 1
    //vector<int> num_clauses_set = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 50, 100, 500, 1000, 10000, 100000000 }; // log 2 10x data
    vector<int> num_clauses_set = { 1, 2, 3, 4, 5, 10, 100, 1000, 10000 };

    // Thread configurations (uncommented explicitly here):
    //vector<pair<int, int>> thread_configs = { {32, 32}, {16, 16}, {8, 8} };
    //vector<string> thread_config_labels = { "32x32", "16x16", "8x8" };
    vector<pair<int, int>> thread_configs = { {32, 32}, {16, 16}};
    vector<string> thread_config_labels = { "32x32", "16x16"};
    //vector<pair<int, int>> thread_configs = { {32, 32} };
    //vector<string> thread_config_labels = { "32x32" };

    struct Timings {
        float binarization;
        float flattening;
        float training;
        float validation;
        float total;
        float total_cpu;
    };
    vector<vector<Timings>> results(thread_configs.size(), vector<Timings>(num_clauses_set.size()));
    for (size_t config_idx = 0; config_idx < thread_configs.size(); ++config_idx) {
        //dim3 [threads_x, threads_y] = thread_configs[config_idx];
        dim3 threads(thread_configs[config_idx].first, thread_configs[config_idx].second);
        cout << "Running for thread config: " << thread_config_labels[config_idx] << "\n";
        for (size_t clause_idx = 0; clause_idx < num_clauses_set.size(); ++clause_idx) {
            int num_clauses_i = num_clauses_set[clause_idx];
            float bin_timing_avg, flat_timing_avg, train_timing_avg, val_timing_avg, total_timing_avg, total_timing_avg_cpu;
            cuda_1_eval(num_clauses_i, threads, cuda_iterations,
                        &bin_timing_avg, &flat_timing_avg, &train_timing_avg, &val_timing_avg, &total_timing_avg, &total_timing_avg_cpu, data_path);

            results[config_idx][clause_idx] = { bin_timing_avg, flat_timing_avg, train_timing_avg, val_timing_avg, total_timing_avg, total_timing_avg_cpu };
        }
    }

    // Printing results:
    for (size_t config_idx = 0; config_idx < thread_configs.size(); ++config_idx) {
        cout << "\n\nThread Config: " << thread_config_labels[config_idx] << "\n";
        cout << "Clauses | Binarization | Flattening | Training | Validation | Total GPU | Total CPU\n";
        for (size_t clause_idx = 0; clause_idx < num_clauses_set.size(); ++clause_idx) {
            const Timings& timing = results[config_idx][clause_idx];
            cout << num_clauses_set[clause_idx] << "\t"
                 << timing.binarization << " ms\t"
                 << timing.flattening << " ms\t"
                 << timing.training << " ms\t"
                 << timing.validation << " ms\t"
                 << timing.total << " ms\t"
                 << timing.total_cpu << " ms\n";
        }
    }


    return 0;
}

