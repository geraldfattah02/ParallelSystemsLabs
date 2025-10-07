# Compiler and flags
CXX = g++
NVCC = nvcc
CXXFLAGS = -O3 -Wall -std=c++17
NVCCFLAGS = -O3 --std c++14 -Xcompiler -Wall

# Output directory
BIN_DIR = ./bin

# Executables
CPU_EXE = $(BIN_DIR)/kmeans_cpu
BASIC_EXE = $(BIN_DIR)/kmeans_cuda_basic
SHMEM_EXE = $(BIN_DIR)/kmeans_cuda_shmem
THRUST_EXE = $(BIN_DIR)/kmeans_thrust

# Default target
all: $(CPU_EXE) $(BASIC_EXE) $(SHMEM_EXE) $(THRUST_EXE)

# CPU version
$(CPU_EXE): main.cpp kmeans_cpu.cpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -DUSE_CPU $^ -o $@

# CUDA basic version
$(BASIC_EXE): main.cpp kmeans_cuda_basic.cu
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA_BASIC $^ -o $@

# CUDA shared memory version
$(SHMEM_EXE): main.cpp kmeans_cuda_shmem.cu
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA_SHMEM $^ -o $@

# Thrust version
$(THRUST_EXE): main.cpp kmeans_thrust.cu
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -DUSE_THRUST $^ -o $@

# Remove compiled files
clean:
	rm -rf $(BIN_DIR)
