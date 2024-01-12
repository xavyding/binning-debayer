#include "Binning2Debayer.h"

static constexpr int KERNEL_BLOCK_SIZE = 16;

__global__ void __gpuKernelv2_Binning2demosaicingC3x16(const ushort* inputData, ushort* outputData, const int outputRows, const int outputCols, \
    int blockSkipFactor, const size_t inputStep, const size_t outputStep, const int inputChannels, const int outputChannels = 3) {
    // 16 bits version
    /* Input mosaic color bloc representation:
    +-------+   x (inputOffsetX)
    | R | Gr| 
    |---+---+
    | Gb| B |
    +---+---+
    
    y (inputOffsetY)
    */
    if (blockIdx.x * blockSkipFactor >= outputCols || blockIdx.y * blockSkipFactor >= outputRows) return;

    int inputPosX = threadIdx.x / 2;
    int inputOffsetX = threadIdx.x % 2;
    int inputPosY = threadIdx.y / 2;
    int inputOffsetY = threadIdx.y % 2;

    if (inputOffsetX == 1 && inputOffsetY == 0) return;

    int outputIdx = (inputPosY + blockIdx.y * blockSkipFactor) * outputStep + (inputPosX + blockIdx.x * blockSkipFactor) * outputChannels;
    int inputIdx = (threadIdx.y + blockIdx.y * blockDim.y) * inputStep + (threadIdx.x + blockIdx.x * blockDim.x) * inputChannels;

    if (inputOffsetX == 1 && inputOffsetY == 1) {  // Blue
        outputData[outputIdx] = inputData[inputIdx];
    } else if (inputOffsetX == 0 && inputOffsetY == 1) {  // Green (Gb)
        outputData[outputIdx + 1] = inputData[inputIdx];
    } else if (inputOffsetX == 0 && inputOffsetY == 0) {  // Red
        outputData[outputIdx + 2] = inputData[inputIdx];
    } else {
        return;
    }
}


__host__ void Binning2Debayer::demosaicing(const cv::cuda::GpuMat & input, cv::cuda::GpuMat & output) {
    // check if already 3 channels, in which case no need to do demosaicing.
    if (input.channels() == 3) {
        output = input;
        return;
    }

    int blockSize = KERNEL_BLOCK_SIZE;

    dim3 blocksPerGrid = dim3(static_cast<int>(input.step/sizeof(ushort)/blockSize), static_cast<int>(input.rows/blockSize));
    dim3 threadsPerBlock = dim3(blockSize, blockSize);
    __gpuKernelv2_Binning2demosaicingC3x16<<<blocksPerGrid, threadsPerBlock>>> (input.ptr<ushort>(), output.ptr<ushort>(), \
        output.rows, output.cols, blockSize/_binningFactor, input.step/sizeof(ushort), output.step/sizeof(ushort),  1, 3);
}

