#pragma once

class Binning2Debayer {
 public:
    void demosaicing(const cv::cuda::GpuMat & input, cv::cuda::GpuMat & output);
    int binningFactor() { return _binningFactor; }
 private:
    int _binningFactor = 2;
};
