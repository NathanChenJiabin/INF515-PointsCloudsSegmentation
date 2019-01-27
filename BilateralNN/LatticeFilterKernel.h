//
// Created by jiabin on 27/01/19.
//

#ifndef BILATERALNN_LATTICEFILTERKERNEL_H
#define BILATERALNN_LATTICEFILTERKERNEL_H

#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;

template<typename Device, typename T>
struct LatticeFilter{
    void operator()(const Device &d,
                    OpKernelContext* context,
                    T *output,
                    const T *val_input,
                    const T *positions,
                    int num_pixels,
                    int pd,
                    int vd,
                    bool reverse);
};

#endif //BILATERALNN_LATTICEFILTERKERNEL_H
