
#include <iostream>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "PermutohedralLattice.h"
#include "LatticeFilterKernel.h"


typedef Eigen::ThreadPoolDevice CPUDevice;

REGISTER_OP("LatticeFilter")
        .Attr("T: {float, double}")
        .Attr("reverse: bool = false")
        .Input("input_image: T")
        .Input("reference_image: T")
        .Output("output: T")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            c->set_output(0, c->input(0));
            return Status::OK();
        });

template <typename T>
struct LatticeFilterFunctor<CPUDevice, T>{
    void operator()(const CPUDevice& d,
                    OpKernelContext* context,
                    T* output,
                    const T *val_input,
                    const T *positions,
                    int num_pixels,
                    int pd,
                    int vd,
                    bool reverse){

        filter<T>( positions, val_input, output, pd, vd, num_pixels, reverse);
    }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class LatticeFilterOp : public OpKernel {
public:
    explicit LatticeFilterOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("reverse", &reverse));
    }

    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        const Tensor& input_tensor = context->input(0); // shape (nb_points, val_channels) Just one batch
        const Tensor& reference_image_tensor = context->input(1); // shape (nb_points, pos_channels)

        // Create an output tensor
        Tensor* output_tensor = nullptr;
        // &ptr returns the address of the pointer variable, likes pointer to a pointer
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));

        // Do the computation.
        OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max, // ???
                    errors::InvalidArgument("Too many elements in tensor"));

        // calculate dimensions;  last dimension is channel
        int rank = input_tensor.dims(); // 2
        int n_spatial_dims = rank - 2;  // 0

        vd = static_cast<int>(input_tensor.dim_size(rank - 1)); // 3

        int num_points = static_cast<int>(input_tensor.dim_size(0));

        pd = static_cast<int>(reference_image_tensor.dim_size(rank - 1));

        assert(reference_image_tensor.dims() == rank);

        // auto allocator = DeviceMemoryAllocator(context);

        auto pos_ptr = &(reference_image_tensor.flat<T>().data()[0]);
        auto in_ptr = &(input_tensor.flat<T>().data()[0]);
        auto out_ptr = &(output_tensor->flat<T>().data()[0]);


        LatticeFilterFunctor<Device, T>()( context->eigen_device<Device>(),
                                           context,
                                           out_ptr,
                                           in_ptr,
                                           pos_ptr,
                                           num_points,
                                           pd,
                                           vd,
                                           reverse );

    }

private:
    bool reverse;
    int pd;
    int vd;
};

// Register the CPU kernel
#define REGISTER_CPU(T)                                              \
    REGISTER_KERNEL_BUILDER(                                         \
    Name("LatticeFilter").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
    LatticeFilterOp<CPUDevice, T>);

REGISTER_CPU(float);
REGISTER_CPU(double);
