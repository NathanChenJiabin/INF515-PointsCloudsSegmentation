
#include <iostream>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "PermutohedralLattice.h"
#include "LatticeFilterKernel.h"

using namespace tensorflow;

REGISTER_OP("LatticeFilter")
        .Attr("T: {float, double}")
        .Attr("bilateral: bool = true")
        .Attr("theta_alpha: float = 1.0")
        .Attr("theta_beta: float = 1.0")
        .Attr("theta_gamma: float = 1.0")
        .Input("input_image: T")
        .Input("reference_image: T")
        .Output("output: T")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            c->set_output(0, c->input(0));
            return Status::OK();
        });

//template<typename T>
//struct ComputeKernel<CPUDevice, T> {
//    void operator()(const CPUDevice& d, T a){
//        cout << a << endl;
//    }
//};
template <typename T>
struct LatticeFilter<CPUDevice, T> {
    void operator()(const CPUDevice& d,
                    OpKernelContext* context,
                    T* output,
                    const T *val_input,
                    const T *positions,
                    int num_pixels,
                    int pd,
                    int vd,
                    bool reverse){

        filter( positions, val_input, output, pd, vd, num_pixels, reverse);
    }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class LatticeFilterOp : public OpKernel {
public:
    explicit LatticeFilterOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("bilateral", &bilateral));
        OP_REQUIRES_OK(context, context->GetAttr("theta_alpha", &theta_alpha));
        OP_REQUIRES_OK(context, context->GetAttr("theta_beta", &theta_beta));
        OP_REQUIRES_OK(context, context->GetAttr("theta_gamma", &theta_gamma));
    }

    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        const Tensor& input_tensor = context->input(0);
        const Tensor& reference_image_tensor = context->input(1);

        // Create an output tensor
        Tensor* output_tensor = nullptr;
        // &ptr returns the address of the pointer variable, likes pointer to a pointer
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));

        // Do the computation.
        OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max, // ???
                    errors::InvalidArgument("Too many elements in tensor"));

        // calculate dimensions; dimension 0 is batch; last dimension is channel
        int rank = input_tensor.dims();
        int n_spatial_dims = rank - 2;

        auto batch_size = static_cast<int>(input_tensor.dim_size(0));
        auto n_input_channels = static_cast<int>(input_tensor.dim_size(rank - 1));
        auto spatial_dims = new int[n_spatial_dims];

        int num_super_pixels{1};
        for (int i = 0; i < n_spatial_dims; i++){
            auto dim_size = static_cast<int>(input_tensor.dim_size(i + 1));
            num_super_pixels *= dim_size;
            spatial_dims[i] = dim_size;
        }

        vd = n_input_channels + 1;
        float spatial_std;
        float features_std;
        int n_reference_channels;

        if(bilateral){
            assert(reference_image_tensor.dims() == rank);
            n_reference_channels = static_cast<int>(reference_image_tensor.dim_size(rank - 1));
            pd = n_reference_channels + n_spatial_dims;
            spatial_std = theta_alpha;
            features_std = theta_beta;
        }else{
            pd = n_spatial_dims;
            n_reference_channels = 0; //set to zero so ComputeKernel does not use reference image channels
            spatial_std = theta_gamma;
            features_std = -1; //does not matter
        }

        // Allocate kernel positions and calculate them
        Tensor positions;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::v(),
                                                       TensorShape({batch_size * num_super_pixels * pd}),
                                                       &positions));

        // auto allocator = DeviceMemoryAllocator(context);

        for(int b=0; b < batch_size; b++){

            auto ref_ptr = &(reference_image_tensor.flat<T>().data()[b * num_super_pixels * n_reference_channels]);
            auto pos_ptr = &(positions.flat<T>().data()[b * num_super_pixels * pd]);
            auto in_ptr = &(input_tensor.flat<T>().data()[b * num_super_pixels * n_input_channels]);
            auto out_ptr = &(output_tensor->flat<T>().data()[b * num_super_pixels * n_input_channels]);

//            ComputeKernel<Device, T>()(context->eigen_device<Device>(),
//                                       context,
//                                       ref_ptr,
//                                       pos_ptr,
//                                       num_super_pixels,
//                                       n_spatial_dims,
//                                       spatial_dims,
//                                       n_reference_channels,
//                                       spatial_std,
//                                       features_std);

            LatticeFilter<Device, T>()(context->eigen_device<Device>(),
                                       context,
                                       out_ptr,
                                       in_ptr,
                                       pos_ptr,
                                       num_super_pixels,
                                       pd,
                                       vd,
                                       reverse);
        }
        delete[](spatial_dims);
    }

private:
    bool bilateral;
    float theta_alpha;
    float theta_beta;
    float theta_gamma;
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

//#define REGISTER_KERNEL(type)                                       \
//  REGISTER_KERNEL_BUILDER(                                          \
//      Name("LatticeFilter").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
//      LatticeFilterOp<CPUDevice, type>)
//
//REGISTER_KERNEL(float);
//REGISTER_KERNEL(double);
//
//#undef REGISTER_KERNEL

