/*
 * PyTorch C++ bindings for Flash Decoding INT4 CUDA kernels
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

extern "C" {
cudaError_t cu_quantize_int4(const float*, int, int, uint8_t*, float*, float*, cudaStream_t);
cudaError_t cu_flash_decode_int4(const float*, const uint8_t*, const float*, const float*,
                                  const float*, const int*, int, int, int, int, int,
                                  float*, cudaStream_t);
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> quantize_int4(torch::Tensor kv) {
    CHECK_INPUT(kv);
    TORCH_CHECK(kv.dim() == 2, "kv must be 2D [num_rows, num_channels]");

    int num_rows = kv.size(0);
    int num_channels = kv.size(1);

    auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(kv.device());
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(kv.device());

    auto q = torch::empty({num_rows, num_channels}, opts_u8);
    auto scale = torch::empty({num_channels}, opts_f32);
    auto zp = torch::empty({num_channels}, opts_f32);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaError_t err = cu_quantize_int4(
        kv.data_ptr<float>(), num_rows, num_channels,
        q.data_ptr<uint8_t>(), scale.data_ptr<float>(), zp.data_ptr<float>(),
        stream
    );
    TORCH_CHECK(err == cudaSuccess, "quantize_int4 failed: ", cudaGetErrorString(err));

    return {q, scale, zp};
}

torch::Tensor flash_decode_int4(
    torch::Tensor query,      // [batch, heads, head_dim]
    torch::Tensor k_q,        // [num_blocks, block_size, head_dim]
    torch::Tensor k_scale,    // [num_blocks, head_dim]
    torch::Tensor k_zp,       // [num_blocks, head_dim]
    torch::Tensor value,      // [num_blocks, block_size, head_dim]
    torch::Tensor block_lens  // [num_blocks]
) {
    CHECK_INPUT(query);
    CHECK_INPUT(k_q);
    CHECK_INPUT(k_scale);
    CHECK_INPUT(k_zp);
    CHECK_INPUT(value);
    CHECK_INPUT(block_lens);

    int batch_size = query.size(0);
    int num_heads = query.size(1);
    int head_dim = query.size(2);
    int num_blocks = k_q.size(0);
    int block_size = k_q.size(1);

    auto output = torch::empty({batch_size, num_heads, head_dim},
                               torch::TensorOptions().dtype(torch::kFloat32).device(query.device()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaError_t err = cu_flash_decode_int4(
        query.data_ptr<float>(),
        k_q.data_ptr<uint8_t>(),
        k_scale.data_ptr<float>(),
        k_zp.data_ptr<float>(),
        value.data_ptr<float>(),
        block_lens.data_ptr<int>(),
        batch_size, num_heads, head_dim, num_blocks, block_size,
        output.data_ptr<float>(),
        stream
    );
    TORCH_CHECK(err == cudaSuccess, "flash_decode_int4 failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_int4", &quantize_int4, "Per-channel asymmetric INT4 quantization (CUDA)");
    m.def("flash_decode_int4", &flash_decode_int4, "Fused online softmax + INT4 dequant attention (CUDA)");
}
