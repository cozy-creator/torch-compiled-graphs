
#include <torch/csrc/inductor/aoti_include/cuda.h>
// Definition of AOTI runtime interface functions

#include <torch/csrc/inductor/aoti_runtime/interface.h>
#include <torch/csrc/inductor/aoti_runtime/model_container.h>

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

// Stores the last error message from a failed AOTI runtime call so that
// callers on the other side of the C ABI boundary can retrieve it via
// AOTInductorGetLastError(). Without this, exception messages (e.g.
// "CUDA error: an illegal memory access was encountered") are lost when
// CONVERT_EXCEPTION_TO_ERROR_CODE catches them and returns an error code.
static thread_local std::string g_aoti_last_error;

#define CONVERT_EXCEPTION_TO_ERROR_CODE(...)      \
  try {                                           \
    g_aoti_last_error.clear();                    \
    __VA_ARGS__                                   \
  } catch (const std::exception& e) {             \
    g_aoti_last_error = e.what();                 \
    std::cerr << "Error: " << e.what() << '\n';   \
    return AOTI_RUNTIME_FAILURE;                  \
  } catch (...) {                                 \
    g_aoti_last_error = "Unknown exception";      \
    std::cerr << "Unknown exception occurred.\n"; \
    return AOTI_RUNTIME_FAILURE;                  \
  }                                               \
  return AOTI_RUNTIME_SUCCESS;

#define AOTI_VECTOR_SIZE_CHECK(actual_size, expected_size, name)  \
  do {                                                            \
    AOTI_RUNTIME_CHECK(                                           \
        actual_size == expected_size,                             \
        "expected " + std::string(name) + " vector size to be " + \
            std::to_string(expected_size) + ", but got " +        \
            std::to_string(actual_size));                         \
  } while (0)

// AOTInductor uses at::addmm_out, which doesn't supports
// arguments that requires gradient. For this reason, we
// enforce no_grad context for run APIs.
//
// A RAII, thread local (!) guard that enables or disables grad mode upon
// construction, and sets it back to the original value upon destruction.
struct AOTINoGradGuard {
  AOTINoGradGuard() {
    aoti_torch_grad_mode_set_enabled(false);
  }
  AOTINoGradGuard(const AOTINoGradGuard&) = delete;
  AOTINoGradGuard(AOTINoGradGuard&&) noexcept = delete;
  ~AOTINoGradGuard() {
    aoti_torch_grad_mode_set_enabled(prev_mode);
  }
  AOTINoGradGuard& operator=(const AOTINoGradGuard&) = delete;
  AOTINoGradGuard& operator=(AOTINoGradGuard&&) noexcept = delete;
  bool prev_mode{aoti_torch_grad_mode_is_enabled()};
};

namespace {

std::unordered_map<std::string, AtenTensorHandle> constant_map_from_pairs(
    const AOTInductorConstantMapEntry* pairs,
    size_t num_pairs) {
  std::unordered_map<std::string, AtenTensorHandle> input_map;
  input_map.reserve(num_pairs);
  for (size_t i = 0; i < num_pairs; ++i) {
    input_map.emplace(pairs[i].name, pairs[i].handle);
  }
  return input_map;
}

// Shared constructor for AOTInductorModelCreate / AOTInductorModelCreateV2.
// `populate(constant_map)` is called between model construction and
// optional embedded-blob loading.
template <typename Populate>
AOTIRuntimeError createModelImpl(
    AOTInductorModelHandle* model_handle,
    bool load_constants_from_blob,
    Populate&& populate) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto constant_map = std::make_shared<torch::aot_inductor::ConstantMap>();
    auto constant_array = std::make_shared<
        std::vector<torch::aot_inductor::ConstantHandle>>();
    auto* model = new torch::aot_inductor::AOTInductorModel(
        constant_map,
        constant_array,
        // device_str is hardcoded, as AOTInductorModelCreate is only used
        // for CPU models.
        "cpu",
        "");
    populate(*constant_map);
    if (load_constants_from_blob) {
      model->load_constants();
    }
    *model_handle = reinterpret_cast<AOTInductorModelHandle>(model);
  })
}

} // namespace

extern "C" {

AOTIRuntimeError AOTInductorModelContainerCreate(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    bool is_cpu,
    const char* cubin_dir) {
      return AOTInductorModelContainerCreateWithDevice(
        container_handle,
        num_models,
        is_cpu ? "cpu" : "cuda",
        cubin_dir);
}

AOTIRuntimeError AOTInductorModelContainerCreateWithDevice(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    const char* device_str,
    const char* cubin_dir) {

  if (num_models == 0) {
    std::cerr << "Error: num_models must be positive, but got 0\n";
    return AOTI_RUNTIME_FAILURE;
  }
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    std::optional<std::string> cubin_dir_opt;
    if (cubin_dir != nullptr) {
      cubin_dir_opt.emplace(cubin_dir);
    }
    auto* container = new torch::aot_inductor::AOTInductorModelContainer(
        num_models, std::string(device_str), cubin_dir_opt);
    *container_handle =
        reinterpret_cast<AOTInductorModelContainerHandle>(container);
  })
}


AOTIRuntimeError AOTInductorModelContainerDelete(
    AOTInductorModelContainerHandle container_handle) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* container =
        reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
            container_handle);
    delete container;
  });
}

AOTIRuntimeError AOTInductorModelContainerRun(
    AOTInductorModelContainerHandle container_handle,
    AtenTensorHandle* input_handles, // array of input AtenTensorHandle; handles
                                     // are stolen; the array itself is borrowed
    size_t num_inputs,
    AtenTensorHandle*
        output_handles, // array for writing output AtenTensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed
    size_t num_outputs,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  AOTI_VECTOR_SIZE_CHECK(num_inputs, container->num_inputs(), "inputs");
  AOTI_VECTOR_SIZE_CHECK(num_outputs, container->num_outputs(), "outputs");

  auto stream =
      reinterpret_cast<torch::aot_inductor::DeviceStreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    container->run(
        input_handles, output_handles, stream, proxy_executor_handle);
  })
}

AOTIRuntimeError AOTInductorModelContainerRunSingleThreaded(
    AOTInductorModelContainerHandle container_handle,
    AtenTensorHandle* input_handles, // array of input AtenTensorHandle; handles
                                     // are stolen; the array itself is borrowed
    size_t num_inputs,
    AtenTensorHandle*
        output_handles, // array for writing output AtenTensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed
    size_t num_outputs,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  AOTI_VECTOR_SIZE_CHECK(num_inputs, container->num_inputs(), "inputs");
  AOTI_VECTOR_SIZE_CHECK(num_outputs, container->num_outputs(), "outputs");

  auto stream =
      reinterpret_cast<torch::aot_inductor::DeviceStreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    container->run_single_threaded(
        input_handles, output_handles, stream, proxy_executor_handle);
  })
}

AOTIRuntimeError AOTInductorModelContainerGetNumConstants(
    AOTInductorModelContainerHandle container_handle,
    size_t* num_constants) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *num_constants = container->num_constants(); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantName(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    const char** name) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *name = container->constant_name(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantOriginalFQN(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    const char** original_fqn) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *original_fqn = container->constant_original_fqn(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantFromFolded(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    bool* from_folded) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *from_folded = container->constant_from_folded(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantType(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    int32_t* type) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *type = container->constant_type(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantDtype(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    int32_t* dtype) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *dtype = container->constant_dtype(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantDataSize(
  AOTInductorModelContainerHandle container_handle,
  size_t idx,
  size_t* data_size) {
  auto* container =
    reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
        container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *data_size = container->constant_data_size(idx); })
}

AOTIRuntimeError AOTInductorModelContainerExtractConstantsMap(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto constants_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { const auto ret = container->extract_constants_map(use_inactive);
      for (const auto& pair: ret) {
        constants_map->emplace(pair.first, pair.second);
      }
    })
}

AOTIRuntimeError AOTInductorModelContainerExtractConstantsMapEntries(
    AOTInductorModelContainerHandle container_handle,
    const AOTInductorConstantMapEntry** entries,
    size_t* num_entries,
    bool use_inactive) {
  if (!entries || !num_entries) {
    return AOTI_RUNTIME_FAILURE;
  }
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    const auto& extracted =
        container->extract_constants_map_entries(use_inactive);
    *entries = extracted.empty() ? nullptr : extracted.data();
    *num_entries = extracted.size();
  })
}

AOTIRuntimeError AOTInductorModelContainerUpdateUserManagedConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive,
    bool validate_full_update) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto input_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->update_constant_buffer(
        *input_map, use_inactive, validate_full_update, /* user_managed = */ true);
  })
}

AOTIRuntimeError AOTInductorModelContainerUpdateUserManagedConstantBufferPairs(
    AOTInductorModelContainerHandle container_handle,
    const AOTInductorConstantMapEntry* pairs,
    size_t num_pairs,
    bool use_inactive,
    bool validate_full_update) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(container_handle);
  // Build a local unordered_map inside
  std::unordered_map<std::string, AtenTensorHandle> input_map;
  input_map.reserve(num_pairs);
  for (size_t i = 0; i < num_pairs; ++i) {
      input_map.emplace(pairs[i].name, pairs[i].handle);
  }
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->update_constant_buffer(
        input_map, use_inactive, validate_full_update, /*user_managed=*/true);
  })
}

AOTIRuntimeError AOTInductorModelContainerUpdateConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive,
    bool validate_full_update) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto input_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->update_constant_buffer(
        *input_map, use_inactive, validate_full_update);
  })
}

AOTIRuntimeError AOTInductorModelContainerUpdateConstantBufferPairs(
    AOTInductorModelContainerHandle container_handle,
    const AOTInductorConstantMapEntry* pairs,
    size_t num_pairs,
    bool use_inactive,
    bool validate_full_update) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto input_map = constant_map_from_pairs(pairs, num_pairs);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->update_constant_buffer(
        input_map, use_inactive, validate_full_update);
  })
}

AOTIRuntimeError AOTInductorModelContainerUpdateConstantBufferFromCpu(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive,
    bool validate_full_update) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto input_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->update_constant_buffer(
        *input_map,
        use_inactive,
        validate_full_update,
        /*user_managed=*/false,
        /*allow_h2d_copy=*/true);
  })
}

AOTIRuntimeError AOTInductorModelContainerUpdateConstantBufferFromCpuPairs(
    AOTInductorModelContainerHandle container_handle,
    const AOTInductorConstantMapEntry* pairs,
    size_t num_pairs,
    bool use_inactive,
    bool validate_full_update) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto input_map = constant_map_from_pairs(pairs, num_pairs);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->update_constant_buffer(
        input_map,
        use_inactive,
        validate_full_update,
        /*user_managed=*/false,
        /*allow_h2d_copy=*/true);
  })
}

AOTIRuntimeError AOTInductorModelContainerUpdateInactiveConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle) {
  return AOTInductorModelContainerUpdateConstantBuffer(
      container_handle,
      constant_map_handle,
      /*use_inactive=*/true,
      /*validate_full_update=*/true);
}

AOTIRuntimeError AOTInductorModelContainerUpdateInactiveConstantBufferPairs(
    AOTInductorModelContainerHandle container_handle,
    const AOTInductorConstantMapEntry* pairs,
    size_t num_pairs) {
  return AOTInductorModelContainerUpdateConstantBufferPairs(
      container_handle,
      pairs,
      num_pairs,
      /*use_inactive=*/true,
      /*validate_full_update=*/true);
}

AOTIRuntimeError AOTInductorModelContainerFreeInactiveConstantBuffer(
    AOTInductorModelContainerHandle container_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->free_inactive_constant_buffer();
  })
}

AOTIRuntimeError AOTInductorModelContainerRunConstantFolding(
    AOTInductorModelContainerHandle container_handle,
    bool use_inactive,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto stream =
      reinterpret_cast<torch::aot_inductor::DeviceStreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    container->run_const_fold(use_inactive, stream, proxy_executor_handle);
  })
}

AOTIRuntimeError AOTInductorModelContainerSwapConstantBuffer(
    AOTInductorModelContainerHandle container_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->swap_constant_buffer();
  })
}

AOTIRuntimeError AOTInductorModelContainerGetNumInputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* ret_num_inputs) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_num_inputs = container->num_inputs(); })
}

AOTIRuntimeError AOTInductorModelContainerGetInputName(
    AOTInductorModelContainerHandle container_handle,
    size_t input_idx,
    const char** ret_input_names) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_input_names = container->input_name(input_idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetNumOutputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* ret_num_outputs) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_num_outputs = container->num_outputs(); })
}

AOTIRuntimeError AOTInductorModelContainerGetOutputName(
    AOTInductorModelContainerHandle container_handle,
    size_t output_idx,
    const char** ret_output_names) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_output_names = container->output_name(output_idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetCallSpec(
    AOTInductorModelContainerHandle container_handle,
    const char** in_spec,
    const char** out_spec) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    *in_spec = container->get_in_spec();
    *out_spec = container->get_out_spec();
  })
}

AOTIRuntimeError AOTInductorModelCreate(
    AOTInductorModelHandle* model_handle,
    AOTInductorConstantMapHandle constant_map_handle) {
  return createModelImpl(
      model_handle, constant_map_handle == nullptr, [=](auto& constant_map) {
        auto* input_map = reinterpret_cast<
            std::unordered_map<std::string, AtenTensorHandle>*>(
            constant_map_handle);
        if (input_map) {
          for (const auto& kv : *input_map) {
            constant_map.emplace(kv.first, kv.second);
          }
        }
      });
}

AOTIRuntimeError AOTInductorModelCreateV2(
    AOTInductorModelHandle* model_handle,
    const AOTInductorConstantMapEntry* pairs,
    size_t num_pairs) {
  return createModelImpl(
      model_handle, pairs == nullptr || num_pairs == 0, [=](auto& constant_map) {
        if (pairs && num_pairs > 0) {
          constant_map.reserve(num_pairs);
          for (size_t i = 0; i < num_pairs; ++i) {
            constant_map.emplace(pairs[i].name, pairs[i].handle);
          }
        }
      });
}

AOTIRuntimeError AOTInductorModelRun(
    AOTInductorModelHandle model_handle,
    AtenTensorHandle* input_handles,
    AtenTensorHandle* output_handles) {
  auto model =
      reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    model->run_impl(
        input_handles,
        output_handles,
        (torch::aot_inductor::DeviceStreamType) nullptr,
        nullptr);
  })
}

AOTIRuntimeError AOTInductorModelDelete(AOTInductorModelHandle model_handle){
    CONVERT_EXCEPTION_TO_ERROR_CODE({
      auto model = reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(
          model_handle);
      delete model;
    })}

AOTIRuntimeError AOTInductorModelGetNumOutputs(
    AOTInductorModelHandle model_handle,
    size_t* ret_num_outputs) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
      auto model = reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
      *ret_num_outputs = model->num_outputs();
  })
}

AOTIRuntimeError AOTInductorModelUpdateConstantsMap(
    AOTInductorModelHandle model_handle,
    AOTInductorConstantMapHandle constant_map_handle) {
  auto model =
      reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto constant_map = std::make_shared<torch::aot_inductor::ConstantMap>();
    auto input_map =
        reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(
            constant_map_handle);

    for (auto const& kv : *input_map) {
      constant_map->emplace(kv.first, kv.second);
    }
    model->update_constants_map(std::move(constant_map));
  })
}

// C-ABI-safe variant: uses an array of (name, handle) pairs instead of an
// opaque pointer to std::unordered_map, so the host and DSO can use
// different C++ standard libraries without ABI conflicts.
AOTIRuntimeError AOTInductorModelUpdateConstantsMapV2(
    AOTInductorModelHandle model_handle,
    const AOTInductorConstantMapEntry* pairs,
    int32_t num_pairs) {
  auto model =
      reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto constant_map = std::make_shared<torch::aot_inductor::ConstantMap>();
    constant_map->reserve(num_pairs);
    for (int32_t i = 0; i < num_pairs; ++i) {
      constant_map->emplace(pairs[i].name, pairs[i].handle);
    }
    model->update_constants_map(std::move(constant_map));
  })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantsBlobSize(
    AOTInductorModelContainerHandle container_handle,
    uint64_t* ret_size) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_size = container->constant_blob_size(); })
}


// Load weights from a single blob in weight_blob_ptr
AOTIRuntimeError AOTInductorModelUpdateConstantsFromBlob(
    AOTInductorModelContainerHandle container_handle,
    const uint8_t* weight_blob_ptr){
    auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      {container->update_constants_from_blob(weight_blob_ptr); })
    }


AOTIRuntimeError AOTInductorGetLastError(
    const char** error_msg) {
  *error_msg = g_aoti_last_error.c_str();
  return AOTI_RUNTIME_SUCCESS;
}

} // extern "C"


#define CUDA_DRIVER_CHECK(EXPR)                    \
do {                                               \
    CUresult code = EXPR;                          \
    const char *msg;                               \
    CUresult code_get_error = cuGetErrorString(code, &msg); \
    if (code_get_error != CUDA_SUCCESS) {          \
        throw std::runtime_error(                  \
            std::string("CUDA driver error: ") +   \
            std::string("invalid error code!"));   \
    }                                              \
    if (code != CUDA_SUCCESS) {                    \
        throw std::runtime_error(                  \
            std::string("CUDA driver error: ") +   \
            std::string(msg));                     \
    }                                              \
} while (0);

static inline CUfunction loadKernel(
        std::string filePath,
        const std::string &funcName,
        uint32_t sharedMemBytes,
        const std::optional<std::string> &cubinDir = std::nullopt,
        std::vector<CUmodule>* loaded_modules = nullptr) {
    if (cubinDir) {
        std::filesystem::path p1{*cubinDir};
        std::filesystem::path p2{filePath};
        filePath = (p1 / p2.filename()).string();
    }

    CUmodule mod;
    CUfunction func;
    CUDA_DRIVER_CHECK(cuModuleLoad(&mod, filePath.c_str()));
    if (loaded_modules) {
        loaded_modules->push_back(mod);
    }
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&func, mod, funcName.c_str()));
    if (sharedMemBytes > 0) {
        CUDA_DRIVER_CHECK(cuFuncSetAttribute(
            func,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            sharedMemBytes
        ))
    }
    return func;
}

static inline CUfunction loadKernel(
        const void* start,
        const std::string &funcName,
        uint32_t sharedMemBytes,
        std::vector<CUmodule>* loaded_modules = nullptr) {
    CUmodule mod;
    CUfunction func;
    CUDA_DRIVER_CHECK(cuModuleLoadData(&mod, start));
    if (loaded_modules) {
        loaded_modules->push_back(mod);
    }
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&func, mod, funcName.c_str()));
    if (sharedMemBytes > 0) {
        CUDA_DRIVER_CHECK(cuFuncSetAttribute(
            func,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            sharedMemBytes
        ))
    }
    return func;
}

static inline void launchKernel(
        CUfunction func,
        uint32_t gridX,
        uint32_t gridY,
        uint32_t gridZ,
        uint32_t numWarps,
        uint32_t sharedMemBytes,
        void* args[],
        cudaStream_t stream) {
    CUDA_DRIVER_CHECK(cuLaunchKernel(
        func, gridX, gridY, gridZ, 32*numWarps, 1, 1, sharedMemBytes, stream, args, nullptr
    ));
}
CACHE_TORCH_DTYPE(float32);
CACHE_TORCH_DEVICE(cuda);
CACHE_TORCH_LAYOUT(strided);
namespace torch::aot_inductor {
namespace {
class AOTInductorModelKernels : public AOTInductorModelKernelsBase {
  public:
    CUfunction triton_poi_fused_add_mul_relu_1{nullptr};
    CUfunction triton_poi_fused_add_relu_0{nullptr};
};
}  // namespace



AOTInductorModel::AOTInductorModel(std::shared_ptr<ConstantMap> constants_map,
                                   std::shared_ptr<std::vector<ConstantHandle>> constants_array,
                                   const std::string& device_str,
                                   std::optional<std::string> cubin_dir)
    : AOTInductorModelBase(1,
                           1,
                           24,
                           device_str,
                           std::move(cubin_dir),
                           false) {
    inputs_info_[0].name = "arg24_1";
    constants_info_[0].name = "w0";
    constants_info_[0].dtype = cached_torch_dtype_float32;
    constants_info_[0].device_type = cached_torch_device_type_cuda;
    constants_info_[0].offset = 0;
    constants_info_[0].data_size = 16384;
    constants_info_[0].from_folded = false;
    constants_info_[0].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[0].shape = {64, 64};
    constants_info_[0].stride = {64, 1};
    constants_info_[0].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[0].original_fqn = "w0";
    constants_info_[1].name = "b0";
    constants_info_[1].dtype = cached_torch_dtype_float32;
    constants_info_[1].device_type = cached_torch_device_type_cuda;
    constants_info_[1].offset = 0;
    constants_info_[1].data_size = 256;
    constants_info_[1].from_folded = false;
    constants_info_[1].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[1].shape = {64};
    constants_info_[1].stride = {1};
    constants_info_[1].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[1].original_fqn = "b0";
    constants_info_[2].name = "w1";
    constants_info_[2].dtype = cached_torch_dtype_float32;
    constants_info_[2].device_type = cached_torch_device_type_cuda;
    constants_info_[2].offset = 0;
    constants_info_[2].data_size = 16384;
    constants_info_[2].from_folded = false;
    constants_info_[2].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[2].shape = {64, 64};
    constants_info_[2].stride = {64, 1};
    constants_info_[2].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[2].original_fqn = "w1";
    constants_info_[3].name = "b1";
    constants_info_[3].dtype = cached_torch_dtype_float32;
    constants_info_[3].device_type = cached_torch_device_type_cuda;
    constants_info_[3].offset = 0;
    constants_info_[3].data_size = 256;
    constants_info_[3].from_folded = false;
    constants_info_[3].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[3].shape = {64};
    constants_info_[3].stride = {1};
    constants_info_[3].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[3].original_fqn = "b1";
    constants_info_[4].name = "w2";
    constants_info_[4].dtype = cached_torch_dtype_float32;
    constants_info_[4].device_type = cached_torch_device_type_cuda;
    constants_info_[4].offset = 0;
    constants_info_[4].data_size = 16384;
    constants_info_[4].from_folded = false;
    constants_info_[4].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[4].shape = {64, 64};
    constants_info_[4].stride = {64, 1};
    constants_info_[4].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[4].original_fqn = "w2";
    constants_info_[5].name = "b2";
    constants_info_[5].dtype = cached_torch_dtype_float32;
    constants_info_[5].device_type = cached_torch_device_type_cuda;
    constants_info_[5].offset = 0;
    constants_info_[5].data_size = 256;
    constants_info_[5].from_folded = false;
    constants_info_[5].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[5].shape = {64};
    constants_info_[5].stride = {1};
    constants_info_[5].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[5].original_fqn = "b2";
    constants_info_[6].name = "w3";
    constants_info_[6].dtype = cached_torch_dtype_float32;
    constants_info_[6].device_type = cached_torch_device_type_cuda;
    constants_info_[6].offset = 0;
    constants_info_[6].data_size = 16384;
    constants_info_[6].from_folded = false;
    constants_info_[6].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[6].shape = {64, 64};
    constants_info_[6].stride = {64, 1};
    constants_info_[6].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[6].original_fqn = "w3";
    constants_info_[7].name = "b3";
    constants_info_[7].dtype = cached_torch_dtype_float32;
    constants_info_[7].device_type = cached_torch_device_type_cuda;
    constants_info_[7].offset = 0;
    constants_info_[7].data_size = 256;
    constants_info_[7].from_folded = false;
    constants_info_[7].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[7].shape = {64};
    constants_info_[7].stride = {1};
    constants_info_[7].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[7].original_fqn = "b3";
    constants_info_[8].name = "w4";
    constants_info_[8].dtype = cached_torch_dtype_float32;
    constants_info_[8].device_type = cached_torch_device_type_cuda;
    constants_info_[8].offset = 0;
    constants_info_[8].data_size = 16384;
    constants_info_[8].from_folded = false;
    constants_info_[8].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[8].shape = {64, 64};
    constants_info_[8].stride = {64, 1};
    constants_info_[8].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[8].original_fqn = "w4";
    constants_info_[9].name = "b4";
    constants_info_[9].dtype = cached_torch_dtype_float32;
    constants_info_[9].device_type = cached_torch_device_type_cuda;
    constants_info_[9].offset = 0;
    constants_info_[9].data_size = 256;
    constants_info_[9].from_folded = false;
    constants_info_[9].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[9].shape = {64};
    constants_info_[9].stride = {1};
    constants_info_[9].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[9].original_fqn = "b4";
    constants_info_[10].name = "w5";
    constants_info_[10].dtype = cached_torch_dtype_float32;
    constants_info_[10].device_type = cached_torch_device_type_cuda;
    constants_info_[10].offset = 0;
    constants_info_[10].data_size = 16384;
    constants_info_[10].from_folded = false;
    constants_info_[10].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[10].shape = {64, 64};
    constants_info_[10].stride = {64, 1};
    constants_info_[10].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[10].original_fqn = "w5";
    constants_info_[11].name = "b5";
    constants_info_[11].dtype = cached_torch_dtype_float32;
    constants_info_[11].device_type = cached_torch_device_type_cuda;
    constants_info_[11].offset = 0;
    constants_info_[11].data_size = 256;
    constants_info_[11].from_folded = false;
    constants_info_[11].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[11].shape = {64};
    constants_info_[11].stride = {1};
    constants_info_[11].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[11].original_fqn = "b5";
    constants_info_[12].name = "w6";
    constants_info_[12].dtype = cached_torch_dtype_float32;
    constants_info_[12].device_type = cached_torch_device_type_cuda;
    constants_info_[12].offset = 0;
    constants_info_[12].data_size = 16384;
    constants_info_[12].from_folded = false;
    constants_info_[12].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[12].shape = {64, 64};
    constants_info_[12].stride = {64, 1};
    constants_info_[12].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[12].original_fqn = "w6";
    constants_info_[13].name = "b6";
    constants_info_[13].dtype = cached_torch_dtype_float32;
    constants_info_[13].device_type = cached_torch_device_type_cuda;
    constants_info_[13].offset = 0;
    constants_info_[13].data_size = 256;
    constants_info_[13].from_folded = false;
    constants_info_[13].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[13].shape = {64};
    constants_info_[13].stride = {1};
    constants_info_[13].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[13].original_fqn = "b6";
    constants_info_[14].name = "w7";
    constants_info_[14].dtype = cached_torch_dtype_float32;
    constants_info_[14].device_type = cached_torch_device_type_cuda;
    constants_info_[14].offset = 0;
    constants_info_[14].data_size = 16384;
    constants_info_[14].from_folded = false;
    constants_info_[14].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[14].shape = {64, 64};
    constants_info_[14].stride = {64, 1};
    constants_info_[14].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[14].original_fqn = "w7";
    constants_info_[15].name = "b7";
    constants_info_[15].dtype = cached_torch_dtype_float32;
    constants_info_[15].device_type = cached_torch_device_type_cuda;
    constants_info_[15].offset = 0;
    constants_info_[15].data_size = 256;
    constants_info_[15].from_folded = false;
    constants_info_[15].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[15].shape = {64};
    constants_info_[15].stride = {1};
    constants_info_[15].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[15].original_fqn = "b7";
    constants_info_[16].name = "w8";
    constants_info_[16].dtype = cached_torch_dtype_float32;
    constants_info_[16].device_type = cached_torch_device_type_cuda;
    constants_info_[16].offset = 0;
    constants_info_[16].data_size = 16384;
    constants_info_[16].from_folded = false;
    constants_info_[16].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[16].shape = {64, 64};
    constants_info_[16].stride = {64, 1};
    constants_info_[16].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[16].original_fqn = "w8";
    constants_info_[17].name = "b8";
    constants_info_[17].dtype = cached_torch_dtype_float32;
    constants_info_[17].device_type = cached_torch_device_type_cuda;
    constants_info_[17].offset = 0;
    constants_info_[17].data_size = 256;
    constants_info_[17].from_folded = false;
    constants_info_[17].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[17].shape = {64};
    constants_info_[17].stride = {1};
    constants_info_[17].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[17].original_fqn = "b8";
    constants_info_[18].name = "w9";
    constants_info_[18].dtype = cached_torch_dtype_float32;
    constants_info_[18].device_type = cached_torch_device_type_cuda;
    constants_info_[18].offset = 0;
    constants_info_[18].data_size = 16384;
    constants_info_[18].from_folded = false;
    constants_info_[18].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[18].shape = {64, 64};
    constants_info_[18].stride = {64, 1};
    constants_info_[18].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[18].original_fqn = "w9";
    constants_info_[19].name = "b9";
    constants_info_[19].dtype = cached_torch_dtype_float32;
    constants_info_[19].device_type = cached_torch_device_type_cuda;
    constants_info_[19].offset = 0;
    constants_info_[19].data_size = 256;
    constants_info_[19].from_folded = false;
    constants_info_[19].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[19].shape = {64};
    constants_info_[19].stride = {1};
    constants_info_[19].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[19].original_fqn = "b9";
    constants_info_[20].name = "w10";
    constants_info_[20].dtype = cached_torch_dtype_float32;
    constants_info_[20].device_type = cached_torch_device_type_cuda;
    constants_info_[20].offset = 0;
    constants_info_[20].data_size = 16384;
    constants_info_[20].from_folded = false;
    constants_info_[20].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[20].shape = {64, 64};
    constants_info_[20].stride = {64, 1};
    constants_info_[20].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[20].original_fqn = "w10";
    constants_info_[21].name = "b10";
    constants_info_[21].dtype = cached_torch_dtype_float32;
    constants_info_[21].device_type = cached_torch_device_type_cuda;
    constants_info_[21].offset = 0;
    constants_info_[21].data_size = 256;
    constants_info_[21].from_folded = false;
    constants_info_[21].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[21].shape = {64};
    constants_info_[21].stride = {1};
    constants_info_[21].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[21].original_fqn = "b10";
    constants_info_[22].name = "w11";
    constants_info_[22].dtype = cached_torch_dtype_float32;
    constants_info_[22].device_type = cached_torch_device_type_cuda;
    constants_info_[22].offset = 0;
    constants_info_[22].data_size = 16384;
    constants_info_[22].from_folded = false;
    constants_info_[22].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[22].shape = {64, 64};
    constants_info_[22].stride = {64, 1};
    constants_info_[22].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[22].original_fqn = "w11";
    constants_info_[23].name = "b11";
    constants_info_[23].dtype = cached_torch_dtype_float32;
    constants_info_[23].device_type = cached_torch_device_type_cuda;
    constants_info_[23].offset = 0;
    constants_info_[23].data_size = 256;
    constants_info_[23].from_folded = false;
    constants_info_[23].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Buffer);
    constants_info_[23].shape = {64};
    constants_info_[23].stride = {1};
    constants_info_[23].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[23].original_fqn = "b11";
    update_constants_map(std::move(constants_map));
    update_constants_array(std::move(constants_array));
    in_spec_ = R"([1, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.dict", "context": "[]", "children_spec": []}]}])";
    out_spec_ = R"([1, {"type": null, "context": null, "children_spec": []}])";
    outputs_info_[0].name = "output0";
    this->kernels_ = std::make_unique<AOTInductorModelKernels>();
}

std::unordered_map<std::string, AtenTensorHandle> AOTInductorModel::const_run_impl(
    DeviceStreamType stream,
    AOTIProxyExecutorHandle proxy_executor,
    bool initialization
) {
    
                std::unordered_map<std::string, AtenTensorHandle> folded_constants_map;
                folded_constants_map.reserve(0);
                std::vector<AtenTensorHandle> output_handles(0);
                

    // The below assignment of output_handles to constants is not used directly.
    // It's only used to memo the correspondence of handle and constants.
    _const_run_impl(output_handles, stream, proxy_executor);
    return folded_constants_map;
}
} // namespace torch::aot_inductor
using namespace torch::aot_inductor;

template <typename in_out_ptr0_type_, typename in_ptr0_type_, typename kernels_type_>
static __attribute__((noinline)) void call_triton_poi_fused_add_relu_0(
    const in_out_ptr0_type_& in_out_ptr0,
    const in_ptr0_type_& in_ptr0,
    int64_t xnumel,
    int32_t device_idx_,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    /*
    async_compile.triton('triton_poi_fused_add_relu_0', '''
    import triton
    import triton.language as tl

    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
    from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.pointwise(
        size_hints={'x': 512}, 
        filename=__file__,
        triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
        inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_relu_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 6400}, 'backend_hash': 'DDAB982AB4538BC5DBDD7197E3A5A3077782114B9DF323E1A880B8C44609DD82', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
        min_elem_per_thread=0
    )
    @triton.jit
    def triton_poi_fused_add_relu_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
        xnumel = 512
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = xindex
        x0 = (xindex % 64)
        tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
        tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tmp3 = tl.full([1], 0, tl.int32)
        tmp4 = triton_helpers.maximum(tmp3, tmp2)
        tl.store(in_out_ptr0 + (x2), tmp4, xmask)
    ''', device_str='cuda')
    */
    uint32_t grid_0 = ((xnumel + (128 - 1)) / (128));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_poi_fused_add_relu_0 == nullptr) {
        kernels_.triton_poi_fused_add_relu_0 = loadKernel("/root/camp2/capture/inductor-cache/cq2adewgv4ave35v4etao54au3yqcbfq4gnvh32amfj2yb5jo4sr/c4gpt4f232djounlhjynikbqazkfh5ytent7arxzdwnjvkiu4hek.cubin", "triton_poi_fused_add_relu_0", 0, cubin_dir_, &kernels_.loaded_modules_); 
    }
    CUdeviceptr var_0 = reinterpret_cast<CUdeviceptr>(in_out_ptr0.data_ptr());
    CUdeviceptr var_1 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    int32_t var_2 = xnumel;
    CUdeviceptr global_scratch_scratch_3 = 0;
    CUdeviceptr profile_scratch_scratch_4 = 0;
    void* kernel_args_[] = {&var_0, &var_1, &var_2, &global_scratch_scratch_3, &profile_scratch_scratch_4};
    launchKernel(kernels_.triton_poi_fused_add_relu_0, grid_0, grid_1, grid_2, 4, 0, kernel_args_, stream_);
}

template <typename in_out_ptr0_type_, typename in_ptr0_type_, typename kernels_type_>
static __attribute__((noinline)) void call_triton_poi_fused_add_mul_relu_1(
    const in_out_ptr0_type_& in_out_ptr0,
    const in_ptr0_type_& in_ptr0,
    int64_t xnumel,
    int32_t device_idx_,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    /*
    async_compile.triton('triton_poi_fused_add_mul_relu_1', '''
    import triton
    import triton.language as tl

    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
    from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.pointwise(
        size_hints={'x': 512}, 
        filename=__file__,
        triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
        inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_mul_relu_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 6400}, 'backend_hash': 'DDAB982AB4538BC5DBDD7197E3A5A3077782114B9DF323E1A880B8C44609DD82', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
        min_elem_per_thread=0
    )
    @triton.jit
    def triton_poi_fused_add_mul_relu_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
        xnumel = 512
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = xindex
        x0 = (xindex % 64)
        tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
        tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tmp3 = tl.full([1], 0, tl.int32)
        tmp4 = triton_helpers.maximum(tmp3, tmp2)
        tmp5 = tl.full([1], 0.5, tl.float32)
        tmp6 = tmp4 * tmp5
        tmp7 = tl.full([1], 1.0, tl.float32)
        tmp8 = tmp6 + tmp7
        tl.store(in_out_ptr0 + (x2), tmp8, xmask)
    ''', device_str='cuda')
    */
    uint32_t grid_0 = ((xnumel + (256 - 1)) / (256));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_poi_fused_add_mul_relu_1 == nullptr) {
        kernels_.triton_poi_fused_add_mul_relu_1 = loadKernel("/root/camp2/capture/inductor-cache/cq2adewgv4ave35v4etao54au3yqcbfq4gnvh32amfj2yb5jo4sr/cx5pequfrqgkoarbkod4wrqm7qw5xs6nlfn2cvacfawcrashbyev.cubin", "triton_poi_fused_add_mul_relu_1", 0, cubin_dir_, &kernels_.loaded_modules_); 
    }
    CUdeviceptr var_5 = reinterpret_cast<CUdeviceptr>(in_out_ptr0.data_ptr());
    CUdeviceptr var_6 = reinterpret_cast<CUdeviceptr>(in_ptr0.data_ptr());
    int32_t var_7 = xnumel;
    CUdeviceptr global_scratch_scratch_8 = 0;
    CUdeviceptr profile_scratch_scratch_9 = 0;
    void* kernel_args_[] = {&var_5, &var_6, &var_7, &global_scratch_scratch_8, &profile_scratch_scratch_9};
    launchKernel(kernels_.triton_poi_fused_add_mul_relu_1, grid_0, grid_1, grid_2, 4, 0, kernel_args_, stream_);
}

namespace torch::aot_inductor {

void AOTInductorModel::_const_run_impl(
    std::vector<AtenTensorHandle>& output_handles,
    DeviceStreamType stream,
    AOTIProxyExecutorHandle proxy_executor
) {
    [[maybe_unused]] auto& kernels = static_cast<AOTInductorModelKernels&>(*this->kernels_.get());
} // AOTInductorModel::_const_run_impl

AOTI_NOINLINE static void check_input_0(
    AtenTensorHandle* input_handles
) {
    ConstantHandle arg24_1 = ConstantHandle(input_handles[0]);
    int32_t arg24_1_dtype;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(arg24_1, &arg24_1_dtype));

    int32_t arg24_1_expected_dtype = aoti_torch_dtype_float32();
    if (arg24_1_expected_dtype != arg24_1_dtype) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched dtype, "
           << "expected: " << arg24_1_expected_dtype << "(at::kFloat), "
           << "but got: " << arg24_1_dtype << "\n";
        throw std::runtime_error(std::move(ss).str());
    }
    auto arg24_1_size = arg24_1.sizes();

    if (8 != arg24_1_size[0]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched dim value at 0, "
           << "expected: 8, " << "but got: " << arg24_1_size[0]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }

    if (64 != arg24_1_size[1]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched dim value at 1, "
           << "expected: 64, " << "but got: " << arg24_1_size[1]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }
    auto arg24_1_stride = arg24_1.strides();

    if (64 != arg24_1_stride[0]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched stride value at 0, "
           << "expected: 64, " << "but got: " << arg24_1_stride[0]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }

    if (1 != arg24_1_stride[1]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched stride value at 1, "
           << "expected: 1, " << "but got: " << arg24_1_stride[1]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }
    int32_t arg24_1_device_type;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_type(arg24_1, &arg24_1_device_type));

    int32_t arg24_1_expected_device_type = 1;
    if (arg24_1_expected_device_type != arg24_1_device_type) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched device type, "
        << "expected: " << arg24_1_expected_device_type << "1(cuda), "
        << "but got: " << arg24_1_device_type << "\n";
        throw std::runtime_error(std::move(ss).str());
    }
}

static bool _check_aoti_runtime_check_inputs_env() {
    const static char* env_var_value = getenv("AOTI_RUNTIME_CHECK_INPUTS");
    const static bool result = env_var_value != nullptr && env_var_value[0] != '0';
    return result;
}

AOTI_NOINLINE static void __check_inputs_outputs(
    AtenTensorHandle* input_handles,
    AtenTensorHandle* output_handles) {
    if (!_check_aoti_runtime_check_inputs_env()){
        return;
    }
    check_input_0(input_handles);
}

void AOTInductorModel::run_impl(
    AtenTensorHandle*
        input_handles, // array of input AtenTensorHandle; handles
                        // are stolen; the array itself is borrowed
    AtenTensorHandle*
        output_handles, // array for writing output AtenTensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed
    DeviceStreamType stream,
    AOTIProxyExecutorHandle proxy_executor
) {
    __check_inputs_outputs(input_handles, output_handles);
    auto inputs = steal_from_raw_handles_to_raii_handles(input_handles, 1);
    auto arg24_1 = std::move(inputs[0]);
    [[maybe_unused]] auto& w0 = constants_->at(0);
    [[maybe_unused]] auto& b0 = constants_->at(1);
    [[maybe_unused]] auto& w1 = constants_->at(2);
    [[maybe_unused]] auto& b1 = constants_->at(3);
    [[maybe_unused]] auto& w2 = constants_->at(4);
    [[maybe_unused]] auto& b2 = constants_->at(5);
    [[maybe_unused]] auto& w3 = constants_->at(6);
    [[maybe_unused]] auto& b3 = constants_->at(7);
    [[maybe_unused]] auto& w4 = constants_->at(8);
    [[maybe_unused]] auto& b4 = constants_->at(9);
    [[maybe_unused]] auto& w5 = constants_->at(10);
    [[maybe_unused]] auto& b5 = constants_->at(11);
    [[maybe_unused]] auto& w6 = constants_->at(12);
    [[maybe_unused]] auto& b6 = constants_->at(13);
    [[maybe_unused]] auto& w7 = constants_->at(14);
    [[maybe_unused]] auto& b7 = constants_->at(15);
    [[maybe_unused]] auto& w8 = constants_->at(16);
    [[maybe_unused]] auto& b8 = constants_->at(17);
    [[maybe_unused]] auto& w9 = constants_->at(18);
    [[maybe_unused]] auto& b9 = constants_->at(19);
    [[maybe_unused]] auto& w10 = constants_->at(20);
    [[maybe_unused]] auto& b10 = constants_->at(21);
    [[maybe_unused]] auto& w11 = constants_->at(22);
    [[maybe_unused]] auto& b11 = constants_->at(23);

    if ((reinterpret_cast<std::uintptr_t>(arg24_1.data_ptr()) & (16 -1)) != 0) {
        AOTI_TORCH_WARN("Input 0 was compiled as 16-bytes aligned, but it is not aligned at run time. Copying to an aligned tensor to guarantee correctness, but expect a performance hit.");
        AtenTensorHandle arg24_1_aligned;
        aoti_torch_clone_preserve_strides(arg24_1, &arg24_1_aligned);
        arg24_1 = std::move(RAIIAtenTensorHandle(arg24_1_aligned));
    }
    inputs.clear();
    [[maybe_unused]] auto& kernels = static_cast<AOTInductorModelKernels&>(*this->kernels_.get());
    if (_check_aoti_runtime_check_inputs_env()) { assert_size_stride(arg24_1, {8L, 64L}, {64L, 1L}, "input"); }

    AOTICudaStreamGuard stream_guard(stream, this->device_idx_);
    static constexpr int64_t int_array_0[] = {8L, 64L};
    static constexpr int64_t int_array_1[] = {64L, 1L};
    AtenTensorHandle buf0_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(2, int_array_0, int_array_1, cached_torch_dtype_float32, cached_torch_device_type_cuda, this->device_idx_, &buf0_handle));
    RAIIAtenTensorHandle buf0(buf0_handle);
    // Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.mm]
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_mm_out(buf0, arg24_1, w0));
    arg24_1.reset();
    auto buf1 = std::move(buf0);  // reuse
    // Topologically Sorted Source Nodes: [add, relu], Original ATen: [aten.add, aten.relu]
    call_triton_poi_fused_add_relu_0(buf1, b0, 512L, this->device_idx_, stream, kernels, this->cubin_dir_);
    AtenTensorHandle buf2_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(2, int_array_0, int_array_1, cached_torch_dtype_float32, cached_torch_device_type_cuda, this->device_idx_, &buf2_handle));
    RAIIAtenTensorHandle buf2(buf2_handle);
    // Topologically Sorted Source Nodes: [add, relu, matmul_1], Original ATen: [aten.add, aten.relu, aten.mm]
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_mm_out(buf2, buf1, w1));
    buf1.reset();
    auto buf3 = std::move(buf2);  // reuse
    // Topologically Sorted Source Nodes: [add_1, relu_1], Original ATen: [aten.add, aten.relu]
    call_triton_poi_fused_add_relu_0(buf3, b1, 512L, this->device_idx_, stream, kernels, this->cubin_dir_);
    AtenTensorHandle buf4_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(2, int_array_0, int_array_1, cached_torch_dtype_float32, cached_torch_device_type_cuda, this->device_idx_, &buf4_handle));
    RAIIAtenTensorHandle buf4(buf4_handle);
    // Topologically Sorted Source Nodes: [add_1, relu_1, matmul_2], Original ATen: [aten.add, aten.relu, aten.mm]
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_mm_out(buf4, buf3, w2));
    buf3.reset();
    auto buf5 = std::move(buf4);  // reuse
    // Topologically Sorted Source Nodes: [add_2, relu_2], Original ATen: [aten.add, aten.relu]
    call_triton_poi_fused_add_relu_0(buf5, b2, 512L, this->device_idx_, stream, kernels, this->cubin_dir_);
    AtenTensorHandle buf6_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(2, int_array_0, int_array_1, cached_torch_dtype_float32, cached_torch_device_type_cuda, this->device_idx_, &buf6_handle));
    RAIIAtenTensorHandle buf6(buf6_handle);
    // Topologically Sorted Source Nodes: [add_2, relu_2, matmul_3], Original ATen: [aten.add, aten.relu, aten.mm]
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_mm_out(buf6, buf5, w3));
    buf5.reset();
    auto buf7 = std::move(buf6);  // reuse
    // Topologically Sorted Source Nodes: [add_3, relu_3], Original ATen: [aten.add, aten.relu]
    call_triton_poi_fused_add_relu_0(buf7, b3, 512L, this->device_idx_, stream, kernels, this->cubin_dir_);
    AtenTensorHandle buf8_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(2, int_array_0, int_array_1, cached_torch_dtype_float32, cached_torch_device_type_cuda, this->device_idx_, &buf8_handle));
    RAIIAtenTensorHandle buf8(buf8_handle);
    // Topologically Sorted Source Nodes: [add_3, relu_3, matmul_4], Original ATen: [aten.add, aten.relu, aten.mm]
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_mm_out(buf8, buf7, w4));
    buf7.reset();
    auto buf9 = std::move(buf8);  // reuse
    // Topologically Sorted Source Nodes: [add_4, relu_4], Original ATen: [aten.add, aten.relu]
    call_triton_poi_fused_add_relu_0(buf9, b4, 512L, this->device_idx_, stream, kernels, this->cubin_dir_);
    AtenTensorHandle buf10_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(2, int_array_0, int_array_1, cached_torch_dtype_float32, cached_torch_device_type_cuda, this->device_idx_, &buf10_handle));
    RAIIAtenTensorHandle buf10(buf10_handle);
    // Topologically Sorted Source Nodes: [add_4, relu_4, matmul_5], Original ATen: [aten.add, aten.relu, aten.mm]
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_mm_out(buf10, buf9, w5));
    buf9.reset();
    auto buf11 = std::move(buf10);  // reuse
    // Topologically Sorted Source Nodes: [add_5, relu_5], Original ATen: [aten.add, aten.relu]
    call_triton_poi_fused_add_relu_0(buf11, b5, 512L, this->device_idx_, stream, kernels, this->cubin_dir_);
    AtenTensorHandle buf12_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(2, int_array_0, int_array_1, cached_torch_dtype_float32, cached_torch_device_type_cuda, this->device_idx_, &buf12_handle));
    RAIIAtenTensorHandle buf12(buf12_handle);
    // Topologically Sorted Source Nodes: [add_5, relu_5, matmul_6], Original ATen: [aten.add, aten.relu, aten.mm]
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_mm_out(buf12, buf11, w6));
    buf11.reset();
    auto buf13 = std::move(buf12);  // reuse
    // Topologically Sorted Source Nodes: [add_6, relu_6], Original ATen: [aten.add, aten.relu]
    call_triton_poi_fused_add_relu_0(buf13, b6, 512L, this->device_idx_, stream, kernels, this->cubin_dir_);
    AtenTensorHandle buf14_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(2, int_array_0, int_array_1, cached_torch_dtype_float32, cached_torch_device_type_cuda, this->device_idx_, &buf14_handle));
    RAIIAtenTensorHandle buf14(buf14_handle);
    // Topologically Sorted Source Nodes: [add_6, relu_6, matmul_7], Original ATen: [aten.add, aten.relu, aten.mm]
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_mm_out(buf14, buf13, w7));
    buf13.reset();
    auto buf15 = std::move(buf14);  // reuse
    // Topologically Sorted Source Nodes: [add_7, relu_7], Original ATen: [aten.add, aten.relu]
    call_triton_poi_fused_add_relu_0(buf15, b7, 512L, this->device_idx_, stream, kernels, this->cubin_dir_);
    AtenTensorHandle buf16_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(2, int_array_0, int_array_1, cached_torch_dtype_float32, cached_torch_device_type_cuda, this->device_idx_, &buf16_handle));
    RAIIAtenTensorHandle buf16(buf16_handle);
    // Topologically Sorted Source Nodes: [add_7, relu_7, matmul_8], Original ATen: [aten.add, aten.relu, aten.mm]
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_mm_out(buf16, buf15, w8));
    buf15.reset();
    auto buf17 = std::move(buf16);  // reuse
    // Topologically Sorted Source Nodes: [add_8, relu_8], Original ATen: [aten.add, aten.relu]
    call_triton_poi_fused_add_relu_0(buf17, b8, 512L, this->device_idx_, stream, kernels, this->cubin_dir_);
    AtenTensorHandle buf18_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(2, int_array_0, int_array_1, cached_torch_dtype_float32, cached_torch_device_type_cuda, this->device_idx_, &buf18_handle));
    RAIIAtenTensorHandle buf18(buf18_handle);
    // Topologically Sorted Source Nodes: [add_8, relu_8, matmul_9], Original ATen: [aten.add, aten.relu, aten.mm]
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_mm_out(buf18, buf17, w9));
    buf17.reset();
    auto buf19 = std::move(buf18);  // reuse
    // Topologically Sorted Source Nodes: [add_9, relu_9], Original ATen: [aten.add, aten.relu]
    call_triton_poi_fused_add_relu_0(buf19, b9, 512L, this->device_idx_, stream, kernels, this->cubin_dir_);
    AtenTensorHandle buf20_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(2, int_array_0, int_array_1, cached_torch_dtype_float32, cached_torch_device_type_cuda, this->device_idx_, &buf20_handle));
    RAIIAtenTensorHandle buf20(buf20_handle);
    // Topologically Sorted Source Nodes: [add_9, relu_9, matmul_10], Original ATen: [aten.add, aten.relu, aten.mm]
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_mm_out(buf20, buf19, w10));
    buf19.reset();
    auto buf21 = std::move(buf20);  // reuse
    // Topologically Sorted Source Nodes: [add_10, relu_10], Original ATen: [aten.add, aten.relu]
    call_triton_poi_fused_add_relu_0(buf21, b10, 512L, this->device_idx_, stream, kernels, this->cubin_dir_);
    AtenTensorHandle buf22_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(2, int_array_0, int_array_1, cached_torch_dtype_float32, cached_torch_device_type_cuda, this->device_idx_, &buf22_handle));
    RAIIAtenTensorHandle buf22(buf22_handle);
    // Topologically Sorted Source Nodes: [add_10, relu_10, matmul_11], Original ATen: [aten.add, aten.relu, aten.mm]
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_mm_out(buf22, buf21, w11));
    buf21.reset();
    auto buf23 = std::move(buf22);  // reuse
    // Topologically Sorted Source Nodes: [add_11, relu_11, mul, add_12], Original ATen: [aten.add, aten.relu, aten.mul]
    call_triton_poi_fused_add_mul_relu_1(buf23, b11, 512L, this->device_idx_, stream, kernels, this->cubin_dir_);
    output_handles[0] = buf23.release();
} // AOTInductorModel::run_impl
} // namespace torch::aot_inductor




// Compile cmd
// g++ /root/camp2/capture/inductor-cache/cq2adewgv4ave35v4etao54au3yqcbfq4gnvh32amfj2yb5jo4sr/c6mzleoieaannxdxt736k25ubbrzyrjygvlyjxo54guc3umcxmbi.wrapper.cpp -D TORCH_INDUCTOR_CPP_WRAPPER -D STANDALONE_TORCH_HEADER -D TORCH_INDUCTOR_PRECOMPILE_HEADERS -D  C10_USING_CUSTOM_GENERATED_MACROS -D CPU_CAPABILITY_AVX2 -D  USE_CUDA  -O1 -DNDEBUG -fno-omit-frame-pointer -g1 -fno-trapping-math -funsafe-math-optimizations -ffinite-math-only -fno-signed-zeros -fno-finite-math-only -fno-unsafe-math-optimizations -fmath-errno -ffp-contract=off -fexcess-precision=fast -fno-tree-loop-vectorize -march=x86-64-v3 -fPIC -Wall -std=c++20 -Wno-unused-variable -Wno-unknown-pragmas -pedantic -fopenmp  -include /tmp/torchinductor_root/precompiled_headers/ctgrya2hexb25sggi27jchuyrvwwatqj3d4wtl5vaagqvpwqu5vt.h -I/usr/include/python3.11 -I/usr/local/lib/python3.11/dist-packages/torch/include -I/usr/local/lib/python3.11/dist-packages/torch/include/torch/csrc/api/include -I/usr/local/cuda/include   -mavx2 -mfma -mf16c  -c -o /root/camp2/capture/inductor-cache/cq2adewgv4ave35v4etao54au3yqcbfq4gnvh32amfj2yb5jo4sr/c6mzleoieaannxdxt736k25ubbrzyrjygvlyjxo54guc3umcxmbi.wrapper.o
// Link cmd
// g++ /root/camp2/capture/inductor-cache/cq2adewgv4ave35v4etao54au3yqcbfq4gnvh32amfj2yb5jo4sr/c6mzleoieaannxdxt736k25ubbrzyrjygvlyjxo54guc3umcxmbi.wrapper.o /root/camp2/capture/inductor-cache/cq2adewgv4ave35v4etao54au3yqcbfq4gnvh32amfj2yb5jo4sr/cbarkoqitjvrvdqmh7rwjtrbjqp5tnvn7zhhryzos3erowthtvhz.kernel.o /root/camp2/capture/inductor-cache/cq2adewgv4ave35v4etao54au3yqcbfq4gnvh32amfj2yb5jo4sr/c6mzleoieaannxdxt736k25ubbrzyrjygvlyjxo54guc3umcxmbi/cgxre62qccm2hvjt32u3oobekucw5zks6ukxlzod6km7a4telymf.weights.o -D TORCH_INDUCTOR_CPP_WRAPPER -D STANDALONE_TORCH_HEADER -D TORCH_INDUCTOR_PRECOMPILE_HEADERS -D  C10_USING_CUSTOM_GENERATED_MACROS -D CPU_CAPABILITY_AVX2 -D  USE_CUDA  -O3 -DNDEBUG -fno-omit-frame-pointer -g1 -fno-trapping-math -funsafe-math-optimizations -ffinite-math-only -fno-signed-zeros -fno-finite-math-only -fno-unsafe-math-optimizations -fmath-errno -ffp-contract=off -fexcess-precision=fast -fno-tree-loop-vectorize -march=x86-64-v3 -shared -fPIC -Wall -std=c++20 -Wno-unused-variable -Wno-unknown-pragmas -pedantic -fopenmp  -I/usr/include/python3.11 -I/usr/local/lib/python3.11/dist-packages/torch/include -I/usr/local/lib/python3.11/dist-packages/torch/include/torch/csrc/api/include -I/usr/local/cuda/include   -mavx2 -mfma -mf16c  -o /root/camp2/capture/inductor-cache/cq2adewgv4ave35v4etao54au3yqcbfq4gnvh32amfj2yb5jo4sr/c6mzleoieaannxdxt736k25ubbrzyrjygvlyjxo54guc3umcxmbi.wrapper.so  -ltorch -ltorch_cpu -lgomp -lcuda -ltorch_cuda  -L/usr/lib/x86_64-linux-gnu -L/usr/local/lib/python3.11/dist-packages/torch/lib -L/usr/local/cuda/lib64 
