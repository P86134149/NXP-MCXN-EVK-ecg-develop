/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
   Copyright 2021-2023 NXP

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/* File modified by NXP. Changes are described in file
   /middleware/eiq/tensorflow-lite/readme.txt in section "Release notes" */

#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <cmath>
#include "fsl_debug_console.h"
#include "model.h"
#include "MIX_LMUEBCnet_1D_normalized_converted.h"
//#include "model_data.h"


static const tflite::Model* s_model = nullptr;
static tflite::MicroInterpreter* s_interpreter = nullptr;

extern tflite::MicroOpResolver &MODEL_GetOpsResolver();

// An area of memory to use for input, output, and intermediate arrays.
// (Can be adjusted based on the model needs.)
static uint8_t s_tensorArena[kTensorArenaSize] __ALIGNED(16);

status_t MODEL_Init(void)
{
    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    s_model = tflite::GetModel(model_data);
    if (s_model->version() != TFLITE_SCHEMA_VERSION)
    {
        PRINTF("Model provided is schema version %d not equal "
               "to supported version %d!\r\n",
               s_model->version(), TFLITE_SCHEMA_VERSION);
        return kStatus_Fail;
    }

    // Pull in only the operation implementations we need.
    // This relies on a complete list of all the ops needed by this graph.
    // NOLINTNEXTLINE(runtime-global-variables)
    tflite::MicroOpResolver &micro_op_resolver = MODEL_GetOpsResolver();

    // Build an interpreter to run the model with.
    static tflite::MicroInterpreter static_interpreter(
        s_model, micro_op_resolver, s_tensorArena, kTensorArenaSize);
    s_interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = s_interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk)
    {
        PRINTF("AllocateTensors() failed!\r\n");
        return kStatus_Fail;
    }

    return kStatus_Success;
}

status_t MODEL_RunInference(void)
{
    if (s_interpreter->Invoke() != kTfLiteOk)
    {
        PRINTF("Invoke failed!\r\n");
        return kStatus_Fail;
    }

    return kStatus_Success;
}

void* GetTensorData(TfLiteTensor* tensor, tensor_dims_t* dims, tensor_type_t* type)
{
    switch (tensor->type)
    {
        case kTfLiteFloat32:
            *type = kTensorType_FLOAT32;
            break;
        case kTfLiteUInt8:
            *type = kTensorType_UINT8;
            break;
        case kTfLiteInt8:
            *type = kTensorType_INT8;
            break;
        default:
            assert("Unknown input tensor data type!\r\n");
            return nullptr;
    }

    dims->size = tensor->dims->size;
    assert(dims->size <= MAX_TENSOR_DIMS);
    for (int i = 0; i < tensor->dims->size; i++)
    {
        dims->data[i] = tensor->dims->data[i];
    }

    // 根據 tensor->type 回傳對應的數據指標
    if (*type == kTensorType_FLOAT32) {
        return tensor->data.f;
    } else if (*type == kTensorType_UINT8) {
        return tensor->data.uint8;
    } else {
        return tensor->data.int8;
    }
}


//uint8_t* MODEL_GetInputTensorData(tensor_dims_t* dims, tensor_type_t* type)
//{
//    TfLiteTensor* inputTensor = s_interpreter->input(0);
//
//    return GetTensorData(inputTensor, dims, type);
//}

void* MODEL_GetInputTensorData(tensor_dims_t* dims, tensor_type_t* type)
{
    TfLiteTensor* inputTensor = s_interpreter->input(0);
    return GetTensorData(inputTensor, dims, type);
}

//uint8_t* MODEL_GetOutputTensorData(tensor_dims_t* dims, tensor_type_t* type)
//{
//    TfLiteTensor* outputTensor = s_interpreter->output(0);
//
//    return GetTensorData(outputTensor, dims, type);
//}

void* MODEL_GetOutputTensorData(tensor_dims_t* dims, tensor_type_t* type)
{
    TfLiteTensor* outputTensor = s_interpreter->output(0);
    return GetTensorData(outputTensor, dims, type);
}

// Convert unsigned 8-bit image data to model input format in-place.
void MODEL_ConvertInput(void* data, tensor_dims_t* dims, tensor_type_t type)
{
    int size = dims->data[1] * dims->data[2];

    switch (type)
    {
        case kTensorType_UINT8:
            break;
        // PCQ quantize
        case kTensorType_INT8:
        {
            float* f32Data = reinterpret_cast<float*>(data);
            int8_t* int8Data = reinterpret_cast<int8_t*>(data);

            // Example: Assume you have 1 channel and the scale is per-channel
            // In practice, you should load `scale[channel]` from the model!
            float per_channel_scales[1] = { 1.0f / 127.0f };  // Example: Get this from TFLite model
//            const int32_t zero_point = 0;  // Symmetric quantization means zero_point is always 0

            int channel = 0;  // Assuming 1D ECG, you may only have 1 channel

            for (int i = 0; i < size; i++)
            {
                float value = f32Data[i];

                // Apply per-channel quantization (use different scale per channel if needed)
                int32_t quantized_value = static_cast<int32_t>(std::round(value / per_channel_scales[channel]));

                // Clamp to int8 range [-128, 127]
                quantized_value = (quantized_value > 127) ? 127 : ((quantized_value < -128) ? -128 : quantized_value);

                int8Data[i] = static_cast<int8_t>(quantized_value);
            }
            break;
        }
//            int8_t* intData = (int8_t*)data;
//            for (int i = size - 1; i >= 0; i--)
//            {
//                intData[i] = static_cast<int>(intData[i]) - 127;
//            }
//            break;
        case kTensorType_FLOAT32:
            // 這裡不再做正規化，假設 `main()` 已經標準化處理
            break;
        default:
            assert("Unknown input tensor data type!\r\n");
    }
}

//void MODEL_ConvertInput(uint8_t* data, tensor_dims_t* dims, tensor_type_t type)
//{
//    int size = 1;
//    for (int i = 1; i < dims->size; i++)  // 避免影響 batch_size (dims->data[0])
//    {
//        size *= dims->data[i];
//    }
//
//    switch (type)
//    {
//        case kTensorType_UINT8:
//            break;
//        case kTensorType_INT8:
//            for (int i = 0; i < size; i++)
//            {
//                reinterpret_cast<int8_t*>(data)[i] =
//                    static_cast<int>(data[i]) - 127;
//            }
//            break;
//        case kTensorType_FLOAT32:
//            for (int i = 0; i < size; i++)
//            {
//                reinterpret_cast<float*>(data)[i] =
//                    (static_cast<int>(data[i]) - MODEL_INPUT_MEAN) / MODEL_INPUT_STD;
//            }
//            break;
//        default:
//            assert("Unknown input tensor data type!\r\n");
//    }
//}


const char* MODEL_GetModelName(void)
{
    return MODEL_NAME;
}
