/*
 * Copyright 2020-2022 NXP
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "fsl_debug_console.h"
#include "output_postproc.h"
#include "get_top_n.h"
#include "demo_config.h"
//#include "labels.h"
#ifdef EIQ_GUI_PRINTF
#include "chgui.h"
#endif

static const char* labels[] = {
    "F",
    "Normal",
    "Q",
    "S",
    "V"
};

status_t MODEL_ProcessOutput(const void* data, const tensor_dims_t* dims,
                             tensor_type_t type, int inferenceTime)
{
    const float threshold = (float)DETECTION_TRESHOLD / 100;
    result_t topResults[NUM_RESULTS];
    const char* label = "No label detected";

    /* 依據 `type` 轉換 `data` */
    if (type == kTensorType_FLOAT32)
    {
        const float* floatData = (const float*)data;
        MODEL_GetTopN((const uint8_t*)floatData, dims->data[dims->size - 1], type, NUM_RESULTS, threshold, topResults);
    }
    else if (type == kTensorType_INT8)
    {
        const int8_t* intData = (const int8_t*)data;
        MODEL_GetTopN((const uint8_t*)intData, dims->data[dims->size - 1], type, NUM_RESULTS, threshold, topResults);
    }
    else
    {
        const uint8_t* uintData = (const uint8_t*)data;
        MODEL_GetTopN(uintData, dims->data[dims->size - 1], type, NUM_RESULTS, threshold, topResults);
    }

    float confidence = 0;
    if (topResults[0].index >= 0)
    {
        auto result = topResults[0];
        confidence = result.score;
        int index = result.index;
        if (confidence * 100 > DETECTION_TRESHOLD)
        {
            label = labels[index];
        }
    }

    int score = (int)(confidence * 100);
    PRINTF("----------------------------------------" EOL);
    PRINTF("     Inference time: %d us" EOL, inferenceTime);
    PRINTF("     Detected: %s (%d%%)" EOL, label, score);
    PRINTF("----------------------------------------" EOL);

#ifdef EIQ_GUI_PRINTF
    GUI_PrintfToBuffer(GUI_X_POS, GUI_Y_POS, "Detected: %.20s (%d%%)", label, score);
#endif

    return kStatus_Success;
}


//status_t MODEL_ProcessOutput(const uint8_t* data, const tensor_dims_t* dims,
//                             tensor_type_t type, int inferenceTime)
//{
//    const float threshold = (float)DETECTION_TRESHOLD / 100;
//    result_t topResults[NUM_RESULTS];
//    const char* label = "No label detected";
//
//    /* Find best label candidates. */
//    MODEL_GetTopN(data, dims->data[dims->size - 1], type, NUM_RESULTS, threshold, topResults);
//
//    float confidence = 0;
//    if (topResults[0].index >= 0)
//    {
//        auto result = topResults[0];
//        confidence = result.score;
//        int index = result.index;
//        if (confidence * 100 > DETECTION_TRESHOLD)
//        {
//            label = labels[index];
//        }
//    }
//
//    int score = (int)(confidence * 100);
//    PRINTF("----------------------------------------" EOL);
//    PRINTF("     Inference time: %d us" EOL, inferenceTime);
//    PRINTF("     Detected: %s (%d%%)" EOL, label, score);
//    PRINTF("----------------------------------------" EOL);
//
//#ifdef EIQ_GUI_PRINTF
//    GUI_PrintfToBuffer(GUI_X_POS, GUI_Y_POS, "Detected: %.20s (%d%%)", label, score);
//#endif
//
//    return kStatus_Success;
//}
