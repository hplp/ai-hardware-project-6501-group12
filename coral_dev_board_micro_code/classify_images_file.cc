// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This example demonstrates how to classify an skin image using a pre-trained tflite model.
// Due to the limitated image qualiti of on-device camera, the image is feed to the network through a file.

#include <vector>

#include "libs/base/filesystem.h"
#include "libs/base/led.h"
#include "libs/tensorflow/classification.h"
#include "libs/tensorflow/utils.h"
#include "libs/tpu/edgetpu_manager.h"
#include "libs/tpu/edgetpu_op.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"

// Performs image classification with MobileNet, running on the Edge TPU,
// using a local bitmap image as input.
// The top 3 class predictions are printed to the serial console.
//
// To build and flash from coralmicro root:
//    bash build.sh
//    python3 scripts/flashtool.py -e classify_images_file

// [start-sphinx-snippet:classify-image]
namespace coralmicro {
namespace {
constexpr char kModelPath[] =
    "/models/skin_image_classification_ResNet18.tflite";
constexpr char kImagePath[] = "/skin_image.rgb";
constexpr int kTensorArenaSize = 1024 * 1024;
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);

void Main() {
  printf("Classify Image Example!\r\n");
  // Turn on Status LED to show the board is on.
  LedSet(Led::kStatus, true);

  std::vector<uint8_t> model;
  if (!LfsReadFile(kModelPath, &model)) {
    printf("ERROR: Failed to load %s\r\n", kModelPath);
    return;
  }

  // [start-sphinx-snippet:edgetpu]
  auto tpu_context = EdgeTpuManager::GetSingleton()->OpenDevice();
  if (!tpu_context) {
    printf("ERROR: Failed to get EdgeTpu context\r\n");
    return;
  }

  tflite::MicroErrorReporter error_reporter;
  tflite::MicroMutableOpResolver<1> resolver;
  resolver.AddCustom(kCustomOp, RegisterCustomOp());

  tflite::MicroInterpreter interpreter(tflite::GetModel(model.data()), resolver,
                                       tensor_arena, kTensorArenaSize,
                                       &error_reporter);
  // [end-sphinx-snippet:edgetpu]
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("ERROR: AllocateTensors() failed\r\n");
    return;
  }

  if (interpreter.inputs().size() != 1) {
    printf("ERROR: Model must have only one input tensor\r\n");
    return;
  }

  auto* input_tensor = interpreter.input_tensor(0);
  if (!LfsReadFile(kImagePath, tflite::GetTensorData<uint8_t>(input_tensor),
                   input_tensor->bytes)) {
    printf("ERROR: Failed to load %s\r\n", kImagePath);
    return;
  }

  if (interpreter.Invoke() != kTfLiteOk) {
    printf("ERROR: Invoke() failed\r\n");
    return;
  }

  auto results = tensorflow::GetClassificationResults(&interpreter, 0.0f, 3);
  auto result = results[0] > 0.5; // note: since we can only show the result through the LED, only 1 image file are feed to the network
  if (result) {
    // Flashing the LED to indicate the malicious image
    LedSet(Led::kUser, true);
    LedSet(Led::kUser, false);
    LedSet(Led::kUser, true);
    LedSet(Led::kUser, false);
  } else {
    // Does not flashing the LED to indicate the benign image
    LedSet(Led::kUser, false);
  }
}
}  // namespace
}  // namespace coralmicro

extern "C" void app_main(void* param) {
  (void)param;
  coralmicro::Main();
  vTaskSuspend(nullptr);
}
// [end-sphinx-snippet:classify-image]
