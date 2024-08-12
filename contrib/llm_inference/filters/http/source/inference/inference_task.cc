#include "source/extensions/filters/http/llm_inference/inference/inference_task.h"
// #include "source/extensions/filters/http/llm_inference/inference/inference_context.h"
#include <memory>

namespace Envoy {
namespace Extensions {
namespace HttpFilters {
namespace LLMInference {

void InferenceTask::initstat(const std::string& string_data, InferenceTaskType type) {
  task_meta_data_.type = type;
  task_meta_data_.data = string_data;
  switch (type) {
    case INFERENCETASKTYPE_CHAT_COMPLETION:
      {
        task_meta_data_.infill    = false;
        task_meta_data_.embedding = false;
      } break;
    case INFERENCETASKTYPE_EMBEDDINGS:
      {
        task_meta_data_.infill    = false;
        task_meta_data_.embedding = true;
      } break;
    case INFERENCETASKTYPE_DEFAULT:
      break;
  }
}

InferenceTask::InferenceTask(Singleton::InstanceSharedPtr owner, InferenceThread& inference_thread):
    owner_(owner), inference_thread_(inference_thread){}

InferenceTask::~InferenceTask() {}

InferenceContextPtr InferenceTask::makeInferenceContext() {
  return std::make_unique<InferenceContext>(*this, task_meta_data_);
}

void InferenceTask::addTask(std::function<void(void)> callback) {
  inference_thread_.addTask(std::move(callback));
}

// void InferenceTask::removeTask(std::shared_ptr<InferenceTaskMetaData> task_meta_data) {
//   inference_thread_.removeTask(task_meta_data);
// }

} // namespace LLMInference
} // namespace HttpFilters
} // namespace Extensions
} // namespace Envoy