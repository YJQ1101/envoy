#include "contrib/llm_inference/filters/http/source/inference/inference_task.h"
#include "inference_context.h"
#include <memory>

namespace Envoy {
namespace Extensions {
namespace HttpFilters {
namespace LLMInference {

InferenceTaskMetaData::InferenceTaskMetaData(const std::string& data,bool infill,bool embe, int id, InferenceTaskType type, int id_target):
  data(data), type(type),infill(infill), embedding(embe),id(id), id_target(id_target) {}

} // namespace LLMInference
} // namespace HttpFilters
} // namespace Extensions
} // namespace Envoy