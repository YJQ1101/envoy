#pragma once

#include "contrib/llm_inference/filters/http/source/inference/inference_thread.h"
#include "contrib/llm_inference/filters/http/source/inference/inference_context.h"
#include "source/extensions/filters/http/common/factory_base.h"

namespace Envoy {
namespace Extensions {
namespace HttpFilters {
namespace LLMInference {

// class InferenceContext;
enum InferenceTaskType {
  INFERENCETASKTYPE_CHAT_COMPLETION,
  INFERENCETASKTYPE_EMBEDDINGS,
  INFERENCETASKTYPE_DEFAULT,
};

struct InferenceTaskMetaData {
  // InferenceTaskMetaData(const std::string&, InferenceTaskType);
  std::string data;
  InferenceTaskType type;
  bool infill    = false;
  bool embedding = false;
};

class InferenceTask {
public:
  InferenceTask(Envoy::Singleton::InstanceSharedPtr, InferenceThread&);
  ~InferenceTask();
  InferenceContextPtr makeInferenceContext();
  void addTask(std::function<void(void)>);
  void initstat(const std::string&, InferenceTaskType);
  // void removeTask(std::shared_ptr<InferenceTaskMetaData>);
  InferenceTaskMetaData task_meta_data_;

private:
  // A shared_ptr to keep the cache singleton alive as long as any of its caches are in use.
  const Envoy::Singleton::InstanceSharedPtr owner_;
  InferenceThread& inference_thread_;
};

using LLMInferenceTaskSharedPtr = std::shared_ptr<InferenceTask>;

} // namespace LLMInference
} // namespace HttpFilters
} // namespace Extensions
} // namespace Envoy