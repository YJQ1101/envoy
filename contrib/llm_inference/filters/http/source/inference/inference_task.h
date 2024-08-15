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
  InferencetasktypeTypeCompletion,
  InferencetasktypeTypeEmbeedings,
  InferencetasktypeTypeCancel,
};

struct InferenceTaskMetaData {
  InferenceTaskMetaData(const std::string&,bool,bool,int, InferenceTaskType,int);
  std::string data;
  InferenceTaskType type;
  bool infill    = false;
  bool embedding = false;
  int id        = -1; 
  int id_target = -1;
};

} // namespace LLMInference
} // namespace HttpFilters
} // namespace Extensions
} // namespace Envoy