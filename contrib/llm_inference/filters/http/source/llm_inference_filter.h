#pragma once

#include <string>

#include "source/extensions/filters/http/common/pass_through_filter.h"
#include "envoy/extensions/filters/http/llm_inference/llm_inference.pb.h"
#include "source/extensions/filters/http/llm_inference/inference/inference_task.h"

namespace Envoy {
namespace Extensions {
namespace HttpFilters {
namespace LLMInference {

class LLMInferenceFilterConfig : public Router::RouteSpecificFilterConfig  {
public:
  LLMInferenceFilterConfig(const envoy::extensions::filters::http::llm_inference::LLMInference& proto_config);

  const std::string& key() const { return key_; }
  const std::string& val() const { return val_; }

private:
  const std::string key_;
  const std::string val_;
};

using LLMInferenceFilterConfigSharedPtr = std::shared_ptr<LLMInferenceFilterConfig>;

class LLMInferenceFilter : public Http::PassThroughDecoderFilter,
                           public std::enable_shared_from_this<LLMInferenceFilter> {
public:
  LLMInferenceFilter(LLMInferenceFilterConfigSharedPtr, LLMInferenceTaskSharedPtr);
  ~LLMInferenceFilter();

  // Http::StreamFilterBase
  void onDestroy() override;

  // Http::StreamDecoderFilter
  Http::FilterHeadersStatus decodeHeaders(Http::RequestHeaderMap&, bool) override;
  
  Http::FilterDataStatus decodeData(Buffer::Instance&, bool) override;
  
  void setDecoderFilterCallbacks(Http::StreamDecoderFilterCallbacks& callbacks) override {
    decoder_callbacks_ = &callbacks;
  }

  void getHeaders();
  void onHeaders(LoadModelResult&& result);
  void getBody();
  void onBody(ModelInferenceResult&&);

private:
  const LLMInferenceFilterConfigSharedPtr config_;
  const LLMInferenceTaskSharedPtr task_;

  InferenceContextPtr ctx_;
  InferenceTaskType task_type_ = INFERENCETASKTYPE_DEFAULT;

  Http::StreamDecoderFilterCallbacks* decoder_callbacks_;

  const std::string headerKey() const;
  const std::string headerValue() const;
};

using LLMInferenceFilterSharedPtr = std::shared_ptr<LLMInferenceFilter>;
using LLMInferenceFilterWeakPtr = std::weak_ptr<LLMInferenceFilter>;

} // namespace LLMInference
} // namespace HttpFilters
} // namespace Extensions
} // namespace Envoy