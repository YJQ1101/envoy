#pragma once

#include <string>

#include "source/extensions/filters/http/common/pass_through_filter.h"
#include "contrib/envoy/extensions/filters/http/llm_inference/v3/llm_inference.pb.h"
#include "contrib/llm_inference/filters/http/source/inference/inference_task.h"

namespace Envoy {
namespace Extensions {
namespace HttpFilters {
namespace LLMInference {

using ModelPath = Protobuf::Map<std::string, std::string>;

class LLMInferenceFilterConfig : public Router::RouteSpecificFilterConfig  {
public:
  LLMInferenceFilterConfig(const envoy::extensions::filters::http::llm_inference::v3::modelParameter& proto_config);

  const int& n_thread() const {return n_thread_;}
  const ModelPath& modelPath() const {return modelPath_; }
  const int& get_id() {
    {
      // absl::MutexLock lock(&mu_);
      return ++id_task_;
    }
  }

private:
  const int n_thread_;
  const ModelPath modelPath_;
  int id_task_ = 0;
  // ABSL_GUARDED_BY(mu_) = false;
  // absl::Mutex mu_;
};

using LLMInferenceFilterConfigSharedPtr = std::shared_ptr<LLMInferenceFilterConfig>;

using ModelChosen = Protobuf::RepeatedPtrField<std::string>;

class LLMInferenceFilterConfigPerRoute : public Router::RouteSpecificFilterConfig  {
public:
  LLMInferenceFilterConfigPerRoute(const envoy::extensions::filters::http::llm_inference::v3::modelChosen& proto_config);

  const ModelChosen& modelChosen() const {return modelChosen_;};

private:
  const ModelChosen modelChosen_;
};

using LLMInferenceFilterConfigPerRouteSharedPtr = std::shared_ptr<LLMInferenceFilterConfigPerRoute>;

class LLMInferenceFilter : public Http::PassThroughDecoderFilter,
                           public std::enable_shared_from_this<LLMInferenceFilter> {
public:
  LLMInferenceFilter(LLMInferenceFilterConfigSharedPtr, InferenceContextSharedPtr);
  ~LLMInferenceFilter();

  // Http::StreamFilterBase
  void onDestroy() override;

  // Http::StreamDecoderFilter
  Http::FilterHeadersStatus decodeHeaders(Http::RequestHeaderMap&, bool) override;
  
  Http::FilterDataStatus decodeData(Buffer::Instance&, bool) override;
  
  void setDecoderFilterCallbacks(Http::StreamDecoderFilterCallbacks& callbacks) override {
    decoder_callbacks_ = &callbacks;
  }

  void getHeaders(std::shared_ptr<InferenceTaskMetaData>&&);
  void onBody(ModelInferenceResult&&);

private:
  const LLMInferenceFilterConfigSharedPtr config_;
  const InferenceContextSharedPtr ctx_;

  // InferenceTaskType task_type_ = INFERENCETASKTYPE_DEFAULT;

  Http::StreamDecoderFilterCallbacks* decoder_callbacks_;
  int id_task_;
  int n_thread() const;
  const ModelPath modelPath() const;
  int get_id() const;
};

using LLMInferenceFilterSharedPtr = std::shared_ptr<LLMInferenceFilter>;
using LLMInferenceFilterWeakPtr = std::weak_ptr<LLMInferenceFilter>;

} // namespace LLMInference
} // namespace HttpFilters
} // namespace Extensions
} // namespace Envoy