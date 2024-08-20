#include "contrib/llm_inference/filters/http/source/llm_inference_filter.h"

#include "inference/inference_context.h"
#include "source/common/buffer/buffer_impl.h"

#include "envoy/server/filter_config.h"

#include "source/common/http/utility.h"
#include "source/common/protobuf/utility.h"
#include "source/common/http/headers.h"
#include "source/common/http/header_map_impl.h"
#include <memory>

namespace Envoy {
namespace Extensions {
namespace HttpFilters {
namespace LLMInference {

LLMInferenceFilterConfig::LLMInferenceFilterConfig(
    const envoy::extensions::filters::http::llm_inference::v3::modelParameter& proto_config)
    : modelParameter_{proto_config.n_threads(), proto_config.n_parallel(), proto_config.embedding()},
      modelPath_(proto_config.modelpath()) {}

LLMInferenceFilterConfigPerRoute::LLMInferenceFilterConfigPerRoute(
    const envoy::extensions::filters::http::llm_inference::v3::modelChosen& proto_config)
    : modelChosen_(proto_config.usemodel()) {}

LLMInferenceFilter::LLMInferenceFilter(LLMInferenceFilterConfigSharedPtr config, InferenceContextSharedPtr ctx)
    : config_(config), ctx_(ctx) {}

LLMInferenceFilter::~LLMInferenceFilter() {}

void LLMInferenceFilter::onDestroy() {
  if (id_task_ != -1) {
    ctx_->modelInference([](ModelInferenceResult&&) {
    }, std::make_shared<InferenceTaskMetaData>("{}", false, ctx_->getId(), InferencetasktypeTypeCancel, id_task_));
  }
}

const ModelParameter LLMInferenceFilter::modelParameter() const {
  return config_->modelParameter();
}

const ModelPath LLMInferenceFilter::modelPath() const {
  return config_->modelPath();
}

Http::FilterHeadersStatus LLMInferenceFilter::decodeHeaders(Http::RequestHeaderMap& headers, bool end_stream) {
  if (end_stream) {
    // If this is a header-only request, we don't need to do any inference.
    return Http::FilterHeadersStatus::Continue;
  }

  // // Route-level configuration overrides filter-level configuration.
  const auto* per_route_inference_settings =
      Http::Utility::resolveMostSpecificPerFilterConfig<LLMInferenceFilterConfigPerRoute>(
          "envoy.filters.http.llm_inference", decoder_callbacks_->route());
  if (!per_route_inference_settings) {
    return Http::FilterHeadersStatus::Continue;
  }
  
  const absl::string_view headersPath = headers.getPathValue();
  if (headersPath == "/v1/chat/completions") {
    task_type_ = InferencetasktypeTypeCompletion;
  } else if (headersPath == "/v1/embeddings") {
    task_type_ = InferencetasktypeTypeEmbeedings;
  } 
  return Http::FilterHeadersStatus::StopIteration;
}

Http::FilterDataStatus LLMInferenceFilter::decodeData(Buffer::Instance& data, bool end_stream) {
  if (!end_stream) {
    id_task_ = ctx_->getId();
    getHeaders(std::make_shared<InferenceTaskMetaData>(data.toString(), false, id_task_, task_type_, -1));
  }
  return Http::FilterDataStatus::StopIterationNoBuffer;
}

void LLMInferenceFilter::getHeaders(std::shared_ptr<InferenceTaskMetaData>&& task_meta_data) {
  LLMInferenceFilterWeakPtr self = weak_from_this();

  ctx_->modelInference([self, &dispatcher = decoder_callbacks_->dispatcher()](ModelInferenceResult&& body) {
    dispatcher.post(
      [self, body = std::move(body)]() mutable {
        if (LLMInferenceFilterSharedPtr llm_inference_filter = self.lock()) {
          llm_inference_filter->onBody(std::move(body));
        }
      }
    );
  }, std::move(task_meta_data));
}

void LLMInferenceFilter::onBody(ModelInferenceResult&& body) {
  if (!body.inference_successed) {
    switch (body.type) {
      case ERROR_TYPE_INVALID_REQUEST:
        decoder_callbacks_->sendLocalReply(Http::Code::BadRequest, body.ss, nullptr, absl::nullopt, "");
        break;
      case ERROR_TYPE_AUTHENTICATION:
        decoder_callbacks_->sendLocalReply(Http::Code::Unauthorized, body.ss, nullptr, absl::nullopt, "");
        break;
      case ERROR_TYPE_SERVER:
        decoder_callbacks_->sendLocalReply(Http::Code::InternalServerError, body.ss, nullptr, absl::nullopt, "");
        break;
      case ERROR_TYPE_NOT_FOUND:
        decoder_callbacks_->sendLocalReply(Http::Code::NotFound, body.ss, nullptr, absl::nullopt, "");
        break;
      case ERROR_TYPE_PERMISSION:
        decoder_callbacks_->sendLocalReply(Http::Code::Forbidden, body.ss, nullptr, absl::nullopt, "");
        break;
      case ERROR_TYPE_UNAVAILABLE:
        decoder_callbacks_->sendLocalReply(Http::Code::ServiceUnavailable, body.ss, nullptr, absl::nullopt, "");
        break;
      case ERROR_TYPE_NOT_SUPPORTED:
        decoder_callbacks_->sendLocalReply(Http::Code::NotImplemented, body.ss, nullptr, absl::nullopt, "");
        break;
      case NO_ERROR:
        break;
    }
  } else {
    if (!header_) {
      Http::ResponseHeaderMapPtr headers{Http::createHeaderMap<Http::ResponseHeaderMapImpl>({{Http::Headers::get().Status, "200"}})};
      decoder_callbacks_->encodeHeaders(std::move(headers), false, "good");
      header_ = true;
    }

    Buffer::InstancePtr request_data = std::make_unique<Buffer::OwnedImpl>(body.ss);

    if (body.stopped) {
      decoder_callbacks_->encodeData(*request_data, true);
    } else {
      decoder_callbacks_->encodeData(*request_data, false);
    }
  }
}
//   //测多次，查看内存情况 
//   //推理时间超出预期，首字节时间，整体时间

} // namespace LLMInference
} // namespace HttpFilters
} // namespace Extensions
} // namespace Envoy
