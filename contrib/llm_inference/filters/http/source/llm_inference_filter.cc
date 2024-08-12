#include "source/extensions/filters/http/llm_inference/llm_inference_filter.h"
#include "source/common/buffer/buffer_impl.h"

#include "envoy/server/filter_config.h"

#include "source/common/http/utility.h"
#include "source/common/protobuf/utility.h"
#include "source/common/http/headers.h"
#include "source/common/http/header_map_impl.h"

namespace Envoy {
namespace Extensions {
namespace HttpFilters {
namespace LLMInference {

LLMInferenceFilterConfig::LLMInferenceFilterConfig(
    const envoy::extensions::filters::http::llm_inference::LLMInference& proto_config)
    : key_(proto_config.key()), val_(proto_config.val()) {}

LLMInferenceFilter::LLMInferenceFilter(LLMInferenceFilterConfigSharedPtr config, LLMInferenceTaskSharedPtr task)
    : config_(config), task_(task) {}

LLMInferenceFilter::~LLMInferenceFilter() {}

void LLMInferenceFilter::onDestroy() {}

const std::string LLMInferenceFilter::headerKey() const {
  return config_->key();
}

const std::string LLMInferenceFilter::headerValue() const {
  return config_->val();
}

Http::FilterHeadersStatus LLMInferenceFilter::decodeHeaders(Http::RequestHeaderMap& headers, bool end_stream) {
  // std::cout << headers.getPathValue() << std::endl;
  if (end_stream) {
    // If this is a header-only request, we don't need to do any inference.
    return Http::FilterHeadersStatus::Continue;
  }

  // Route-level configuration overrides filter-level configuration.
  // const auto* per_route_inference_settings =
  //     Http::Utility::resolveMostSpecificPerFilterConfig<LLMInferenceFilterConfig>(
  //         "envoy.filters.http.llm_inference", decoder_callbacks_->route());
  // if (!per_route_inference_settings) {
  //   return Http::FilterHeadersStatus::Continue;
  // }
  
  const absl::string_view headersPath = headers.getPathValue();
  if (headersPath == "/v1/chat/completions") {
    task_type_ = INFERENCETASKTYPE_CHAT_COMPLETION;
  } else if (headersPath == "/v1/embeddings") {
    task_type_ = INFERENCETASKTYPE_EMBEDDINGS;
  } 

  if (task_type_ == INFERENCETASKTYPE_DEFAULT) {
    // If this is not match, we don't need to do any inference.
    return Http::FilterHeadersStatus::Continue;
  }
  return Http::FilterHeadersStatus::StopIteration;
}

Http::FilterDataStatus LLMInferenceFilter::decodeData(Buffer::Instance& data, bool end_stream) {
  if (!end_stream) {
    task_->initstat(data.toString(), task_type_);
    ctx_ = task_->makeInferenceContext();
    getHeaders();
  }
  return Http::FilterDataStatus::StopIterationAndWatermark;
}

void LLMInferenceFilter::getHeaders() {
  LLMInferenceFilterWeakPtr self = weak_from_this();

  ctx_->loadSingleTask([self, &dispatcher = decoder_callbacks_->dispatcher()](LoadModelResult&&result) {
    dispatcher.post([self, result = std::move(result)]() mutable {
        if (LLMInferenceFilterSharedPtr llm_inference_filter = self.lock()) {
          llm_inference_filter->onHeaders(std::move(result));
        }
      }
    );
  });
}

void LLMInferenceFilter::onHeaders(LoadModelResult&& result) {
  if (!result.load_successed) {
    switch (result.error_type) {
      case ERROR_TYPE_INVALID_REQUEST:
        decoder_callbacks_->sendLocalReply(Http::Code::BadRequest, result.error_reason, nullptr, absl::nullopt, "");
        break;
      case ERROR_TYPE_AUTHENTICATION:
        decoder_callbacks_->sendLocalReply(Http::Code::Unauthorized, result.error_reason, nullptr, absl::nullopt, "");
        break;
      case ERROR_TYPE_SERVER:
        decoder_callbacks_->sendLocalReply(Http::Code::InternalServerError, result.error_reason, nullptr, absl::nullopt, "");
        break;
      case ERROR_TYPE_NOT_FOUND:
        decoder_callbacks_->sendLocalReply(Http::Code::NotFound, result.error_reason, nullptr, absl::nullopt, "");
        break;
      case ERROR_TYPE_PERMISSION:
        decoder_callbacks_->sendLocalReply(Http::Code::Forbidden, result.error_reason, nullptr, absl::nullopt, "");
        break;
      case ERROR_TYPE_UNAVAILABLE:
        decoder_callbacks_->sendLocalReply(Http::Code::ServiceUnavailable, result.error_reason, nullptr, absl::nullopt, "");
        break;
      case ERROR_TYPE_NOT_SUPPORTED:
        decoder_callbacks_->sendLocalReply(Http::Code::NotImplemented, result.error_reason, nullptr, absl::nullopt, "");
        break;
      case ERROR_TYPE_NO_ERROR:
        break;
    }
  } else {
    Http::ResponseHeaderMapPtr headers{
      Http::createHeaderMap<Http::ResponseHeaderMapImpl>({{Http::Headers::get().Status, "200"}})};
    decoder_callbacks_->encodeHeaders(std::move(headers), false, "good");
    getBody();
  }
}

void LLMInferenceFilter::getBody() {
  LLMInferenceFilterWeakPtr self = weak_from_this();
  ctx_->modelInference([self, &dispatcher = decoder_callbacks_->dispatcher()](ModelInferenceResult&& body) {
    dispatcher.post(
      [self, body = std::move(body)]() mutable {
        if (LLMInferenceFilterSharedPtr llm_inference_filter = self.lock()) {
          llm_inference_filter->onBody(std::move(body));
        }
      }
    );
  });
}

void LLMInferenceFilter::onBody(ModelInferenceResult&& body) {
  Buffer::InstancePtr request_data = std::make_unique<Buffer::OwnedImpl>(body.ss);
  if (body.stopped) {
    decoder_callbacks_->encodeData(*request_data, true);
  } else {
    decoder_callbacks_->encodeData(*request_data, false);
  }
  //多平台
  //卸载模型，查看内存情况
  //测多次，查看内存情况
  //资源复用


  //推理时间超出预期，首字节时间，整体时间
  //请求中止，推理中止
}

} // namespace LLMInference
} // namespace HttpFilters
} // namespace Extensions
} // namespace Envoy
