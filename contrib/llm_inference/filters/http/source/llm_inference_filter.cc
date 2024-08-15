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
    : n_thread_(proto_config.n_thread()), modelPath_(proto_config.modelpath()) {}

LLMInferenceFilterConfigPerRoute::LLMInferenceFilterConfigPerRoute(
    const envoy::extensions::filters::http::llm_inference::v3::modelChosen& proto_config)
    : modelChosen_(proto_config.usemodel()) {}

LLMInferenceFilter::LLMInferenceFilter(LLMInferenceFilterConfigSharedPtr config, InferenceContextSharedPtr ctx)
    : config_(config), ctx_(ctx) {}

LLMInferenceFilter::~LLMInferenceFilter() {}

void LLMInferenceFilter::onDestroy() {
  ctx_->modelInference([](ModelInferenceResult&&) {
  }, std::make_shared<InferenceTaskMetaData>("{}", false, false, get_id(), InferencetasktypeTypeCancel, id_task_));
}

int LLMInferenceFilter::n_thread() const {
  return config_->n_thread();
}

const ModelPath LLMInferenceFilter::modelPath() const {
  return config_->modelPath();
}

int LLMInferenceFilter::get_id() const {
  return config_->get_id();
}

Http::FilterHeadersStatus LLMInferenceFilter::decodeHeaders(Http::RequestHeaderMap&, bool end_stream) {
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
  return Http::FilterHeadersStatus::StopIteration;
}

Http::FilterDataStatus LLMInferenceFilter::decodeData(Buffer::Instance& data, bool end_stream) {
  if (!end_stream) {
    id_task_ = get_id();
    getHeaders(std::make_shared<InferenceTaskMetaData>(data.toString(), false, false, id_task_, InferencetasktypeTypeCompletion, -1));
  }
  return Http::FilterDataStatus::StopIterationAndWatermark;
}

void LLMInferenceFilter::getHeaders(std::shared_ptr<InferenceTaskMetaData>&& task_meta_data) {

  std::cout << "hhhh" << task_meta_data->id << std::endl;
  LLMInferenceFilterWeakPtr self = weak_from_this();

  Http::ResponseHeaderMapPtr headers{Http::createHeaderMap<Http::ResponseHeaderMapImpl>({{Http::Headers::get().Status, "200"}})};
  decoder_callbacks_->encodeHeaders(std::move(headers), false, "good");
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
    Buffer::InstancePtr request_data = std::make_unique<Buffer::OwnedImpl>(body.ss);

    if (body.stopped) {
      decoder_callbacks_->encodeData(*request_data, true);
    } else {
      decoder_callbacks_->encodeData(*request_data, false);
    }
  }
}
//   //多平台
//   //卸载模型，查看内存情况
//   //测多次，查看内存情况
//   //资源复用


//   //推理时间超出预期，首字节时间，整体时间
//   //请求中止，推理中止
// }

} // namespace LLMInference
} // namespace HttpFilters
} // namespace Extensions
} // namespace Envoy
