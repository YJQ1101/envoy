#pragma once

#include <functional>
#include <memory>
#include "llama.h"

namespace Envoy {
namespace Extensions {
namespace HttpFilters {
namespace LLMInference {

class InferenceTask;
struct InferenceTaskMetaData;
struct server_slot;
struct completion_token_output;

// https://community.openai.com/t/openai-chat-list-of-error-codes-and-types/357791/11
enum error_type {
    ERROR_TYPE_INVALID_REQUEST,
    ERROR_TYPE_AUTHENTICATION,
    ERROR_TYPE_SERVER,
    ERROR_TYPE_NOT_FOUND,
    ERROR_TYPE_PERMISSION,
    ERROR_TYPE_UNAVAILABLE, // custom error
    ERROR_TYPE_NOT_SUPPORTED, // custom error
    ERROR_TYPE_NO_ERROR, // custom error
};

struct LoadModelResult {
  bool load_successed = false;
  error_type error_type;
  std::string error_reason;
};

struct ModelInferenceResult {
  bool inference_successed = false;
  bool stopped;
  error_type error_type;
  std::string ss;
};

using LookupHeadersCallback = std::function<void(LoadModelResult&&)>;
using LookupBodyCallback = std::function<void(ModelInferenceResult&&)>;

class InferenceContext {
public:
  InferenceContext(InferenceTask& task, InferenceTaskMetaData& task_meta_data)
      : task_(task), task_meta_data_(task_meta_data) {}

  void loadSingleTask(LookupHeadersCallback&& cb);
  void modelInference(LookupBodyCallback&& cb);

private:

  bool launchSlotWithTask();
  bool updateSlots();
  void processSingleTask();
  bool processToken(completion_token_output&);
  void sendPartialResponse(completion_token_output&);
  void sendFinalResponse();

  bool loadModel(std::string);
  void generate();


  InferenceTask& task_;
  InferenceTaskMetaData& task_meta_data_;
  LookupHeadersCallback callback_headers_;
  LookupBodyCallback callback_body_;
  // File actions may be initiated in the file thread or the filter thread, and cancelled or
  // completed from either, therefore must be guarded by a mutex.
  // absl::Mutex mu_;
  std::string completion_id_;
  llama_model * model = nullptr;
  llama_context * ctx = nullptr;
  llama_batch batch;
  int32_t n_ctx;
  std::shared_ptr<server_slot> slot;

};

using InferenceContextPtr = std::unique_ptr<InferenceContext>;
} // namespace LLMInference
} // namespace HttpFilters
} // namespace Extensions
} // namespace Envoy