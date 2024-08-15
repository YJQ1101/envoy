#pragma once

#include <functional>
#include <memory>
#include <vector>
#include "inference_thread.h"
#include "source/extensions/filters/http/common/factory_base.h"
#include "llama.h"

namespace Envoy {
namespace Extensions {
namespace HttpFilters {
namespace LLMInference {

struct server_task;
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
    NO_ERROR,
};

struct ModelInferenceResult {
  bool inference_successed = false;
  bool stopped = false;
  std::string ss;
  error_type type;
};

// using LookupHeadersCallback = std::function<void(LoadModelResult&&)>;
using LookupBodyCallback = std::function<void(ModelInferenceResult&&)>;

class InferenceContext {
public:

  InferenceContext(Envoy::Singleton::InstanceSharedPtr, InferenceThread&, const int&, const std::string&, const std::string&);
  ~InferenceContext();
  bool loadModel(const int& n_thread, const std::string& model_name);
  void modelInference(LookupBodyCallback&& cb, std::shared_ptr<InferenceTaskMetaData>&&);

private:

  bool launchSlotWithTask(server_slot &, const server_task &);
  void updateSlots();

  void processSingleTask(const server_task &);
  bool processToken(completion_token_output &, server_slot &);
  void sendPartialResponse(completion_token_output&, server_slot &);
  void sendFinalResponse(server_slot &);
  void sendEmbedding(server_slot &, const llama_batch &);
  void sendError(const int &, const std::string &, const enum error_type);

  
  // A shared_ptr to keep the cache singleton alive as long as any of its caches are in use.
  const Envoy::Singleton::InstanceSharedPtr owner_;
  InferenceThread& inference_thread_;
  std::string model_name_;
  // InferenceTask& task_;
  // InferenceTaskMetaData& task_meta_data_;
  // LookupHeadersCallback callback_headers_;
  absl::flat_hash_map<int, LookupBodyCallback> callback_body_;
  // File actions may be initiated in the file thread or the filter thread, and cancelled or
  // completed from either, therefore must be guarded by a mutex.
  // absl::Mutex mu_;

  llama_model * model_ = nullptr;
  llama_context * ctx_ = nullptr;
  llama_batch batch_;
  bool clean_kv_cache_ = true;
  bool add_bos_token_  = true;
  int32_t n_ctx_; // total context for all clients / slots
  
  std::vector<server_slot> slots_;
  std::string chat_template_ = "";
  std::string completion_id_;
  bool is_openai_;
};

using InferenceContextPtr = std::unique_ptr<InferenceContext>;
using InferenceContextSharedPtr = std::shared_ptr<InferenceContext>;

} // namespace LLMInference
} // namespace HttpFilters
} // namespace Extensions
} // namespace Envoy