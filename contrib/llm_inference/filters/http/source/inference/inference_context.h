#pragma once

#include "contrib/llm_inference/filters/http/source/inference/inference_thread.h"
#include "contrib/llm_inference/filters/http/source/inference/inference_task.h"
#include "source/extensions/filters/http/common/factory_base.h"

#include <functional>
#include <memory>
#include <vector>
#include "llama.h"

namespace Envoy {
namespace Extensions {
namespace HttpFilters {
namespace LLMInference {

struct server_task;
struct server_slot;
struct completion_token_output;

struct server_metrics {
    int64_t t_start = 0;

    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t t_prompt_processing_total       = 0;
    uint64_t n_tokens_predicted_total        = 0;
    uint64_t t_tokens_generation_total       = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing       = 0;

    uint64_t n_tokens_predicted  = 0;
    uint64_t t_tokens_generation = 0;

    void init() {
        t_start = ggml_time_us();
    }

    void on_prompt_eval(const server_slot & slot);
    void on_prediction(const server_slot & slot);
    void reset_bucket();
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

  InferenceContext(Envoy::Singleton::InstanceSharedPtr, InferenceThread&, const ModelParameter&, const std::string&, const std::string&);
  ~InferenceContext();
  bool loadModel(const ModelParameter& model_parameter, const std::string& model_name);
  void modelInference(LookupBodyCallback&& cb, std::shared_ptr<InferenceTaskMetaData>&&);
  int getId();

  llama_model * model = nullptr;
  llama_context * ctx = nullptr;
  llama_batch batch;
  bool clean_kv_cache = true;
  bool add_bos_token  = true;
  int32_t n_ctx; // total context for all clients / slots
  // system prompt
  bool system_need_update_ = false;
  std::string              system_prompt;
  std::vector<llama_token> system_tokens;

  // slots / clients
  std::vector<server_slot> slots;
  server_metrics metrics;
  std::string chat_template_ = "";
  std::string completion_id_;
  bool is_openai_;


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
  absl::flat_hash_map<int, LookupBodyCallback> callback_body_;
  std::string model_name_;
};

using InferenceContextSharedPtr = std::shared_ptr<InferenceContext>;

} // namespace LLMInference
} // namespace HttpFilters
} // namespace Extensions
} // namespace Envoy