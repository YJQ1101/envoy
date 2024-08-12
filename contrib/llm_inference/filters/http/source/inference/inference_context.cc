#include "common/sampling.cpp"
#include "common/common.cpp"
#include "common/json-schema-to-grammar.cpp"
#include "common/grammar-parser.cpp"
#include "utils.hpp"
#include "contrib/llm_inference/filters/http/source/inference/inference_context.h"
#include "contrib/llm_inference/filters/http/source/inference/inference_task.h"
#include <cstdio>
#include <memory>

char const *LLAMA_COMMIT = "123";
char const *LLAMA_COMPILER = "123";
char const *LLAMA_BUILD_TARGET = "123";
int LLAMA_BUILD_NUMBER = 1;
gpt_params params;

namespace Envoy {
namespace Extensions {
namespace HttpFilters {
namespace LLMInference {

struct server_slot {
    struct slot_params params;
    json task_data;
    // generation props
    int32_t n_ctx       = 0;  // context size per slot
    int32_t n_decoded   = 0;
    // int32_t n_remaining = -1;
    int32_t i_batch     = -1;
    int32_t n_predict   = -1; // TODO: disambiguate from params.n_predict

    int32_t n_prompt_tokens           = 0;
    int32_t n_prompt_tokens_processed = 0;

    json prompt;
    json data;

    // when a task is submitted, we first tokenize the prompt and store it here
    std::vector<llama_token> prompt_tokens;

    std::string generated_text;
    std::vector<llama_token> cache_tokens;
    std::vector<completion_token_output> generated_token_probs;

    bool infill         = false;
    bool embedding      = false;
    bool has_next_token = true;
    bool truncated      = false;
    bool stopped_eos    = false;
    bool stopped_word   = false;
    bool stopped_limit  = false;

    bool oaicompat = false;

    std::string oaicompat_model;
    std::string stopping_word;

    // sampling
    llama_token sampled;
    struct llama_sampling_params sparams;
    llama_sampling_context * ctx_sampling = nullptr;
    json json_schema;

    // stats
    size_t n_sent_text = 0; // number of sent text character
    size_t n_sent_token_probs = 0;

    void add_token_string(const completion_token_output & token) {
        generated_token_probs.push_back(token);
    }
};

bool InferenceContext::launchSlotWithTask() {
  n_ctx = llama_n_ctx(ctx);

  slot->n_ctx = n_ctx;
  const int32_t n_batch = llama_n_batch(ctx);
  batch = llama_batch_init(n_batch, 0, 1);

  slot_params default_params;
  llama_sampling_params default_sparams;
  auto & data = slot->data;

  if (data.count("__oaicompat") != 0) {
    slot->oaicompat = true;
    slot->oaicompat_model = json_value(data, "model", std::string(DEFAULT_OAICOMPAT_MODEL));
  } else {
    slot->oaicompat = false;
    slot->oaicompat_model = "";
  }

  slot->params.stream             = json_value(data, "stream",            false);
  slot->params.cache_prompt       = json_value(data, "cache_prompt",      false);
  slot->params.n_predict          = json_value(data, "n_predict",         default_params.n_predict);
  slot->sparams.top_k             = json_value(data, "top_k",             default_sparams.top_k);
  slot->sparams.top_p             = json_value(data, "top_p",             default_sparams.top_p);
  slot->sparams.min_p             = json_value(data, "min_p",             default_sparams.min_p);
  slot->sparams.tfs_z             = json_value(data, "tfs_z",             default_sparams.tfs_z);
  slot->sparams.typical_p         = json_value(data, "typical_p",         default_sparams.typical_p);
  slot->sparams.temp              = json_value(data, "temperature",       default_sparams.temp);
  slot->sparams.dynatemp_range    = json_value(data, "dynatemp_range",    default_sparams.dynatemp_range);
  slot->sparams.dynatemp_exponent = json_value(data, "dynatemp_exponent", default_sparams.dynatemp_exponent);
  slot->sparams.penalty_last_n    = json_value(data, "repeat_last_n",     default_sparams.penalty_last_n);
  slot->sparams.penalty_repeat    = json_value(data, "repeat_penalty",    default_sparams.penalty_repeat);
  slot->sparams.penalty_freq      = json_value(data, "frequency_penalty", default_sparams.penalty_freq);
  slot->sparams.penalty_present   = json_value(data, "presence_penalty",  default_sparams.penalty_present);
  slot->sparams.mirostat          = json_value(data, "mirostat",          default_sparams.mirostat);
  slot->sparams.mirostat_tau      = json_value(data, "mirostat_tau",      default_sparams.mirostat_tau);
  slot->sparams.mirostat_eta      = json_value(data, "mirostat_eta",      default_sparams.mirostat_eta);
  slot->sparams.penalize_nl       = json_value(data, "penalize_nl",       default_sparams.penalize_nl);
  slot->params.n_keep             = json_value(data, "n_keep",            slot->params.n_keep);
  slot->params.n_discard          = json_value(data, "n_discard",         default_params.n_discard);
  slot->sparams.seed              = json_value(data, "seed",              default_sparams.seed);
  slot->sparams.n_probs           = json_value(data, "n_probs",           default_sparams.n_probs);
  slot->sparams.min_keep          = json_value(data, "min_keep",          default_sparams.min_keep);

  if (slot->n_predict > 0 && slot->params.n_predict > slot->n_predict) {
    slot->params.n_predict = slot->n_predict;
  }

  // get prompt
  {
    const auto & prompt = data.find("prompt");
    if (prompt == data.end()) {
      return false;
    } else {
      slot->prompt = *prompt;
    }
    if (slot->prompt.is_array() && slot->prompt.size() == 0) {
      return false;
    }
  }

  {
    slot->sparams.logit_bias.clear();

    if (json_value(data, "ignore_eos", false)) {
      slot->sparams.logit_bias[llama_token_eos(model)] = -INFINITY;
    }

    const auto & logit_bias = data.find("logit_bias");
    if (logit_bias != data.end() && logit_bias->is_array()) {
      const int n_vocab = llama_n_vocab(model);
      for (const auto & el : *logit_bias) {
        // TODO: we may want to throw errors here, in case "el" is incorrect
        if (el.is_array() && el.size() == 2) {
          float bias;
          if (el[1].is_number()) {
            bias = el[1].get<float>();
          } else if (el[1].is_boolean() && !el[1].get<bool>()) {
            bias = -INFINITY;
          } else {
            continue;
          }

          if (el[0].is_number_integer()) {
            llama_token tok = el[0].get<llama_token>();
            if (tok >= 0 && tok < n_vocab) {
              slot->sparams.logit_bias[tok] = bias;
            }
          } else if (el[0].is_string()) {
            auto toks = llama_tokenize(model, el[0].get<std::string>(), false);
            for (auto tok : toks) {
              slot->sparams.logit_bias[tok] = bias;
            }
          }
        }
      }
    }
  }

  {
    slot->params.antiprompt.clear();

    const auto & stop = data.find("stop");
    if (stop != data.end() && stop->is_array()) {
      for (const auto & word : *stop) {
        if (!word.empty()) {
          slot->params.antiprompt.push_back(word);
        }
      }
    }
  }

  {
    const auto & samplers_sequence = data.find("samplers");
    if (samplers_sequence != data.end() && samplers_sequence->is_array()) {
      std::vector<std::string> sampler_names;
      for (const auto & sampler_name : *samplers_sequence) {
        if (sampler_name.is_string()) {
          sampler_names.emplace_back(sampler_name);
        }
      }
      slot->sparams.samplers_sequence = sampler_types_from_names(sampler_names, false);
    } else {
      slot->sparams.samplers_sequence = default_sparams.samplers_sequence;
    }
  }

  {
    if (slot->ctx_sampling != nullptr) {
      llama_sampling_free(slot->ctx_sampling);
    }
    slot->ctx_sampling = llama_sampling_init(slot->sparams);
    if (slot->ctx_sampling == nullptr) {
      // for now, the only error that may happen here is invalid grammar
      return false;
    }
  }

  slot->prompt_tokens.clear();

  return true;
}

bool InferenceContext::loadModel(std::string model_name) {
  if (model_name == "qwen2") {
      params.model = "/home/yuanjq/model/qwen2-7b-instruct-q5_k_m.gguf";
  } else if (model_name == "llama3") {
      params.model = "/home/yuanjq/model/Meta-Llama-3-8B-Instruct-fp16.gguf";
  } else if (model_name == "bge") {
      params.model = "/home/yuanjq/model/bge-small-zh-v1.5-f32.gguf";
  }
  llama_backend_init();
  llama_numa_init(params.numa);

  std::tie(model, ctx) = llama_init_from_gpt_params(params);
  if (model == nullptr) {
    return false;
  }
  return true;
}

void InferenceContext::processSingleTask() {
  json data;
  try {
    data = json::parse(task_meta_data_.data);
  } catch (const std::exception & e) {
    callback_headers_(LoadModelResult{false, ERROR_TYPE_INVALID_REQUEST, "request data doesn't meet specifications"});
    return;
  }

  if (!loadModel(json_value(data, "model", std::string("unknown")))) {
    callback_headers_(LoadModelResult{false, ERROR_TYPE_UNAVAILABLE, "can't load model, please check your model"});
    return;
  }

  switch (task_meta_data_.type) {
    case INFERENCETASKTYPE_CHAT_COMPLETION:
      {
        data = oaicompat_completion_params_parse(model, data, "chatml");
      } break;
    case INFERENCETASKTYPE_EMBEDDINGS:
      {
        // handleEmbeddings();
      } break;
    case INFERENCETASKTYPE_DEFAULT:
      break;
  }

  slot = std::make_shared<server_slot>();
  slot->data = std::move(data);
  if (!launchSlotWithTask()) {
    callback_headers_(LoadModelResult{false, ERROR_TYPE_UNAVAILABLE, "can't load inference task"});
    return;
  }
  callback_headers_(LoadModelResult{true, ERROR_TYPE_NO_ERROR, ""});
}

void InferenceContext::loadSingleTask(LookupHeadersCallback&& cb) {
  callback_headers_ = std::move(cb);
  task_.addTask(std::bind(
        &InferenceContext::processSingleTask, this));
}

/* ================================================================= */
/*
The top part mainly does the work of loading models and loading inference tasks, 
and the bottom part mainly does the work of model inference
*/
/* ================================================================= */

size_t find_stopping_strings(std::shared_ptr<server_slot>& slot, gpt_params& params, const std::string & text, const size_t last_token_size, bool full) {
  size_t stop_pos = std::string::npos;

  for (const std::string & word : params.antiprompt) {
    size_t pos;

    if (full) {
      const size_t tmp      = word.size() + last_token_size;
      const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

      pos = text.find(word, from_pos);
    } else {
      pos = find_partial_stop_string(word, text);
    }

    if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
      if (full) {
        slot->stopped_word   = true;
        slot->stopping_word  = word;
        slot->has_next_token = false;
      }
      stop_pos = pos;
    }
  }

  return stop_pos;
}

std::vector<llama_token> tokenize(llama_context *ctx, const json & json_prompt, bool add_special) {
  // TODO: currently, we tokenize using special tokens by default
  //       this is not always correct (see https://github.com/ggerganov/llama.cpp/pull/4160#issuecomment-1824826216)
  //       but it's better compared to completely ignoring ChatML and other chat templates
  const bool TMP_FORCE_SPECIAL = true;

  // If `add_bos` is true, we only add BOS, when json_prompt is a string,
  // or the first element of the json_prompt array is a string.
  std::vector<llama_token> prompt_tokens;

  if (json_prompt.is_array()) {
    bool first = true;
    for (const auto & p : json_prompt) {
      if (p.is_string()) {
        auto s = p.template get<std::string>();

        std::vector<llama_token> p;
        if (first) {
          p = ::llama_tokenize(ctx, s, add_special, TMP_FORCE_SPECIAL);
          first = false;
        } else {
          p = ::llama_tokenize(ctx, s, false, TMP_FORCE_SPECIAL);
        }

        prompt_tokens.insert(prompt_tokens.end(), p.begin(), p.end());
      } else {
        if (first) {
          first = false;
        }

        prompt_tokens.push_back(p.template get<llama_token>());
      }
    }
  } else {
    auto s = json_prompt.template get<std::string>();
    prompt_tokens = ::llama_tokenize(ctx, s, add_special, TMP_FORCE_SPECIAL);
  }

  return prompt_tokens;
}

void InferenceContext::sendPartialResponse(completion_token_output& tkn) {
  json res;
  res     = json {
    {"content",    tkn.text_to_send},
    {"stop",       false},
    {"multimodal", false}
  };

  if (slot->sparams.n_probs > 0) {
    const std::vector<llama_token> to_send_toks = llama_tokenize(ctx, tkn.text_to_send, false);
    const size_t probs_pos      = std::min(slot->n_sent_token_probs,                       slot->generated_token_probs.size());
    const size_t probs_stop_pos = std::min(slot->n_sent_token_probs + to_send_toks.size(), slot->generated_token_probs.size());

    std::vector<completion_token_output> probs_output;
    if (probs_pos < probs_stop_pos) {
      probs_output = std::vector<completion_token_output>(
          slot->generated_token_probs.begin() + probs_pos,
          slot->generated_token_probs.begin() + probs_stop_pos);
    }
    slot->n_sent_token_probs = probs_stop_pos;

    res["completion_probabilities"] = probs_vector_to_json(ctx, probs_output);
  }

  if (slot->oaicompat) {
    res["oaicompat_token_ctr"] = slot->n_decoded;
    res["model"] = slot->oaicompat_model;
  }

  std::vector<json> result_array = format_partial_response_oaicompat(res, completion_id_);
  
  for (auto it = result_array.begin(); it != result_array.end(); ++it) {
    if (!it->empty()) {
      const std::string str =
        "data: " +
        it->dump(-1, ' ', false, json::error_handler_t::replace) +
        "\n\n";
      callback_body_(ModelInferenceResult{true, false, ERROR_TYPE_NO_ERROR, str});
    }
  }
}

void InferenceContext::sendFinalResponse() {
  if (slot->params.stream) {
    callback_body_(ModelInferenceResult{true, true, ERROR_TYPE_NO_ERROR, ""});
    return;
  }

  json res;
  res = json {
    {"content",             !slot->params.stream ? slot->generated_text : ""},
    {"stop",                true},
    {"model",               params.model_alias},
    {"tokens_predicted",    slot->n_decoded},
    {"tokens_evaluated",    slot->n_prompt_tokens},
    {"prompt",              slot->prompt},
    {"truncated",           slot->truncated},
    {"stopped_eos",         slot->stopped_eos},
    {"stopped_word",        slot->stopped_word},
    {"stopped_limit",       slot->stopped_limit},
    {"stopping_word",       slot->stopping_word},
  };

  if (slot->sparams.n_probs > 0) {
    std::vector<completion_token_output> probs;
    if (!slot->params.stream && slot->stopped_word) {
      const std::vector<llama_token> stop_word_toks = llama_tokenize(ctx, slot->stopping_word, false);

      size_t safe_offset = std::min(slot->generated_token_probs.size(), stop_word_toks.size());
      probs = std::vector<completion_token_output>(
              slot->generated_token_probs.begin(),
              slot->generated_token_probs.end() - safe_offset);
    } else {
      probs = std::vector<completion_token_output>(
            slot->generated_token_probs.begin(),
            slot->generated_token_probs.end());
    }

    res["completion_probabilities"] = probs_vector_to_json(ctx, probs);
  }

  if (slot->oaicompat) {
    res["oaicompat_token_ctr"] = slot->n_decoded;
    res["model"] = slot->oaicompat_model;
  }

  json result_oai = format_final_response_oaicompat(slot->data, res, completion_id_);
  callback_body_(ModelInferenceResult{true, true, ERROR_TYPE_NO_ERROR, result_oai.dump(-1, ' ', false, json::error_handler_t::replace)});
}

bool InferenceContext::processToken(completion_token_output & result) {
  // remember which tokens were sampled - used for repetition penalties during sampling
  const std::string token_str = llama_token_to_piece(ctx, result.tok, false);
  slot->sampled = result.tok;
  // std::cout <<  << std::endl;
  // search stop word and delete it
  slot->generated_text += token_str;
  slot->has_next_token = true;

  // check if there is incomplete UTF-8 character at the end
  bool incomplete = false;
  for (unsigned i = 1; i < 5 && i <= slot->generated_text.size(); ++i) {
    unsigned char c = slot->generated_text[slot->generated_text.size() - i];
    if ((c & 0xC0) == 0x80) {
      // continuation byte: 10xxxxxx
      continue;
    }
    if ((c & 0xE0) == 0xC0) {
      // 2-byte character: 110xxxxx ...
      incomplete = i < 2;
    } else if ((c & 0xF0) == 0xE0) {
      // 3-byte character: 1110xxxx ...
      incomplete = i < 3;
    } else if ((c & 0xF8) == 0xF0) {
      // 4-byte character: 11110xxx ...
      incomplete = i < 4;
    }
    // else 1-byte character or invalid byte
    break;
  }

  if (!incomplete) {
    size_t pos = std::min(slot->n_sent_text, slot->generated_text.size());

    const std::string str_test = slot->generated_text.substr(pos);
    bool is_stop_full = false;

    size_t stop_pos = find_stopping_strings(slot, params, str_test, token_str.size(), true);
    if (stop_pos != std::string::npos) {
      is_stop_full = true;
      slot->generated_text.erase(
        slot->generated_text.begin() + pos + stop_pos,
        slot->generated_text.end());
      pos = std::min(slot->n_sent_text, slot->generated_text.size());
    } else {
      is_stop_full = false;
      stop_pos = find_stopping_strings(slot, params, str_test, token_str.size(), false);
    }

    // check if there is any token to predict
    if (stop_pos == std::string::npos || (!slot->has_next_token && !is_stop_full && stop_pos > 0)) {
      // no send the stop word in the response
      result.text_to_send = slot->generated_text.substr(pos, std::string::npos);
      slot->n_sent_text += result.text_to_send.size();
      // add the token to slot queue and cache
    }

    slot->add_token_string(result);
  }

  if (incomplete) {
    slot->has_next_token = true;
  }

  if (llama_token_is_eog(model, result.tok)) {
    slot->stopped_eos    = true;
    slot->has_next_token = false;
  }

  auto n_ctx_train = llama_n_ctx_train(model);
  if (slot->params.n_predict < 1 && slot->n_predict < 1
        && slot->n_prompt_tokens + slot->n_decoded >= n_ctx_train) {
    slot->truncated      = true;
    slot->stopped_limit  = true;
    slot->has_next_token = false; // stop prediction
  }

  if (slot->params.stream) {
    sendPartialResponse(result);
  }
  return slot->has_next_token; // continue
}

bool InferenceContext::updateSlots() {
  // evaluate the initial prompt
  slot->prompt_tokens = tokenize(ctx, slot->prompt, true);
  slot->n_prompt_tokens = slot->prompt_tokens.size();
  for (int i = 0; i < slot->n_prompt_tokens; i++) {
    llama_batch_add(batch, slot->prompt_tokens[i], i, { 0 }, false);
  }

  // llama_decode will output logits only for the last token of the prompt
  batch.logits[batch.n_tokens - 1] = true;
  if (llama_decode(ctx, batch) != 0) {
    return false;
  }
  int n_cur    = batch.n_tokens;

  while (n_cur < n_ctx) {
    {
      // prompt evaluated for embedding
      // if (slot->embedding) {
      //   send_embedding();
      //   return;
      // }

      completion_token_output result;
      const llama_token id = llama_sampling_sample(slot->ctx_sampling, ctx, NULL);

      llama_sampling_accept(slot->ctx_sampling, ctx, id, true);

      slot->n_decoded += 1;

      llama_token_data_array cur_p = { slot->ctx_sampling->cur.data(), slot->ctx_sampling->cur.size(), false };
      result.tok = id;

      const size_t n_probs = std::min(cur_p.size, static_cast<size_t>(slot->sparams.n_probs));
      if (n_probs > 0) {
        const size_t n_valid = slot->ctx_sampling->n_valid;

        // Make sure at least n_probs top tokens are at the front of the vector:
        if (slot->sparams.temp == 0.0f && n_probs > n_valid) {
          llama_sample_top_k(ctx, &cur_p, n_probs, 0);
        }

        if (slot->sparams.temp == 0.0f) {
        // With greedy sampling the probabilities have possibly not been calculated.
          for (size_t i = 0; i < n_probs; ++i) {
            result.probs.push_back({
              cur_p.data[i].id,
              i == 0 ? 1.0f : 0.0f
            });
          }
        } else {
          for (size_t i = 0; i < n_probs; ++i) {
            result.probs.push_back({
              cur_p.data[i].id,
              i >= n_valid ? 0.0f : cur_p.data[i].p // Tokens filtered out due to e.g. top_k have 0 probability.
            });
          }
        }
      }

      if (!processToken(result)) {
        sendFinalResponse();
        break;
      }
      // prepare the next batch
      llama_batch_clear(batch);
      // push this new token for next evaluation
      llama_batch_add(batch, id, n_cur, { 0 }, true);
    }
    n_cur += 1;
    // evaluate the current batch with the transformer model
    if (llama_decode(ctx, batch) != 0) {
      return false;
    }
  }
  return true;
}

void InferenceContext::generate() {
  completion_id_ = gen_chatcmplid();
  if (!updateSlots()) {
    callback_body_(ModelInferenceResult{false, true, ERROR_TYPE_UNAVAILABLE, "llama_decode was wrong!"});
    return;
  }
}

void InferenceContext::modelInference(LookupBodyCallback&& cb) {
  callback_body_ = std::move(cb);
  task_.addTask(std::bind(
        &InferenceContext::generate, this));
}

} // namespace LLMInference
} // namespace HttpFilters
} // namespace Extensions
} // namespace Envoy
