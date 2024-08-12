#include "contrib/llm_inference/filters/http/source/config.h"

#include "contrib/llm_inference/filters/http/source/llm_inference_filter.h"

namespace Envoy {
namespace Extensions {
namespace HttpFilters {
namespace LLMInference {

class InferenceSingleton : public Envoy::Singleton::Instance {
public:
  InferenceSingleton(Thread::ThreadFactory& thread_factory)
      : inference_thread_(thread_factory) {}

  std::shared_ptr<InferenceTask> get(std::shared_ptr<InferenceSingleton> singleton,
                                           const int& id_task) {
    std::shared_ptr<InferenceTask> task;
    absl::MutexLock lock(&mu_);
    auto it = task_.find(id_task);
    if (it != task_.end()) {
      task = it->second.lock();
    }
    if (!task) {
      task = std::make_shared<InferenceTask>(singleton, inference_thread_);
      task_[id_task] = task;
    }
    return task;
  }

private:
  // std::shared_ptr<Common::AsyncFiles::AsyncFileManagerFactory> async_file_manager_factory_;
  InferenceThread inference_thread_;
  absl::Mutex mu_;
  // We keep weak_ptr here so the caches can be destroyed if the config is updated to stop using
  // that config of cache. The caches each keep shared_ptrs to this singleton, which keeps the
  // singleton from being destroyed unless it's no longer keeping track of any caches.
  // (The singleton shared_ptr is *only* held by cache instances.)
  absl::flat_hash_map<int, std::weak_ptr<InferenceTask>> task_ ABSL_GUARDED_BY(mu_);
  // absl::flat_hash_map<std::string, std::weak_ptr<FileSystemHttpCache>> caches_ ABSL_GUARDED_BY(mu_);
};

SINGLETON_MANAGER_REGISTRATION(http_inference_singleton);

Http::FilterFactoryCb LLMInferenceFilterConfigFactory::createFilterFactoryFromProtoTyped(
    const envoy::extensions::filters::http::llm_inference::v3::modelParameter& proto_config,
    const std::string&, Server::Configuration::FactoryContext& context) {

    LLMInferenceFilterConfigSharedPtr config =
        std::make_shared<LLMInferenceFilterConfig>(LLMInferenceFilterConfig(proto_config));

    std::shared_ptr<InferenceSingleton> inference =
        context.singletonManager().getTyped<InferenceSingleton>(
            SINGLETON_MANAGER_REGISTERED_NAME(http_inference_singleton), [&context] {
              return std::make_shared<InferenceSingleton>(context.api().threadFactory());
            });
    LLMInferenceTaskSharedPtr task = inference->get(inference, 1);

    std::cout << "n_thread: " << config->n_thread()<< std::endl;

    const auto a = modelPath();
    for (const auto& pair : a) {
      std::cout << pair.first << ": " << pair.second << std::endl;
    }

    return [config, task](Http::FilterChainFactoryCallbacks& callbacks) -> void {
      callbacks.addStreamDecoderFilter(std::make_shared<LLMInferenceFilter>(config, task));
    };
}


Router::RouteSpecificFilterConfigConstSharedPtr LLMInferenceFilterConfigFactory::createRouteSpecificFilterConfigTyped(
    const envoy::extensions::filters::http::llm_inference::v3::modelChosen& proto_config,
    Server::Configuration::ServerFactoryContext&, ProtobufMessage::ValidationVisitor&) {
    return std::make_shared<const LLMInferenceFilterConfigPerRoute>(proto_config);
}

/**
 * Static registration for this llm inference filter. @see RegisterFactory.
 */
REGISTER_FACTORY(LLMInferenceFilterConfigFactory, Server::Configuration::NamedHttpFilterConfigFactory);

} // namespace LLMInference
} // namespace HttpFilters
} // namespace Extensions
} // namespace Envoy