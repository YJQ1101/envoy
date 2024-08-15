#include "contrib/llm_inference/filters/http/source/inference/inference_thread.h"

#include "envoy/thread/thread.h"
#include "inference_context.h"
#include <algorithm>
#include <cstdio>
#include <unistd.h>

namespace Envoy {
namespace Extensions {
namespace HttpFilters {
namespace LLMInference {
InferenceThread::InferenceThread(Thread::ThreadFactory& thread_factory)
    : thread_(thread_factory.createThread([this]() { work(); })) {}


InferenceThread::~InferenceThread() {
  terminate();
  thread_->join();
}

void InferenceThread::addTask(std::function<void(void)> callback) {
  {
    absl::MutexLock lock(&cache_mu_);
    tasks_.push_back(std::move(callback));
  }
  // Signal to unblock InferenceThread to perform the initial cache measurement
  // (and possibly eviction if it's starting out oversized!)
  signal();
}

// void InferenceThread::removeTask(std::shared_ptr<InferenceTaskMetaData>& task) {
//   absl::MutexLock lock(&cache_mu_);
  
//   tasks_.erase(task);
// }

void InferenceThread::signal() {
  absl::MutexLock lock(&mu_);
  signalled_ = true;
}

void InferenceThread::terminate() {
  absl::MutexLock lock(&mu_);
  terminating_ = true;
  signalled_ = true;
}

bool InferenceThread::waitForSignal() {
  absl::MutexLock lock(&mu_);
  // Worth noting here that if `signalled_` is already true, the lock is not released
  // until idle_ is false again, so waitForIdle will not return until `signalled_`
  // stays false for the duration of an eviction cycle.
  idle_ = true;
  mu_.Await(absl::Condition(&signalled_));
  signalled_ = false;
  idle_ = false;
  return !terminating_;
}

void InferenceThread::work() {
  // ENVOY_LOG(info, "Starting cache eviction thread.");
  
  while (waitForSignal()) {
    std::vector<std::function<void(void)>> tasks;
    {
      // Take a local copy of the set of caches, so we don't hold the lock while
      // work is being performed.
      absl::MutexLock lock(&cache_mu_);
      tasks = std::move(tasks_);
    }

    for (const std::function<void(void)>& callback_context_function: tasks) {
      std::cout << "receive signal\n";
      callback_context_function();
    }   
  }
  // ENVOY_LOG(info, "Ending cache eviction thread.");
}

// void InferenceThread::waitForIdle() {
//   absl::MutexLock lock(&mu_);
//   auto cond = [this]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) { return idle_ && !signalled_; };
//   mu_.Await(absl::Condition(&cond));
// }

} // namespace LLMInference
} // namespace HttpFilters
} // namespace Extensions
} // namespace Envoy
