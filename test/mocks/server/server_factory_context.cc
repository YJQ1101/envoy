#include "test/mocks/server/server_factory_context.h"

namespace Envoy {
namespace Server {
namespace Configuration {

using ::testing::Return;
using ::testing::ReturnRef;

MockServerFactoryContext::MockServerFactoryContext()
    : singleton_manager_(new Singleton::ManagerImpl(Thread::threadFactoryForTest())),
      grpc_context_(store_.symbolTable()), router_context_(store_.symbolTable()) {
  ON_CALL(*this, clusterManager()).WillByDefault(ReturnRef(cluster_manager_));
  ON_CALL(*this, mainThreadDispatcher()).WillByDefault(ReturnRef(dispatcher_));
  ON_CALL(*this, drainDecision()).WillByDefault(ReturnRef(drain_manager_));
  ON_CALL(*this, localInfo()).WillByDefault(ReturnRef(local_info_));
  ON_CALL(*this, runtime()).WillByDefault(ReturnRef(runtime_loader_));
  ON_CALL(*this, scope()).WillByDefault(ReturnRef(*store_.rootScope()));
  ON_CALL(*this, serverScope()).WillByDefault(ReturnRef(*store_.rootScope()));
  ON_CALL(*this, singletonManager()).WillByDefault(ReturnRef(*singleton_manager_));
  ON_CALL(*this, threadLocal()).WillByDefault(ReturnRef(thread_local_));
  ON_CALL(*this, admin()).WillByDefault(Return(OptRef<Server::Admin>{admin_}));
  ON_CALL(*this, api()).WillByDefault(ReturnRef(api_));
  ON_CALL(*this, timeSource()).WillByDefault(ReturnRef(time_system_));
  ON_CALL(*this, messageValidationContext()).WillByDefault(ReturnRef(validation_context_));
  ON_CALL(*this, messageValidationVisitor())
      .WillByDefault(ReturnRef(ProtobufMessage::getStrictValidationVisitor()));
  ON_CALL(*this, api()).WillByDefault(ReturnRef(api_));
  ON_CALL(*this, drainManager()).WillByDefault(ReturnRef(drain_manager_));
  ON_CALL(*this, statsConfig()).WillByDefault(ReturnRef(stats_config_));
  ON_CALL(*this, accessLogManager()).WillByDefault(ReturnRef(access_log_manager_));
  ON_CALL(*this, initManager()).WillByDefault(ReturnRef(init_manager_));
  ON_CALL(*this, lifecycleNotifier()).WillByDefault(ReturnRef(lifecycle_notifier_));
  ON_CALL(*this, options()).WillByDefault(ReturnRef(options_));
}
MockServerFactoryContext::~MockServerFactoryContext() = default;

MockStatsConfig::MockStatsConfig() = default;
MockStatsConfig::~MockStatsConfig() = default;

StatelessMockServerFactoryContext::StatelessMockServerFactoryContext()
    : filter_config_provider_manager_(
          std::make_shared<Filter::HttpFilterConfigProviderManagerImpl>()) {}

} // namespace Configuration
} // namespace Server
} // namespace Envoy
