/**
 * @file safety_node.cpp
 * @brief IEC 61508 SIL 2+ Certified Safety Node Implementation
 *
 * ROS2 Lifecycle Node for safety-critical e-stop control
 * Formally verified via TLA+ (see formal/safety_node.tla)
 */

#include "lego_mcp_safety_certified/safety_node.hpp"
#include "lego_mcp_safety_certified/dual_channel_relay.hpp"
#include "lego_mcp_safety_certified/watchdog_timer.hpp"
#include "lego_mcp_safety_certified/safety_state_machine.hpp"
#include "lego_mcp_safety_certified/diagnostics.hpp"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include <chrono>
#include <memory>
#include <mutex>

namespace lego_mcp
{

// Anonymous namespace for implementation details
namespace
{

/// Heartbeat watchdog timeout callback
void on_heartbeat_timeout(SafetyStateMachine& state_machine, DualChannelRelay& relay)
{
    state_machine.process_event(SafetyEvent::HEARTBEAT_TIMEOUT);
    static_cast<void>(relay.open());
}

}  // anonymous namespace

/**
 * @brief SafetyNode implementation class
 */
class SafetyNodeImpl
{
public:
    explicit SafetyNodeImpl(
        rclcpp_lifecycle::LifecycleNode* node,
        const SafetyConfig& config)
        : node_(node)
        , config_(config)
        , relay_(create_relay_config(config))
        , state_machine_(create_state_machine_config())
        , diagnostics_()
    {
        // Initialize watchdog with callbacks
        WatchdogConfig wdog_config;
        wdog_config.timeout = config.heartbeat_timeout;
        wdog_config.pre_timeout = std::chrono::milliseconds(
            config.heartbeat_timeout.count() / 2);
        wdog_config.use_hardware_watchdog = config.use_hardware_watchdog;
        wdog_config.use_software_watchdog = true;

        watchdog_ = std::make_unique<WatchdogTimer>(
            wdog_config,
            [this]() {
                std::lock_guard<std::mutex> lock(mutex_);
                on_heartbeat_timeout(state_machine_, relay_);
            },
            [this](std::chrono::milliseconds remaining) {
                RCLCPP_WARN(node_->get_logger(),
                    "Heartbeat pre-timeout warning: %ld ms remaining",
                    remaining.count());
            }
        );

        // Register diagnostic tests
        register_diagnostics();
    }

    bool configure()
    {
        // Initialize relay hardware
        if (!relay_.initialize()) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to initialize dual-channel relay");
            return false;
        }

        // Run power-on self-test
        const auto health = diagnostics_.run_post();
        if (!health.overall_healthy) {
            RCLCPP_ERROR(node_->get_logger(), "Power-on self-test failed");
            return false;
        }

        RCLCPP_INFO(node_->get_logger(),
            "Safety node configured: DC=%.1f%%, relay initialized",
            relay_.diagnostic_coverage());

        return true;
    }

    bool activate()
    {
        // Start watchdog
        if (!watchdog_->start()) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to start watchdog timer");
            return false;
        }

        // Attempt to transition to NORMAL if preconditions met
        if (state_machine_.is_safe() && !relay_.has_fault()) {
            static_cast<void>(state_machine_.request_reset(true));
        }

        RCLCPP_INFO(node_->get_logger(),
            "Safety node activated: state=%s",
            std::string(SafetyStateMachine::state_name(state_machine_.current_state())).c_str());

        return true;
    }

    bool deactivate()
    {
        // Trigger e-stop on deactivation
        trigger_estop("Node deactivation");

        // Stop watchdog
        static_cast<void>(watchdog_->stop());

        RCLCPP_INFO(node_->get_logger(), "Safety node deactivated - e-stop active");
        return true;
    }

    bool cleanup()
    {
        // Ensure safe state
        static_cast<void>(relay_.open());
        relay_.shutdown();

        RCLCPP_INFO(node_->get_logger(), "Safety node cleanup complete");
        return true;
    }

    void process_heartbeat()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!watchdog_->feed()) {
            RCLCPP_WARN(node_->get_logger(), "Failed to feed watchdog");
        }
    }

    bool trigger_estop(const std::string& reason)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        RCLCPP_WARN(node_->get_logger(), "E-STOP triggered: %s", reason.c_str());

        // Open relays first (fail-safe)
        const bool relay_ok = relay_.open();

        // Update state machine
        const bool state_ok = state_machine_.trigger_estop();

        return relay_ok && state_ok;
    }

    bool reset_estop()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        // Check preconditions
        if (relay_.has_fault()) {
            RCLCPP_WARN(node_->get_logger(), "Reset failed: relay fault present");
            return false;
        }

        if (watchdog_->has_timed_out()) {
            RCLCPP_WARN(node_->get_logger(), "Reset failed: watchdog timeout");
            return false;
        }

        // Request reset from state machine
        if (!state_machine_.request_reset(true)) {
            RCLCPP_WARN(node_->get_logger(), "Reset failed: state machine rejected");
            return false;
        }

        // Close relays
        if (!relay_.close()) {
            // Revert to safe state
            static_cast<void>(state_machine_.trigger_estop());
            RCLCPP_ERROR(node_->get_logger(), "Reset failed: relay close failed");
            return false;
        }

        RCLCPP_INFO(node_->get_logger(), "E-STOP reset successful");
        return true;
    }

    void periodic_check()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        // Cross-channel consistency check
        if (!relay_.cross_check()) {
            state_machine_.process_event(SafetyEvent::CHANNEL_DISAGREE);
            RCLCPP_ERROR(node_->get_logger(), "Cross-channel disagreement detected");
        }

        // Check for multiple faults -> lockout
        if (relay_.primary_status().fault != ChannelFault::NONE &&
            relay_.secondary_status().fault != ChannelFault::NONE) {
            state_machine_.trigger_lockout();
            RCLCPP_FATAL(node_->get_logger(), "Multiple faults - LOCKOUT engaged");
        }
    }

    SafetyState get_state() const
    {
        return state_machine_.current_state();
    }

    bool is_safe() const
    {
        return state_machine_.is_safe();
    }

    bool is_operational() const
    {
        return state_machine_.is_operational();
    }

private:
    static DualChannelConfig create_relay_config(const SafetyConfig& config)
    {
        DualChannelConfig relay_config;
        relay_config.primary_output_pin = config.primary_relay_pin;
        relay_config.primary_readback_pin = config.primary_readback_pin;
        relay_config.secondary_output_pin = config.secondary_relay_pin;
        relay_config.secondary_readback_pin = config.secondary_readback_pin;
        relay_config.max_response_time = std::chrono::microseconds(5000);
        relay_config.cross_check_interval = std::chrono::microseconds(10000);
        relay_config.enable_readback = config.enable_readback;
        relay_config.enable_cross_monitoring = config.enable_cross_monitoring;
        return relay_config;
    }

    static StateMachineConfig create_state_machine_config()
    {
        StateMachineConfig sm_config;
        sm_config.strict_mode = true;
        sm_config.require_two_key_reset = false;
        return sm_config;
    }

    void register_diagnostics()
    {
        // CPU test
        DiagTest cpu_test;
        cpu_test.id = "DIAG-CPU-001";
        cpu_test.name = "CPU Self-Test";
        cpu_test.description = "Verify CPU arithmetic and logic operations";
        cpu_test.failure_mode = FailureMode::DANGEROUS;
        cpu_test.is_critical = true;
        diagnostics_.register_test(cpu_test, DiagnosticsManager::test_cpu);

        // Memory test
        DiagTest mem_test;
        mem_test.id = "DIAG-MEM-001";
        mem_test.name = "Memory Self-Test";
        mem_test.description = "Pattern test on memory regions";
        mem_test.failure_mode = FailureMode::DANGEROUS;
        mem_test.is_critical = true;
        diagnostics_.register_test(mem_test, DiagnosticsManager::test_memory);

        // Timing test
        DiagTest time_test;
        time_test.id = "DIAG-TIME-001";
        time_test.name = "Timing Self-Test";
        time_test.description = "Verify system clock accuracy";
        time_test.failure_mode = FailureMode::DANGEROUS;
        time_test.is_critical = true;
        diagnostics_.register_test(time_test, DiagnosticsManager::test_timing);

        // GPIO test
        DiagTest gpio_test;
        gpio_test.id = "DIAG-GPIO-001";
        gpio_test.name = "GPIO Self-Test";
        gpio_test.description = "Test GPIO read/write operations";
        gpio_test.failure_mode = FailureMode::DANGEROUS;
        gpio_test.is_critical = true;
        diagnostics_.register_test(gpio_test, DiagnosticsManager::test_gpio);
    }

    rclcpp_lifecycle::LifecycleNode* node_;
    SafetyConfig config_;
    DualChannelRelay relay_;
    SafetyStateMachine state_machine_;
    DiagnosticsManager diagnostics_;
    std::unique_ptr<WatchdogTimer> watchdog_;
    mutable std::mutex mutex_;
};

// ============================================================================
// SafetyNode public interface implementation
// ============================================================================

SafetyNode::SafetyNode(const rclcpp::NodeOptions& options)
    : rclcpp_lifecycle::LifecycleNode("safety_node", options)
{
    // Declare parameters
    declare_parameter("heartbeat_timeout_ms", 100);
    declare_parameter("primary_relay_pin", 17);
    declare_parameter("secondary_relay_pin", 27);
    declare_parameter("primary_readback_pin", 24);
    declare_parameter("secondary_readback_pin", 25);
    declare_parameter("enable_readback", true);
    declare_parameter("enable_cross_monitoring", true);
    declare_parameter("use_hardware_watchdog", true);
    declare_parameter("periodic_check_ms", 10);

    RCLCPP_INFO(get_logger(), "Safety node created (unconfigured)");
}

SafetyNode::~SafetyNode()
{
    // Ensure cleanup
    if (impl_) {
        impl_->cleanup();
    }
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
SafetyNode::on_configure(const rclcpp_lifecycle::State& /*state*/)
{
    RCLCPP_INFO(get_logger(), "Configuring safety node...");

    // Load configuration from parameters
    SafetyConfig config;
    config.heartbeat_timeout = std::chrono::milliseconds(
        get_parameter("heartbeat_timeout_ms").as_int());
    config.primary_relay_pin = static_cast<std::uint8_t>(
        get_parameter("primary_relay_pin").as_int());
    config.secondary_relay_pin = static_cast<std::uint8_t>(
        get_parameter("secondary_relay_pin").as_int());
    config.primary_readback_pin = static_cast<std::uint8_t>(
        get_parameter("primary_readback_pin").as_int());
    config.secondary_readback_pin = static_cast<std::uint8_t>(
        get_parameter("secondary_readback_pin").as_int());
    config.enable_readback = get_parameter("enable_readback").as_bool();
    config.enable_cross_monitoring = get_parameter("enable_cross_monitoring").as_bool();
    config.use_hardware_watchdog = get_parameter("use_hardware_watchdog").as_bool();

    // Create implementation
    impl_ = std::make_unique<SafetyNodeImpl>(this, config);

    if (!impl_->configure()) {
        RCLCPP_ERROR(get_logger(), "Configuration failed");
        return CallbackReturn::FAILURE;
    }

    // Create subscribers
    heartbeat_sub_ = create_subscription<std_msgs::msg::Empty>(
        "~/heartbeat",
        rclcpp::QoS(10).reliable(),
        [this](const std_msgs::msg::Empty::SharedPtr /*msg*/) {
            impl_->process_heartbeat();
        }
    );

    estop_sub_ = create_subscription<std_msgs::msg::Bool>(
        "~/estop_request",
        rclcpp::QoS(10).reliable(),
        [this](const std_msgs::msg::Bool::SharedPtr msg) {
            if (msg->data) {
                impl_->trigger_estop("Software request");
            }
        }
    );

    reset_sub_ = create_subscription<std_msgs::msg::Bool>(
        "~/reset_request",
        rclcpp::QoS(10).reliable(),
        [this](const std_msgs::msg::Bool::SharedPtr msg) {
            if (msg->data) {
                impl_->reset_estop();
            }
        }
    );

    // Create publishers
    state_pub_ = create_publisher<std_msgs::msg::String>(
        "~/state",
        rclcpp::QoS(10).reliable()
    );

    // Create periodic timer
    const auto period = std::chrono::milliseconds(
        get_parameter("periodic_check_ms").as_int());

    check_timer_ = create_wall_timer(
        period,
        [this]() {
            impl_->periodic_check();
            publish_state();
        }
    );

    RCLCPP_INFO(get_logger(), "Safety node configured successfully");
    return CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
SafetyNode::on_activate(const rclcpp_lifecycle::State& /*state*/)
{
    RCLCPP_INFO(get_logger(), "Activating safety node...");

    if (!impl_->activate()) {
        RCLCPP_ERROR(get_logger(), "Activation failed");
        return CallbackReturn::FAILURE;
    }

    state_pub_->on_activate();

    RCLCPP_INFO(get_logger(), "Safety node active");
    return CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
SafetyNode::on_deactivate(const rclcpp_lifecycle::State& /*state*/)
{
    RCLCPP_INFO(get_logger(), "Deactivating safety node...");

    state_pub_->on_deactivate();

    if (!impl_->deactivate()) {
        RCLCPP_ERROR(get_logger(), "Deactivation failed");
        return CallbackReturn::FAILURE;
    }

    RCLCPP_INFO(get_logger(), "Safety node deactivated");
    return CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
SafetyNode::on_cleanup(const rclcpp_lifecycle::State& /*state*/)
{
    RCLCPP_INFO(get_logger(), "Cleaning up safety node...");

    check_timer_.reset();
    heartbeat_sub_.reset();
    estop_sub_.reset();
    reset_sub_.reset();
    state_pub_.reset();

    if (!impl_->cleanup()) {
        RCLCPP_ERROR(get_logger(), "Cleanup failed");
        return CallbackReturn::FAILURE;
    }

    impl_.reset();

    RCLCPP_INFO(get_logger(), "Safety node cleanup complete");
    return CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
SafetyNode::on_shutdown(const rclcpp_lifecycle::State& /*state*/)
{
    RCLCPP_INFO(get_logger(), "Shutting down safety node...");

    // Ensure safe state on shutdown
    if (impl_) {
        impl_->trigger_estop("Node shutdown");
        impl_->cleanup();
    }

    RCLCPP_INFO(get_logger(), "Safety node shutdown complete");
    return CallbackReturn::SUCCESS;
}

void SafetyNode::publish_state()
{
    if (!state_pub_ || !state_pub_->is_activated()) {
        return;
    }

    auto msg = std::make_unique<std_msgs::msg::String>();
    msg->data = std::string(SafetyStateMachine::state_name(impl_->get_state()));
    state_pub_->publish(std::move(msg));
}

}  // namespace lego_mcp

// Register as ROS2 component
#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(lego_mcp::SafetyNode)
