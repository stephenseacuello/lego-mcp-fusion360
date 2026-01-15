cmake_minimum_required(VERSION 3.16)
project(lego_mcp_safety_certified)

# ==============================================================================
# IEC 61508 SIL 2+ Safety-Critical Build Configuration
# ==============================================================================

# Require C++20 for safety-critical features
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# MISRA C++ 2023 Compliance Flags
set(SAFETY_CRITICAL_FLAGS
    -Wall
    -Wextra
    -Wpedantic
    -Werror
    -Wconversion
    -Wsign-conversion
    -Wcast-qual
    -Wcast-align
    -Wformat=2
    -Wformat-security
    -Wnull-dereference
    -Wstack-protector
    -fstack-protector-strong
    -D_FORTIFY_SOURCE=2
    -fno-exceptions      # No exceptions in safety-critical code
    -fno-rtti            # No RTTI overhead
    -fno-threadsafe-statics  # Explicit initialization control
)

# Debug vs Release configurations
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND SAFETY_CRITICAL_FLAGS -O0 -g3 -DDEBUG_BUILD)
else()
    # Release with debug info for WCET analysis
    list(APPEND SAFETY_CRITICAL_FLAGS -O2 -g -DNDEBUG -DRELEASE_BUILD)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(lifecycle_msgs REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(lego_mcp_msgs REQUIRED)

# ==============================================================================
# Safety-Critical Library
# ==============================================================================

add_library(safety_core SHARED
    src/safety_node.cpp
    src/dual_channel_relay.cpp
    src/watchdog_timer.cpp
    src/safety_state_machine.cpp
    src/diagnostics.cpp
)

target_include_directories(safety_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_options(safety_core PRIVATE ${SAFETY_CRITICAL_FLAGS})

ament_target_dependencies(safety_core
    rclcpp
    rclcpp_lifecycle
    rclcpp_components
    std_msgs
    std_srvs
    diagnostic_msgs
    lifecycle_msgs
    realtime_tools
    lego_mcp_msgs
)

# ==============================================================================
# Executable Nodes
# ==============================================================================

add_executable(safety_node src/main.cpp)
target_link_libraries(safety_node safety_core)
target_compile_options(safety_node PRIVATE ${SAFETY_CRITICAL_FLAGS})

add_executable(safety_lifecycle_node src/main_lifecycle.cpp)
target_link_libraries(safety_lifecycle_node safety_core)
target_compile_options(safety_lifecycle_node PRIVATE ${SAFETY_CRITICAL_FLAGS})

# ==============================================================================
# Component Registration
# ==============================================================================

rclcpp_components_register_node(safety_core
    PLUGIN "lego_mcp::SafetyNode"
    EXECUTABLE safety_node_component
)

# ==============================================================================
# Installation
# ==============================================================================

install(DIRECTORY include/
    DESTINATION include
)

install(DIRECTORY config/
    DESTINATION share/${PROJECT_NAME}/config
)

install(DIRECTORY launch/
    DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY formal/
    DESTINATION share/${PROJECT_NAME}/formal
)

install(TARGETS
    safety_core
    safety_node
    safety_lifecycle_node
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# ==============================================================================
# Testing (Safety-Critical Test Suite)
# ==============================================================================

if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    find_package(ament_cmake_gtest REQUIRED)

    # Unit tests
    ament_add_gtest(test_safety_state_machine test/test_safety_state_machine.cpp)
    target_link_libraries(test_safety_state_machine safety_core)

    ament_add_gtest(test_dual_channel_relay test/test_dual_channel_relay.cpp)
    target_link_libraries(test_dual_channel_relay safety_core)

    ament_add_gtest(test_watchdog_timer test/test_watchdog_timer.cpp)
    target_link_libraries(test_watchdog_timer safety_core)

    # WCET analysis (requires special instrumentation)
    add_executable(wcet_instrumented_safety src/safety_node.cpp)
    target_link_libraries(wcet_instrumented_safety safety_core)
    target_compile_options(wcet_instrumented_safety PRIVATE
        ${SAFETY_CRITICAL_FLAGS}
        -finstrument-functions  # WCET instrumentation
    )

    ament_lint_auto_find_test_dependencies()
endif()

ament_package()
