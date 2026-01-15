# SROS2 Security Setup Guide

## LEGO MCP Fusion 360 - Industry 4.0/5.0 Architecture

This guide covers the setup and configuration of SROS2 security for the LEGO MCP system.

---

## Overview

The LEGO MCP system implements security according to:
- **IEC 62443** - Industrial Automation and Control Systems Security
- **NIST 800-82** - Guide to Industrial Control Systems Security
- **ISO/IEC 27001** - Information Security Management

### Security Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     LEGO MCP Security Architecture                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    SROS2 Security Layer                          │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │    │
│  │  │ Key Mgmt │  │ Encrypt  │  │  Auth    │  │ Access Control   │ │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                   IEC 62443 Security Zones                       │    │
│  │                                                                   │    │
│  │  Zone 0       Zone 1        Zone 2         Zone 3       Zone 4   │    │
│  │  Safety       Control       Supervisory    MES          Enterprise│    │
│  │  SL-4         SL-3          SL-2           SL-2         SL-1     │    │
│  │                                                                   │    │
│  │  ──────────── Conduits ──────────────────────────────────────────│    │
│  │                                                                   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                   Security Services                              │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │    │
│  │  │ Audit Trail  │  │   IDS        │  │ Compliance Checking │   │    │
│  │  │ (Hash Chain) │  │ (Anomaly)    │  │ (IEC 62443)        │   │    │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

- ROS2 Humble or later
- OpenSSL 1.1+
- Python 3.10+

---

## Quick Start

### 1. Generate Keystore

```bash
# Navigate to security package
cd ros2_ws/src/lego_mcp_security

# Run keystore generation script
./scripts/generate_keystore.sh /path/to/keystore 0
```

### 2. Enable SROS2

```bash
# Set environment variables
export ROS_SECURITY_KEYSTORE=/path/to/keystore
export ROS_SECURITY_ENABLE=true
export ROS_SECURITY_STRATEGY=Enforce
export ROS_DOMAIN_ID=0
```

### 3. Launch Secure Nodes

```bash
# Source ROS2 workspace
source install/setup.bash

# Launch with security
ros2 launch lego_mcp_security security.launch.py \
  keystore_path:=/path/to/keystore
```

---

## Security Zones (IEC 62443)

### Zone Definitions

| Zone | Name | Security Level | Description |
|------|------|---------------|-------------|
| 0 | Safety | SL-4 | E-stop, watchdog, safety interlocks |
| 1 | Control | SL-3 | Equipment nodes (CNC, printers, robots) |
| 2 | Supervisory | SL-2 | Orchestrator, AGV fleet, vision |
| 3 | MES | SL-2 | Dashboard, scheduling, digital twin |
| 4 | Enterprise | SL-1 | Cloud connectors, analytics |

### Security Levels

- **SL-4**: Protection against intentional violation using sophisticated means
- **SL-3**: Protection against intentional violation using moderate resources
- **SL-2**: Protection against intentional violation using simple means
- **SL-1**: Protection against casual or coincidental violation

### Zone Configuration

Edit `config/security_policy.yaml`:

```yaml
security_zones:
  zone_0_safety:
    name: "Safety Zone"
    security_level: SL-4
    nodes:
      - safety_node
      - watchdog_node
    allowed_protocols:
      - DDS_ENCRYPTED
    access_policy: DENY_ALL_EXCEPT_LISTED
```

---

## Key Management

### Keystore Structure

```
keystore/
├── enclaves/
│   ├── safety_node/
│   │   ├── cert.pem
│   │   ├── key.pem
│   │   └── permissions.xml
│   ├── orchestrator_node/
│   │   ├── cert.pem
│   │   ├── key.pem
│   │   └── permissions.xml
│   └── ...
├── private/
│   └── ca.key.pem
├── public/
│   ├── ca.cert.pem
│   └── permissions_ca.cert.pem
└── governance.xml
```

### Key Rotation

```bash
# Manual key rotation
ros2 service call /lego_mcp/security/rotate_keys std_srvs/srv/Trigger

# Automated rotation (cron)
0 0 1 * * /path/to/generate_keystore.sh /path/to/keystore
```

---

## Access Control (RBAC)

### Predefined Roles

| Role | Permissions | Use Case |
|------|------------|----------|
| operator | VIEW_STATUS, START_JOB, PAUSE_JOB | Production floor |
| engineer | + MODIFY_CONFIG, CALIBRATE | Engineering |
| maintenance | + MAINTENANCE_MODE, DIAGNOSTICS | Maintenance |
| security_admin | SECURITY_SETTINGS, AUDIT_LOG | Security |
| system_admin | ALL_PERMISSIONS | Administration |

### User Management

```python
from lego_mcp_security import AccessControlManager

controller = AccessControlManager()
controller.create_user('john_doe', ['operator'])
controller.grant_role('john_doe', 'maintenance')  # Add role
controller.revoke_role('john_doe', 'operator')    # Remove role
```

---

## Audit Trail

### Tamper-Evident Logging

All security events are logged with cryptographic hash chains:

```
Event N:
  - Timestamp: 2024-01-15T10:30:00Z
  - Type: AUTHENTICATION_FAILURE
  - Previous Hash: 7a8b9c...
  - Event Hash: SHA256(previous + data) = 4d5e6f...
```

### Query Audit Log

```bash
# View recent events
ros2 service call /lego_mcp/security/audit/export std_srvs/srv/Trigger

# Verify chain integrity
ros2 service call /lego_mcp/security/audit/verify_chain std_srvs/srv/Trigger
```

---

## Intrusion Detection

### Detection Types

- **Unauthorized Node**: Unknown node joins network
- **Topic Flooding**: Excessive message rate
- **Message Tampering**: Invalid signatures
- **Clock Skew**: Time synchronization issues
- **Unusual Traffic**: Anomaly detection

### Configuration

```yaml
intrusion_detection:
  enabled: true
  detectors:
    unauthorized_node:
      action: ALERT_AND_BLOCK
    topic_flooding:
      threshold_msgs_per_sec: 1000
      action: THROTTLE
```

### Alert Handling

```bash
# Subscribe to alerts
ros2 topic echo /lego_mcp/security/intrusion_alerts
```

---

## Compliance

### IEC 62443 Compliance Checklist

- [x] Security zones defined
- [x] Access control implemented
- [x] Audit logging enabled
- [x] Encryption for DDS traffic
- [x] Authentication for all nodes
- [x] Intrusion detection
- [x] Key management procedures

### Compliance Verification

```bash
# Run compliance check
ros2 run lego_mcp_security compliance_checker

# Generate compliance report
ros2 service call /lego_mcp/security/compliance/report std_srvs/srv/Trigger
```

---

## Troubleshooting

### Common Issues

**Nodes fail to start with security enabled:**
```bash
# Check keystore permissions
ls -la $ROS_SECURITY_KEYSTORE

# Verify environment variables
echo $ROS_SECURITY_ENABLE
echo $ROS_SECURITY_STRATEGY
```

**Authentication failures:**
```bash
# Check certificate validity
openssl x509 -in keystore/enclaves/node/cert.pem -text -noout

# Regenerate keys
./scripts/generate_keystore.sh
```

**Permission denied errors:**
```bash
# Check permissions.xml for the node
cat keystore/enclaves/node/permissions.xml

# Verify governance.xml
cat keystore/governance.xml
```

---

## References

- [ROS2 Security Documentation](https://docs.ros.org/en/humble/Tutorials/Security.html)
- [IEC 62443 Standard](https://www.isa.org/standards-and-publications/isa-standards/isa-iec-62443-series-of-standards)
- [NIST 800-82](https://csrc.nist.gov/publications/detail/sp/800-82/rev-2/final)
