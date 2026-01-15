# LEGO MCP v8.0 API Reference

## Overview

This document describes the new API endpoints introduced in v8.0 (Command & Control Edition).
All endpoints follow REST conventions and return JSON responses.

**Base URL**: `http://localhost:5000/api/v8`

**Authentication**: Bearer token or mTLS certificate (see Security section)

---

## Command Center APIs

### System Health

#### GET /api/v8/command-center/status

Get complete system status including all services, ROS2 nodes, and equipment.

**Response:**
```json
{
  "success": true,
  "data": {
    "overall_status": "healthy",
    "timestamp": "2026-01-15T10:30:00Z",
    "services": {
      "dashboard": {"status": "healthy", "latency_ms": 12},
      "mcp_server": {"status": "healthy", "latency_ms": 8},
      "slicer": {"status": "healthy", "latency_ms": 15},
      "ros2_bridge": {"status": "healthy", "nodes": 32}
    },
    "equipment": {
      "total": 12,
      "online": 11,
      "offline": 1,
      "maintenance": 0
    },
    "metrics": {
      "uptime_hours": 720.5,
      "requests_per_minute": 450,
      "error_rate": 0.001
    }
  }
}
```

#### GET /api/v8/command-center/kpis

Get aggregated KPIs across all systems.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `category` | string | Filter by category: production, quality, equipment, sustainability |
| `period` | string | Time period: hour, day, week, month |
| `equipment_id` | string | Filter by specific equipment |

**Response:**
```json
{
  "success": true,
  "data": {
    "period": "day",
    "kpis": {
      "oee": {"value": 87.5, "target": 85.0, "unit": "%", "trend": "up"},
      "fpy": {"value": 98.2, "target": 95.0, "unit": "%", "trend": "stable"},
      "throughput": {"value": 1250, "target": 1000, "unit": "parts/hr", "trend": "up"},
      "scrap_rate": {"value": 0.8, "target": 1.0, "unit": "%", "trend": "down"},
      "energy_per_unit": {"value": 0.45, "target": 0.5, "unit": "kWh", "trend": "down"},
      "mtbf": {"value": 720, "target": 500, "unit": "hours", "trend": "up"}
    },
    "charts": {
      "oee_trend": [85.0, 86.2, 87.1, 87.5],
      "quality_distribution": {"pass": 982, "rework": 15, "scrap": 3}
    }
  }
}
```

### Alert Management

#### GET /api/v8/command-center/alerts

List all alerts with filtering.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `status` | string | Filter: active, acknowledged, resolved |
| `severity` | string | Filter: critical, high, medium, low |
| `source` | string | Filter by source system |
| `limit` | int | Max results (default 50) |
| `offset` | int | Pagination offset |

**Response:**
```json
{
  "success": true,
  "data": {
    "total": 45,
    "alerts": [
      {
        "id": "ALT-2026-0115-001",
        "title": "Temperature Warning",
        "message": "Extruder temperature above threshold",
        "severity": "high",
        "status": "active",
        "source": "equipment",
        "entity_type": "extruder",
        "entity_id": "extruder-001",
        "created_at": "2026-01-15T10:25:00Z",
        "acknowledged_at": null,
        "acknowledged_by": null
      }
    ],
    "summary": {
      "by_severity": {"critical": 2, "high": 8, "medium": 20, "low": 15},
      "by_status": {"active": 10, "acknowledged": 15, "resolved": 20}
    }
  }
}
```

#### POST /api/v8/command-center/alerts

Create a new alert.

**Request Body:**
```json
{
  "title": "Custom Alert",
  "message": "Alert details here",
  "severity": "medium",
  "source": "custom",
  "entity_type": "process",
  "entity_id": "process-001"
}
```

#### POST /api/v8/command-center/alerts/{alert_id}/acknowledge

Acknowledge an alert.

**Request Body:**
```json
{
  "acknowledged_by": "operator@plant-1",
  "notes": "Investigating issue"
}
```

#### POST /api/v8/command-center/alerts/{alert_id}/resolve

Resolve an alert.

**Request Body:**
```json
{
  "resolved_by": "engineer@plant-1",
  "resolution": "Replaced faulty sensor",
  "root_cause": "Sensor degradation"
}
```

---

## Action Management APIs

### GET /api/v8/actions

List pending and recent actions.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `status` | string | Filter: pending, approved, executing, completed, rejected |
| `category` | string | Filter: equipment, quality, scheduling, maintenance |
| `requires_approval` | bool | Filter by approval requirement |

**Response:**
```json
{
  "success": true,
  "data": {
    "pending": [
      {
        "id": "ACT-001",
        "title": "Adjust Feed Rate",
        "description": "Reduce feed rate by 10% on CNC-001",
        "category": "equipment",
        "status": "pending",
        "requires_approval": true,
        "risk_level": "medium",
        "estimated_impact": {"duration_minutes": 5, "production_loss": 2},
        "created_at": "2026-01-15T10:30:00Z",
        "source": {
          "type": "ai_insight",
          "model": "quality_predictor",
          "confidence": 0.92
        }
      }
    ],
    "recent": []
  }
}
```

### POST /api/v8/actions

Create a new action.

**Request Body:**
```json
{
  "title": "Schedule Maintenance",
  "description": "Schedule preventive maintenance for Robot-Arm-01",
  "category": "maintenance",
  "target_type": "equipment",
  "target_id": "robot-arm-01",
  "requires_approval": true,
  "parameters": {
    "maintenance_type": "preventive",
    "estimated_duration": 120,
    "parts_required": ["servo-motor-001", "belt-002"]
  }
}
```

### POST /api/v8/actions/{action_id}/approve

Approve an action.

**Request Body:**
```json
{
  "approved_by": "supervisor@plant-1",
  "notes": "Approved for next shift"
}
```

### POST /api/v8/actions/{action_id}/reject

Reject an action.

**Request Body:**
```json
{
  "rejected_by": "supervisor@plant-1",
  "reason": "Not required at this time"
}
```

### POST /api/v8/actions/{action_id}/execute

Execute an approved action.

**Response:**
```json
{
  "success": true,
  "data": {
    "execution_id": "EXEC-001",
    "status": "executing",
    "started_at": "2026-01-15T10:35:00Z",
    "estimated_completion": "2026-01-15T10:40:00Z"
  }
}
```

---

## Co-Simulation APIs

### GET /api/v8/cosim/scenarios

List simulation scenarios.

**Response:**
```json
{
  "success": true,
  "data": {
    "scenarios": [
      {
        "id": "SCEN-001",
        "name": "Demand Surge Analysis",
        "description": "Simulate 2x demand increase",
        "type": "what-if",
        "status": "completed",
        "created_at": "2026-01-15T09:00:00Z",
        "duration_seconds": 45
      }
    ]
  }
}
```

### POST /api/v8/cosim/scenarios

Create a new simulation scenario.

**Request Body:**
```json
{
  "name": "New Product Introduction",
  "description": "Simulate adding new brick type to production",
  "type": "what-if",
  "parameters": {
    "new_product": {
      "type": "4x4_brick",
      "volume": 10000,
      "start_date": "2026-02-01"
    },
    "simulation": {
      "duration_days": 30,
      "monte_carlo_iterations": 1000
    }
  }
}
```

### POST /api/v8/cosim/run

Run a simulation.

**Request Body:**
```json
{
  "scenario_id": "SCEN-001",
  "mode": "accelerated",
  "speed_factor": 100,
  "include_pinn": true,
  "include_des": true
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "run_id": "RUN-001",
    "status": "running",
    "progress": 0,
    "started_at": "2026-01-15T10:45:00Z"
  }
}
```

### POST /api/v8/cosim/compare

Compare multiple scenarios.

**Request Body:**
```json
{
  "scenario_ids": ["SCEN-001", "SCEN-002", "SCEN-003"],
  "metrics": ["throughput", "oee", "cost_per_unit", "lead_time"]
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "comparison": {
      "SCEN-001": {
        "throughput": 1250,
        "oee": 87.5,
        "cost_per_unit": 0.45,
        "lead_time": 24
      },
      "SCEN-002": {
        "throughput": 1400,
        "oee": 82.3,
        "cost_per_unit": 0.52,
        "lead_time": 20
      }
    },
    "recommendation": "SCEN-001",
    "reasoning": "Best balance of OEE and cost efficiency"
  }
}
```

---

## Security APIs

### POST /api/v8/security/authenticate

Authenticate using Zero-Trust.

**Request Body:**
```json
{
  "method": "certificate",
  "certificate": "-----BEGIN CERTIFICATE-----...",
  "device_posture": {
    "os_version": "Ubuntu 22.04",
    "security_agent": true,
    "disk_encrypted": true
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "token": "eyJhbGciOiJFUzM4NCIsInR5cCI6IkpXVCJ9...",
    "expires_at": "2026-01-15T14:45:00Z",
    "trust_score": 0.95,
    "trust_level": "high",
    "permissions": ["read:all", "write:equipment", "execute:actions"]
  }
}
```

### GET /api/v8/security/audit

Get audit log entries.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `from` | datetime | Start time |
| `to` | datetime | End time |
| `actor` | string | Filter by actor |
| `action` | string | Filter by action type |
| `resource` | string | Filter by resource |

**Response:**
```json
{
  "success": true,
  "data": {
    "entries": [
      {
        "id": "AUD-001",
        "timestamp": "2026-01-15T10:30:00Z",
        "actor": "operator@plant-1",
        "action": "equipment_control",
        "resource": "robot-arm-01",
        "result": "success",
        "ip_address": "192.168.1.100",
        "hash": "sha256:abc123...",
        "previous_hash": "sha256:xyz789..."
      }
    ],
    "chain_status": {
      "valid": true,
      "verified_at": "2026-01-15T10:35:00Z"
    }
  }
}
```

### POST /api/v8/security/verify-chain

Verify audit chain integrity.

**Response:**
```json
{
  "success": true,
  "data": {
    "valid": true,
    "entries_verified": 15420,
    "first_entry": "2026-01-01T00:00:00Z",
    "last_entry": "2026-01-15T10:35:00Z",
    "verification_time_ms": 1250
  }
}
```

---

## Compliance APIs

### GET /api/v8/compliance/cmmc/status

Get CMMC compliance status.

**Response:**
```json
{
  "success": true,
  "data": {
    "target_level": 3,
    "current_score": 0.96,
    "practices": {
      "total": 130,
      "fully_implemented": 125,
      "partially_implemented": 3,
      "not_implemented": 2
    },
    "domains": {
      "AC": {"score": 1.0, "practices": 22},
      "AU": {"score": 0.95, "practices": 9},
      "SC": {"score": 0.92, "practices": 16}
    },
    "next_assessment": "2026-03-01",
    "poam_items": 5
  }
}
```

### GET /api/v8/compliance/sbom

Get Software Bill of Materials.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `format` | string | cyclonedx, spdx |

**Response (CycloneDX):**
```json
{
  "bomFormat": "CycloneDX",
  "specVersion": "1.5",
  "serialNumber": "urn:uuid:...",
  "version": 1,
  "metadata": {
    "timestamp": "2026-01-15T10:00:00Z",
    "tools": [{"vendor": "LEGO MCP", "name": "sbom-generator", "version": "8.0.0"}],
    "component": {
      "type": "application",
      "name": "lego-mcp",
      "version": "8.0.0"
    }
  },
  "components": [
    {
      "type": "library",
      "name": "flask",
      "version": "3.0.0",
      "purl": "pkg:pypi/flask@3.0.0",
      "licenses": [{"license": {"id": "BSD-3-Clause"}}]
    }
  ]
}
```

### POST /api/v8/compliance/cato/scan

Run continuous ATO compliance scan.

**Response:**
```json
{
  "success": true,
  "data": {
    "scan_id": "SCAN-001",
    "status": "completed",
    "started_at": "2026-01-15T10:40:00Z",
    "completed_at": "2026-01-15T10:42:00Z",
    "results": {
      "controls_checked": 320,
      "passed": 318,
      "failed": 2,
      "findings": [
        {
          "control": "AU.2.042",
          "severity": "low",
          "message": "Log retention below recommended 1 year",
          "remediation": "Increase log retention period"
        }
      ]
    }
  }
}
```

---

## WebSocket Events

Connect to WebSocket at `ws://localhost:5000/socket.io/`

### Events (Server to Client)

| Event | Description |
|-------|-------------|
| `command_center:status` | System status update (every 5s) |
| `command_center:kpi` | KPI update (every 5s) |
| `command_center:alert` | New alert created |
| `action:pending` | New action awaiting approval |
| `action:executed` | Action execution completed |
| `cosim:progress` | Simulation progress update |
| `cosim:result` | Simulation completed |
| `equipment:state` | Equipment state change |
| `quality:result` | Quality inspection result |
| `security:anomaly` | Security anomaly detected |

### Events (Client to Server)

| Event | Description |
|-------|-------------|
| `subscribe:equipment` | Subscribe to equipment updates |
| `subscribe:alerts` | Subscribe to alert updates |
| `subscribe:kpis` | Subscribe to KPI updates |
| `action:approve` | Approve pending action |
| `action:reject` | Reject pending action |

---

## Error Responses

All errors follow this format:

```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "severity",
      "issue": "Must be one of: critical, high, medium, low"
    }
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid request parameters |
| `UNAUTHORIZED` | 401 | Authentication required |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `CONFLICT` | 409 | Resource state conflict |
| `RATE_LIMITED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Server error |

---

## Rate Limits

| Endpoint Type | Limit |
|--------------|-------|
| Read endpoints | 1000 requests/minute |
| Write endpoints | 100 requests/minute |
| Action execution | 10 requests/minute |
| Simulation runs | 5 requests/minute |

---

## Security

### Post-Quantum Cryptography

All API communications support hybrid classical/PQ encryption:

- **Key Encapsulation**: ML-KEM-768 (NIST FIPS 203)
- **Digital Signatures**: ML-DSA-65 (NIST FIPS 204)
- **Hash-Based Signatures**: SLH-DSA-SHA2-128s (NIST FIPS 205)

### Zero-Trust Requirements

1. All requests require authentication
2. Device posture is evaluated on each request
3. Trust score determines access level
4. Sessions require periodic revalidation

### Audit Trail

All API calls are logged to tamper-evident audit chain:
- SHA-256 hash chain for integrity
- HSM-sealed daily checkpoints
- SIEM integration for correlation

---

*LEGO MCP v8.0 - DoD/ONR-Class Manufacturing Excellence*
