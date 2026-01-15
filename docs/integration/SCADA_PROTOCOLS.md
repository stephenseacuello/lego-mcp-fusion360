# SCADA/MES Protocol Integration Guide

## LEGO MCP Fusion 360 - Industry 4.0/5.0 Architecture

This guide covers industrial protocol integration for SCADA/MES connectivity.

---

## Overview

The LEGO MCP system supports multiple industrial protocols for integration with SCADA systems, MES platforms, and enterprise applications.

### Supported Protocols

| Protocol | Standard | Use Case |
|----------|----------|----------|
| OPC UA | IEC 62541, OPC 40501 | Industrial automation, CNC systems |
| MTConnect | ANSI/MTC1.4-2018 | CNC data streaming |
| Sparkplug B | Eclipse Sparkplug | IIoT MQTT with birth/death |
| MQTT | ISO/IEC 20922 | Lightweight IoT messaging |
| Modbus | IEC 61158 | Legacy PLC communication |

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SCADA/MES Integration Layer                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Enterprise / SCADA / MES                                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │  SAP     │  │ Ignition │  │ Historian│  │ Custom   │                │
│  │  ERP     │  │  SCADA   │  │  Server  │  │   HMI    │                │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘                │
│       │             │             │             │                        │
│       │    OPC UA   │  MTConnect  │ Sparkplug B │   MQTT                │
│       │             │             │             │                        │
│  ═════╪═════════════╪═════════════╪═════════════╪════════════════════   │
│                                                                          │
│                Protocol Adapters (dashboard/services/edge/)              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │ OPC UA   │  │MTConnect │  │Sparkplug │  │  MQTT    │                │
│  │ Server   │  │  Agent   │  │  Edge    │  │ Adapter  │                │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘                │
│       │             │             │             │                        │
│  ═════╪═════════════╪═════════════╪═════════════╪════════════════════   │
│                                                                          │
│                    ROS2 DDS / Rosbridge                                  │
│                                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │Equipment │  │  Safety  │  │Orchestr- │  │   AGV    │                │
│  │  Nodes   │  │   Node   │  │   ator   │  │  Fleet   │                │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## OPC UA Integration

### OPC 40501 CNC Information Model

The LEGO MCP implements OPC 40501 for CNC system data:

```python
from dashboard.services.edge.protocol_adapters import OPCUAAdapter
from dashboard.services.edge.protocol_adapters.opcua_cnc_model import (
    CncInterfaceModel,
    create_grbl_cnc_model,
)

# Create CNC information model
cnc = create_grbl_cnc_model()

# Create OPC UA server
server = OPCUAAdapter(
    endpoint="opc.tcp://0.0.0.0:4840",
    namespace="http://legomcp.dev/cnc",
)

# Register CNC model
server.register_cnc_model(cnc)

# Start server
await server.start()
```

### OPC UA Address Space

```
Objects
└── CncInterface (ns=2;s=grbl_cnc)
    ├── CncAxisList
    │   ├── X-Axis
    │   │   ├── ActPos (Double)
    │   │   ├── CmdPos (Double)
    │   │   ├── ActVel (Double)
    │   │   └── IsHomed (Boolean)
    │   ├── Y-Axis
    │   └── Z-Axis
    ├── CncSpindleList
    │   └── MainSpindle
    │       ├── ActSpeed (Double)
    │       ├── IsRunning (Boolean)
    │       └── Load (Double)
    ├── CncChannelList
    │   └── MainChannel
    │       ├── OperatingMode (Int32)
    │       ├── ExecutionState (Int32)
    │       └── ProgramName (String)
    └── CncAlarmList
```

### Client Connection

```python
# Python client (using asyncua)
from asyncua import Client

async with Client("opc.tcp://localhost:4840") as client:
    # Read axis position
    x_pos = await client.get_node("ns=2;s=grbl_cnc/Axis/X/ActPos").read_value()
    print(f"X Position: {x_pos}")

    # Subscribe to changes
    handler = DataChangeHandler()
    subscription = await client.create_subscription(100, handler)
    await subscription.subscribe_data_change(
        client.get_node("ns=2;s=grbl_cnc/Axis/X/ActPos")
    )
```

---

## MTConnect Integration

### MTConnect Agent

```python
from dashboard.services.edge.protocol_adapters import (
    MTConnectAgent,
    MTConnectAdapter,
    create_cnc_device,
)

# Create device
device = create_cnc_device("grbl_cnc")

# Create agent (HTTP server)
agent = MTConnectAgent(
    port=5000,
    device=device,
)

# Create adapter (data source)
adapter = MTConnectAdapter(
    agent_host="localhost",
    agent_port=5000,
)

# Start services
await agent.start()
await adapter.start()

# Update data
adapter.update_data_item("execution", "ACTIVE")
adapter.update_data_item("Xact", 100.5)
```

### MTConnect Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /probe` | Device capability (MTConnectDevices) |
| `GET /current` | Current state (MTConnectStreams) |
| `GET /sample?from=N` | Historical data |
| `GET /sample?count=N` | Last N samples |

### Example Response (Current)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<MTConnectStreams>
  <Streams>
    <DeviceStream name="grbl_cnc">
      <ComponentStream component="Controller">
        <Events>
          <Execution dataItemId="execution">ACTIVE</Execution>
          <ControllerMode dataItemId="mode">AUTOMATIC</ControllerMode>
        </Events>
      </ComponentStream>
      <ComponentStream component="LinearAxis" name="X">
        <Samples>
          <Position dataItemId="Xact" timestamp="...">100.500</Position>
        </Samples>
      </ComponentStream>
    </DeviceStream>
  </Streams>
</MTConnectStreams>
```

---

## Sparkplug B Integration

### Edge Node Setup

```python
from dashboard.services.edge.protocol_adapters import (
    SparkplugBEdgeNode,
    SparkplugMetric,
    MetricDataType,
)

# Create edge node
edge = SparkplugBEdgeNode(
    group_id="lego_mcp",
    edge_node_id="factory_floor",
    mqtt_host="mqtt.local",
    mqtt_port=1883,
)

# Add device
edge.add_device(
    device_id="grbl_cnc",
    metrics=[
        SparkplugMetric(
            name="position/x",
            data_type=MetricDataType.DOUBLE,
            value=0.0,
        ),
        SparkplugMetric(
            name="status",
            data_type=MetricDataType.STRING,
            value="idle",
        ),
    ],
)

# Connect (publishes NBIRTH)
await edge.connect()

# Update metrics (publishes DDATA)
await edge.update_metric("grbl_cnc", "position/x", 100.5)
await edge.update_metric("grbl_cnc", "status", "running")
```

### Sparkplug B Topics

```
spBv1.0/lego_mcp/NBIRTH/factory_floor          # Node birth
spBv1.0/lego_mcp/NDEATH/factory_floor          # Node death
spBv1.0/lego_mcp/DBIRTH/factory_floor/grbl_cnc # Device birth
spBv1.0/lego_mcp/DDEATH/factory_floor/grbl_cnc # Device death
spBv1.0/lego_mcp/DDATA/factory_floor/grbl_cnc  # Device data
spBv1.0/lego_mcp/DCMD/factory_floor/grbl_cnc   # Device command
```

### Host Application

```python
from dashboard.services.edge.protocol_adapters import SparkplugBHostApplication

# Create host application
host = SparkplugBHostApplication(
    host_id="scada_master",
    mqtt_host="mqtt.local",
)

# Register callback for device data
@host.on_device_data
def handle_data(group_id, edge_node_id, device_id, metrics):
    for metric in metrics:
        print(f"{device_id}/{metric.name}: {metric.value}")

# Connect
await host.connect()
```

---

## MQTT Integration

### Basic MQTT Adapter

```python
from dashboard.services.edge.protocol_adapters import MQTTAdapter

adapter = MQTTAdapter(
    host="mqtt.local",
    port=1883,
    client_id="lego_mcp_bridge",
)

# Subscribe to topics
@adapter.subscribe("lego_mcp/+/status")
def handle_status(topic, payload):
    print(f"Status update: {topic} = {payload}")

# Publish equipment data
await adapter.publish(
    "lego_mcp/grbl_cnc/position",
    {"x": 100.5, "y": 50.2, "z": 10.0},
)

# Connect
await adapter.connect()
```

### Topic Structure

```
lego_mcp/
├── equipment/
│   ├── grbl_cnc/
│   │   ├── status          # Equipment status
│   │   ├── position        # Current position
│   │   ├── telemetry       # Detailed telemetry
│   │   └── commands        # Command input
│   ├── formlabs/
│   └── bambu/
├── safety/
│   ├── estop               # E-stop status
│   └── alarms              # Active alarms
├── production/
│   ├── jobs                # Job status
│   └── schedule            # Production schedule
└── analytics/
    └── oee                 # OEE metrics
```

---

## Modbus Integration

### Modbus TCP/RTU Adapter

```python
from dashboard.services.edge.protocol_adapters import ModbusAdapter

adapter = ModbusAdapter(
    mode="tcp",  # or "rtu"
    host="192.168.1.100",
    port=502,
)

# Read holding registers
registers = await adapter.read_holding_registers(
    address=0,
    count=10,
)

# Write single register
await adapter.write_register(address=100, value=1)

# Map to equipment data
adapter.register_mapping({
    0: "x_position",
    1: "y_position",
    2: "z_position",
    10: "spindle_speed",
})
```

---

## Configuration

### protocol_adapters.yaml

```yaml
# OPC UA Server
opcua:
  enabled: true
  endpoint: "opc.tcp://0.0.0.0:4840"
  namespace: "http://legomcp.dev"
  security_policy: "Basic256Sha256"
  certificate_path: "/etc/lego_mcp/certs/opcua.pem"

# MTConnect Agent
mtconnect:
  enabled: true
  port: 5000
  device_name: "lego_mcp_factory"
  buffer_size: 100000

# Sparkplug B
sparkplug:
  enabled: true
  group_id: "lego_mcp"
  edge_node_id: "factory_floor"
  mqtt:
    host: "mqtt.local"
    port: 8883
    use_tls: true
    ca_cert: "/etc/lego_mcp/certs/ca.pem"

# MQTT
mqtt:
  enabled: true
  host: "mqtt.local"
  port: 1883
  qos: 1
  retain: false

# Modbus
modbus:
  enabled: false
  mode: "tcp"
  host: "192.168.1.100"
  port: 502
```

---

## Security Considerations

### OPC UA Security

- Use `Basic256Sha256` security policy minimum
- Enable user authentication
- Use certificates signed by trusted CA

### MQTT Security

- Use TLS (port 8883)
- Require client certificates
- Use ACLs to restrict topic access

### Network Segmentation

Follow IEC 62443 zones:

```
Zone 3 (MES) ──► DMZ ──► Zone 4 (Enterprise)
       │
       └──► Protocol Adapters (in DMZ)
```

---

## Troubleshooting

### OPC UA Connection Issues

```bash
# Test OPC UA connection
python -c "
from asyncua import Client
import asyncio
async def test():
    async with Client('opc.tcp://localhost:4840') as c:
        print('Connected!')
asyncio.run(test())
"
```

### MTConnect Not Responding

```bash
# Check agent status
curl http://localhost:5000/probe

# Check adapter connection
curl http://localhost:5000/current
```

### Sparkplug Birth Not Received

1. Check MQTT connection
2. Verify topic permissions
3. Check birth certificate payload
4. Review MQTT broker logs

---

## References

- [OPC UA Specification](https://opcfoundation.org/developer-tools/specifications-unified-architecture)
- [OPC 40501 CNC Systems](https://opcfoundation.org/markets-collaboration/cnc/)
- [MTConnect Standard](https://www.mtconnect.org/standard)
- [Eclipse Sparkplug](https://sparkplug.eclipse.org/)
- [MQTT 5.0 Specification](https://docs.oasis-open.org/mqtt/mqtt/v5.0/mqtt-v5.0.html)
