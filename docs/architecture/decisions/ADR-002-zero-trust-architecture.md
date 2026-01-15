# ADR-002: Zero-Trust Architecture

## Status

Accepted

## Date

2026-01-10

## Context

Traditional perimeter-based security ("castle and moat") is insufficient for modern manufacturing systems because:

1. **Lateral Movement Risk**: Once inside the network, attackers can move freely
2. **Cloud/Edge Hybrid**: System spans on-premise, cloud, and edge devices
3. **Supply Chain Attacks**: Third-party integrations expand attack surface
4. **Insider Threats**: Trusted users may be compromised
5. **OT/IT Convergence**: Manufacturing systems now connected to IT networks

DoD requirements (CMMC, NIST 800-207) mandate zero-trust principles:
- Never trust, always verify
- Assume breach
- Verify explicitly
- Least privilege access
- Micro-segmentation

## Decision

We will implement **Zero-Trust Architecture** following NIST SP 800-207 guidelines:

### 1. Identity Verification

- **SPIFFE/SPIRE** for workload identity
- **mTLS** for all service-to-service communication
- **Certificate-based authentication** for users and devices
- **Multi-factor authentication** where applicable

### 2. Device Posture Assessment

- Verify OS patch level
- Check security agent status
- Validate disk encryption
- Assess network context (VPN, location)

### 3. Continuous Authorization

- Re-evaluate trust on every request
- Dynamic trust score based on context
- Session tokens with short expiration
- Risk-based access decisions

### 4. Micro-Segmentation

- Network policies per workload
- Equipment zones with explicit allow rules
- East-west traffic inspection
- Data classification-based policies

### 5. Policy Decision Point (PDP)

- Centralized policy engine (OPA/Rego)
- Real-time policy evaluation
- Audit trail of all decisions

### Implementation

- `dashboard/services/security/zero_trust.py` - Gateway and policy engine
- Kubernetes NetworkPolicies for micro-segmentation
- Istio service mesh for mTLS

## Consequences

### Positive

- **Defense in Depth**: Multiple verification layers
- **Reduced Blast Radius**: Compromised component can't access everything
- **Audit Trail**: All access decisions logged
- **Compliance**: Meets NIST 800-207, CMMC requirements
- **Cloud-Ready**: Works in hybrid environments

### Negative

- **Complexity**: More moving parts to manage
- **Performance Overhead**: Every request requires policy check (~5-10ms)
- **Certificate Management**: Must manage PKI infrastructure
- **User Experience**: More authentication prompts

### Risks

- Policy misconfiguration could block legitimate access
- Certificate rotation complexity
- Performance impact on high-frequency operations

### Mitigations

- Extensive testing of policies before deployment
- Automated certificate rotation with short validity
- Policy caching for repeated access patterns
- Gradual rollout with monitoring

## Architecture Diagram

```
                    +----------------+
                    |   User/Device  |
                    +-------+--------+
                            |
                    +-------v--------+
                    | Policy Decision|
                    |     Point      |
                    +-------+--------+
                            |
            +---------------+---------------+
            |               |               |
    +-------v----+  +-------v----+  +-------v----+
    |   Zone A   |  |   Zone B   |  |   Zone C   |
    | (Equipment)|  |   (Data)   |  |   (AI/ML)  |
    +------------+  +------------+  +------------+
```

## Implementation Notes

```python
# Zero-Trust Gateway
gateway = ZeroTrustGateway()

# Authenticate with device context
identity = gateway.authenticate(
    credentials={"certificate": cert},
    method=AuthenticationMethod.MTLS,
)

# Authorize with continuous verification
authorized = gateway.authorize(
    identity=identity,
    resource_type=ResourceType.EQUIPMENT,
    resource_id="cnc-001",
    access_level=AccessLevel.CONTROL,
)

# Trust score calculation
score = gateway.calculate_trust_score(
    device_posture={"os_patched": True, "vpn": True},
    user_behavior={"failed_logins": 0},
)
```

## References

- [NIST SP 800-207: Zero Trust Architecture](https://csrc.nist.gov/publications/detail/sp/800-207/final)
- [SPIFFE: Secure Production Identity Framework](https://spiffe.io/)
- [Open Policy Agent](https://www.openpolicyagent.org/)
- [IEC 62443: Industrial Cybersecurity](https://www.isa.org/standards-and-publications/isa-standards/isa-iec-62443-series-of-standards)
