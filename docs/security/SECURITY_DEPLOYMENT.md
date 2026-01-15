# LEGO MCP Security Deployment Guide

## Overview

This document covers security-specific deployment configurations for LEGO MCP Manufacturing System, ensuring compliance with:

- **NIST SP 800-171 Rev 2** - Protecting CUI in Nonfederal Systems
- **CMMC Level 3** - Cybersecurity Maturity Model Certification
- **IEC 62443** - Industrial Automation and Control Systems Security
- **NIST SP 800-207** - Zero Trust Architecture

---

## Security Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            EXTERNAL BOUNDARY                                 │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         WAF + DDoS Protection                          │  │
│  │                    (AWS Shield / Cloudflare / Akamai)                  │  │
│  └────────────────────────────────┬──────────────────────────────────────┘  │
│                                   │                                          │
│  ┌────────────────────────────────▼──────────────────────────────────────┐  │
│  │                          API Gateway                                   │  │
│  │            (Rate Limiting, OAuth 2.0, API Key Validation)              │  │
│  └────────────────────────────────┬──────────────────────────────────────┘  │
└───────────────────────────────────┼──────────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼──────────────────────────────────────────┐
│                              ZERO TRUST BOUNDARY                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                    Identity Provider (OIDC)                              │ │
│  │         (Azure AD / Okta / Keycloak with MFA + Device Trust)            │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                     Service Mesh (mTLS)                                  │ │
│  │              (Linkerd/Istio with SPIFFE/SPIRE)                          │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                    Policy Decision Point                                 │ │
│  │           (OPA/Gatekeeper + ABAC/RBAC Policies)                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼──────────────────────────────────────────┐
│                            DATA PROTECTION BOUNDARY                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                 Encryption at Rest (AES-256-GCM)                         │ │
│  │                Encryption in Transit (TLS 1.3 + mTLS)                    │ │
│  │                 Key Management (HSM-backed + Vault)                      │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## Pre-Deployment Security Checklist

### Infrastructure Security

- [ ] Kubernetes cluster hardened per CIS Benchmark
- [ ] Pod Security Standards enforced (restricted)
- [ ] Network policies default-deny configured
- [ ] Node security groups locked down
- [ ] etcd encryption enabled
- [ ] API server audit logging enabled
- [ ] Admission controllers configured (OPA Gatekeeper)

### Identity & Access

- [ ] OIDC provider configured with MFA
- [ ] Service accounts use Workload Identity
- [ ] RBAC roles follow least privilege
- [ ] Privileged containers prohibited
- [ ] No default service account tokens

### Secrets Management

- [ ] HashiCorp Vault deployed in HA mode
- [ ] HSM configured for key storage
- [ ] Secrets never in Git (sealed-secrets or external-secrets)
- [ ] Rotation policies defined
- [ ] Break-glass procedures documented

### Container Security

- [ ] Images from trusted registries only
- [ ] Image signing enforced (cosign)
- [ ] Vulnerability scanning in CI/CD
- [ ] No root containers
- [ ] Read-only root filesystem

---

## Zero Trust Configuration

### SPIFFE/SPIRE Setup

```bash
# Install SPIRE server
helm repo add spiffe https://spiffe.github.io/helm-charts
helm install spire-server spiffe/spire-server \
  --namespace spire \
  --set trustDomain=lego-mcp.io

# Install SPIRE agent (DaemonSet)
helm install spire-agent spiffe/spire-agent \
  --namespace spire \
  --set server.address=spire-server.spire.svc.cluster.local:8081

# Register workloads
kubectl exec -n spire spire-server-0 -- \
  spire-server entry create \
    -spiffeID spiffe://lego-mcp.io/dashboard \
    -parentID spiffe://lego-mcp.io/spire/agent/k8s_psat/lego-mcp-prod \
    -selector k8s:ns:lego-mcp-prod \
    -selector k8s:sa:dashboard
```

### OPA Gatekeeper Policies

```yaml
# Require container image signing
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: RequireImageSignature
metadata:
  name: require-cosign-signature
spec:
  match:
    kinds:
    - apiGroups: [""]
      kinds: ["Pod"]
    namespaces: ["lego-mcp-prod"]
  parameters:
    publicKeys:
    - keyless:
        identities:
        - issuer: "https://token.actions.githubusercontent.com"
          subject: "https://github.com/lego-mcp/lego-mcp/*"
---
# Deny privileged containers
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sPSPPrivilegedContainer
metadata:
  name: deny-privileged-containers
spec:
  match:
    kinds:
    - apiGroups: [""]
      kinds: ["Pod"]
    excludedNamespaces: ["kube-system"]
---
# Require resource limits
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sContainerLimits
metadata:
  name: require-container-limits
spec:
  match:
    kinds:
    - apiGroups: [""]
      kinds: ["Pod"]
  parameters:
    cpu: "4"
    memory: "8Gi"
```

### Anomaly Detection Configuration

```python
# Configure security anomaly detection
from dashboard.services.security.anomaly_detection import (
    SecurityAnomalyDetector,
    AnomalyConfig,
)

config = AnomalyConfig(
    # Geographic analysis
    max_velocity_km_h=900.0,  # Impossible travel threshold
    trusted_countries=["US", "CA", "GB", "DE"],

    # Temporal analysis
    working_hours_start=6,
    working_hours_end=22,
    unusual_hour_threshold=0.1,  # 10% normal activity

    # Behavioral analysis
    baseline_window_days=30,
    deviation_threshold=3.0,  # Standard deviations
    privilege_escalation_sensitivity="high",

    # Volumetric analysis
    requests_per_minute_threshold=1000,
    data_exfil_threshold_mb=100,
)

detector = SecurityAnomalyDetector(config)
```

---

## Cryptographic Controls

### Post-Quantum Cryptography

```python
# Configure post-quantum algorithms (NIST FIPS 203/204/205)
from dashboard.services.security.pq_crypto import (
    PQCryptoProvider,
    EncryptionMode,
)

pq = PQCryptoProvider(
    # ML-KEM for key encapsulation
    kem_algorithm="ML-KEM-768",  # FIPS 203

    # ML-DSA for signatures
    signature_algorithm="ML-DSA-65",  # FIPS 204

    # Hybrid mode for transition
    hybrid_mode=True,
    classical_fallback="ECDH-P384",
)
```

### HSM Integration

```yaml
# HSM Configuration (Thales Luna / AWS CloudHSM)
apiVersion: v1
kind: Secret
metadata:
  name: hsm-credentials
  namespace: lego-mcp-prod
type: Opaque
stringData:
  HSM_SLOT: "0"
  HSM_PIN: "vault:secret/data/lego-mcp/hsm#pin"
  HSM_LIBRARY: "/usr/lib/libCryptoki2.so"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dashboard
spec:
  template:
    spec:
      containers:
      - name: dashboard
        env:
        - name: HSM_ENABLED
          value: "true"
        - name: HSM_SLOT
          valueFrom:
            secretKeyRef:
              name: hsm-credentials
              key: HSM_SLOT
        volumeMounts:
        - name: hsm-socket
          mountPath: /var/run/hsm
      volumes:
      - name: hsm-socket
        hostPath:
          path: /var/run/hsm
          type: Socket
```

---

## Network Security

### Network Policies

```yaml
# Default deny all ingress/egress
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: lego-mcp-prod
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
---
# Allow DNS resolution
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-dns
  namespace: lego-mcp-prod
spec:
  podSelector: {}
  policyTypes:
  - Egress
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          kubernetes.io/metadata.name: kube-system
      podSelector:
        matchLabels:
          k8s-app: kube-dns
    ports:
    - protocol: UDP
      port: 53
---
# Dashboard to MCP Server
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: dashboard-to-mcp
  namespace: lego-mcp-prod
spec:
  podSelector:
    matchLabels:
      app: dashboard
  policyTypes:
  - Egress
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: mcp-server
    ports:
    - protocol: TCP
      port: 8080
---
# Allow ingress from load balancer
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-ingress
  namespace: lego-mcp-prod
spec:
  podSelector:
    matchLabels:
      app: dashboard
  policyTypes:
  - Ingress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          kubernetes.io/metadata.name: ingress-nginx
    ports:
    - protocol: TCP
      port: 5000
```

### TLS Configuration

```yaml
# TLS 1.3 only with secure cipher suites
apiVersion: v1
kind: ConfigMap
metadata:
  name: nginx-config
  namespace: ingress-nginx
data:
  ssl-protocols: "TLSv1.3"
  ssl-ciphers: "TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256:TLS_AES_128_GCM_SHA256"
  ssl-prefer-server-ciphers: "true"
  hsts: "true"
  hsts-max-age: "31536000"
  hsts-include-subdomains: "true"
  hsts-preload: "true"
```

---

## Audit & Compliance

### Audit Logging

```python
# Enable comprehensive audit logging
from dashboard.services.compliance import ComplianceAuditLogger

audit = ComplianceAuditLogger(
    # Destinations
    syslog_server="siem.lego-mcp.io:514",
    splunk_hec="https://splunk.lego-mcp.io:8088",

    # Retention
    retention_days=365 * 7,  # 7 years for CMMC

    # Signing
    sign_with_hsm=True,
    hsm_key_label="audit-signing-key",

    # Format
    format="CEF",  # Common Event Format
)
```

### Continuous Compliance Monitoring

```yaml
# CronJob for compliance checks
apiVersion: batch/v1
kind: CronJob
metadata:
  name: compliance-scan
  namespace: lego-mcp-prod
spec:
  schedule: "0 */6 * * *"  # Every 6 hours
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: compliance-scanner
            image: ghcr.io/lego-mcp/compliance-scanner:latest
            command:
            - python
            - -c
            - |
              from dashboard.services.compliance import NISTComplianceChecker
              from dashboard.services.compliance import CMMCAssessment

              # NIST 800-171 check
              nist = NISTComplianceChecker()
              nist_report = nist.run_assessment()

              # CMMC Level 3 check
              cmmc = CMMCAssessment(target_level=3)
              cmmc_report = cmmc.run_assessment()

              # Alert if non-compliant
              if nist_report.score < 100 or cmmc_report.score < 100:
                  raise Exception("Compliance drift detected!")
          restartPolicy: OnFailure
```

---

## Incident Response

### Security Event Detection

```python
# Configure SIEM integration
from dashboard.services.observability.siem_integration import (
    SIEMConnector,
    SIEMProvider,
)

siem = SIEMConnector(
    provider=SIEMProvider.SPLUNK,
    endpoint="https://splunk.lego-mcp.io:8088",
    token="vault:secret/data/lego-mcp/splunk#hec_token",

    # Event forwarding
    forward_security_events=True,
    forward_compliance_events=True,
    forward_audit_events=True,

    # Alert thresholds
    alert_on_failed_auth=True,
    failed_auth_threshold=5,
    alert_on_privilege_escalation=True,
)
```

### Automated Response

```yaml
# Falco rules for runtime security
apiVersion: falco.org/v1alpha1
kind: FalcoRule
metadata:
  name: lego-mcp-security-rules
spec:
  rules:
  - rule: Detect Shell in Container
    desc: Alert when shell is spawned in LEGO MCP container
    condition: >
      spawned_process and container and
      container.image.repository contains "lego-mcp" and
      proc.name in (bash, sh, zsh)
    output: >
      Shell spawned in LEGO MCP container
      (user=%user.name container=%container.id command=%proc.cmdline)
    priority: WARNING
    tags: [container, shell, lego-mcp]

  - rule: Detect Crypto Mining
    desc: Alert on crypto mining activity
    condition: >
      spawned_process and container and
      (proc.name in (xmrig, ethminer) or
       proc.cmdline contains "stratum+tcp")
    output: >
      Crypto mining detected (container=%container.id command=%proc.cmdline)
    priority: CRITICAL
    tags: [crypto, mining]
```

---

## Key Rotation

### Automated Key Rotation

```python
# Configure key rotation
from dashboard.services.security.hsm import KeyManager, RotationPolicy

key_manager = KeyManager(
    rotation_policy=RotationPolicy(
        # Signing keys
        signing_key_rotation_days=90,

        # Encryption keys
        encryption_key_rotation_days=365,

        # TLS certificates
        certificate_rotation_days=30,

        # API keys
        api_key_rotation_days=90,
    ),
    notify_before_expiry_days=14,
    auto_rotate=True,
)
```

---

## Verification Commands

```bash
# Verify security configuration
kubectl exec -it deploy/dashboard -n lego-mcp-prod -- python -c "
from dashboard.services.security import ZeroTrustGateway, PQCryptoProvider
from dashboard.services.compliance import NISTComplianceChecker

# Check Zero Trust
zt = ZeroTrustGateway()
print(f'Zero Trust Status: {zt.get_status()}')

# Check PQ Crypto
pq = PQCryptoProvider()
print(f'PQ Algorithms: {pq.supported_algorithms}')

# Check NIST Compliance
nist = NISTComplianceChecker()
report = nist.run_assessment()
print(f'NIST 800-171 Score: {report.score}%')
"

# Verify network policies
kubectl get networkpolicies -n lego-mcp-prod

# Check pod security standards
kubectl get pods -n lego-mcp-prod -o jsonpath='{range .items[*]}{.metadata.name}: {.spec.securityContext}{"\n"}{end}'

# Verify image signatures
for pod in $(kubectl get pods -n lego-mcp-prod -o jsonpath='{.items[*].spec.containers[*].image}' | tr ' ' '\n' | sort -u); do
  cosign verify $pod 2>/dev/null && echo "✓ $pod signed" || echo "✗ $pod NOT signed"
done
```

---

## Emergency Procedures

### Security Incident Response

1. **Isolate**: Apply emergency network policy to isolate affected pods
2. **Preserve**: Capture forensic data before any changes
3. **Analyze**: Review audit logs and SIEM alerts
4. **Remediate**: Patch vulnerability or revoke compromised credentials
5. **Recover**: Restore from known-good state
6. **Report**: Document incident per CMMC IR requirements

```bash
# Emergency isolation
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: emergency-isolate
  namespace: lego-mcp-prod
spec:
  podSelector:
    matchLabels:
      app: compromised-pod
  policyTypes:
  - Ingress
  - Egress
EOF

# Capture forensics
kubectl exec compromised-pod -- tar -czf /tmp/forensics.tar.gz /var/log /etc
kubectl cp lego-mcp-prod/compromised-pod:/tmp/forensics.tar.gz ./forensics-$(date +%s).tar.gz
```

---

## Contacts

- **Security Team:** security@lego-mcp.io
- **SOC:** soc@lego-mcp.io (24/7)
- **Incident Hotline:** +1-XXX-XXX-XXXX
- **PGP Key:** 0xABCD1234 (security@lego-mcp.io)
