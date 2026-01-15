# LEGO MCP World-Class Manufacturing Deployment Guide

## Overview

This guide covers deployment of the LEGO MCP Manufacturing System for DoD/ONR-class production environments with IEC 61508 SIL 2+ functional safety certification.

**Target Environment:**
- Kubernetes 1.28+ (AKS, EKS, GKE, or on-premise)
- PostgreSQL 15+ with TimescaleDB
- Redis 7+ for caching
- NATS/Kafka for messaging
- Harbor/GHCR for container registry

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Control Plane | 3 nodes, 4 vCPU, 16GB RAM | 5 nodes, 8 vCPU, 32GB RAM |
| Worker Nodes | 6 nodes, 8 vCPU, 32GB RAM | 12 nodes, 16 vCPU, 64GB RAM |
| Storage | 1TB NVMe SSD | 4TB NVMe RAID-10 |
| Network | 10 Gbps | 25 Gbps |
| GPU (Vision) | NVIDIA T4 | NVIDIA A100 |

### Software Prerequisites

```bash
# Required CLI tools
kubectl >= 1.28
helm >= 3.12
cosign >= 2.2
tctl >= 1.4  # Temporal CLI
flux >= 2.1  # GitOps
```

### Security Prerequisites

- [ ] HSM configured (AWS CloudHSM, Azure Dedicated HSM, or Thales Luna)
- [ ] Certificate Authority provisioned
- [ ] OIDC provider configured (Azure AD, Okta, or Keycloak)
- [ ] Network policies defined
- [ ] Secrets management (HashiCorp Vault) deployed

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                          External Load Balancer                       │
│                         (TLS termination, WAF)                        │
└─────────────────────────────────┬────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼────────────────────────────────────┐
│                           Ingress Controller                          │
│                        (NGINX/Traefik + mTLS)                         │
└─────────────────────────────────┬────────────────────────────────────┘
                                  │
     ┌────────────────────────────┼────────────────────────────────────┐
     │                            │                                    │
┌────▼─────┐  ┌──────────────────▼────────────────────┐  ┌────────────▼──────────┐
│ Dashboard │  │           MCP Server                  │  │   Fusion 360 Add-in  │
│ (Flask)   │  │    (Core Manufacturing Logic)        │  │   (HTTP Bridge)       │
└─────┬─────┘  └──────────────────┬────────────────────┘  └───────────────────────┘
      │                           │
      │  ┌───────────────────────▼───────────────────────┐
      │  │              Service Mesh (Linkerd/Istio)     │
      │  └───────────────────────┬───────────────────────┘
      │                          │
┌─────▼──────────────────────────▼───────────────────────────────────────────────┐
│                              Worker Services                                    │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐ │
│  │   Vision    │ │   Slicer    │ │   Quality   │ │   Digital   │ │  Safety   │ │
│  │   Service   │ │   Service   │ │   Control   │ │   Twin      │ │  Monitor  │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
      │
┌─────▼───────────────────────────────────────────────────────────────────────────┐
│                           Data & Messaging Layer                                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐  │
│  │ PostgreSQL  │ │ TimescaleDB │ │   Redis     │ │    NATS     │ │  Kafka    │  │
│  │ (Primary)   │ │ (Telemetry) │ │  (Cache)    │ │  (Events)   │ │ (Streams) │  │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Deployment Steps

### Step 1: Prepare Kubernetes Cluster

```bash
# Create namespace with network policies
kubectl create namespace lego-mcp-prod

# Apply Pod Security Standards
kubectl label namespace lego-mcp-prod \
  pod-security.kubernetes.io/enforce=restricted \
  pod-security.kubernetes.io/audit=restricted \
  pod-security.kubernetes.io/warn=restricted

# Install Linkerd service mesh
linkerd install --crds | kubectl apply -f -
linkerd install | kubectl apply -f -
linkerd check

# Inject Linkerd into namespace
kubectl annotate namespace lego-mcp-prod linkerd.io/inject=enabled
```

### Step 2: Deploy Secrets Management

```bash
# Install HashiCorp Vault
helm repo add hashicorp https://helm.releases.hashicorp.com
helm install vault hashicorp/vault \
  --namespace vault \
  --set "server.ha.enabled=true" \
  --set "server.ha.replicas=3"

# Configure Kubernetes auth
vault auth enable kubernetes
vault write auth/kubernetes/config \
  kubernetes_host="https://$KUBERNETES_PORT_443_TCP_ADDR:443"

# Create secrets policy for LEGO MCP
vault policy write lego-mcp - <<EOF
path "secret/data/lego-mcp/*" {
  capabilities = ["read"]
}
path "transit/encrypt/lego-mcp" {
  capabilities = ["update"]
}
path "transit/decrypt/lego-mcp" {
  capabilities = ["update"]
}
EOF
```

### Step 3: Deploy Database Layer

```bash
# Deploy PostgreSQL with TimescaleDB
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install postgres bitnami/postgresql \
  --namespace lego-mcp-prod \
  --set image.tag=15-debian-11 \
  --set auth.postgresPassword=$(vault kv get -field=password secret/lego-mcp/postgres) \
  --set primary.persistence.size=100Gi \
  --set replication.enabled=true \
  --set replication.numSynchronousReplicas=2

# Enable TimescaleDB extension
kubectl exec -it postgres-postgresql-0 -- psql -U postgres -c \
  "CREATE EXTENSION IF NOT EXISTS timescaledb;"

# Run migrations
kubectl apply -f k8s/jobs/database-migration.yaml
```

### Step 4: Deploy Core Services

```bash
# Add LEGO MCP Helm repo
helm repo add lego-mcp https://charts.lego-mcp.io

# Deploy with production values
helm install lego-mcp lego-mcp/manufacturing \
  --namespace lego-mcp-prod \
  --values helm/values-production.yaml \
  --set global.imageRegistry=ghcr.io/lego-mcp \
  --set global.imagePullSecrets[0]=ghcr-secret \
  --set dashboard.replicas=3 \
  --set mcpServer.replicas=5 \
  --set visionService.replicas=2 \
  --set slicerService.replicas=2

# Verify deployment
kubectl get pods -n lego-mcp-prod
kubectl get svc -n lego-mcp-prod
```

### Step 5: Configure Ingress with TLS

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create ClusterIssuer for Let's Encrypt
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: security@lego-mcp.io
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF

# Create Ingress with mTLS
kubectl apply -f k8s/ingress/production.yaml
```

### Step 6: Deploy Observability Stack

```bash
# Install Grafana Stack
helm repo add grafana https://grafana.github.io/helm-charts

# Deploy Loki for logs
helm install loki grafana/loki-stack \
  --namespace monitoring \
  --set grafana.enabled=true \
  --set prometheus.enabled=true

# Deploy Tempo for traces
helm install tempo grafana/tempo \
  --namespace monitoring

# Deploy Mimir for metrics
helm install mimir grafana/mimir-distributed \
  --namespace monitoring

# Configure OpenTelemetry Collector
kubectl apply -f k8s/monitoring/otel-collector.yaml
```

### Step 7: Enable Formal Verification CI

The formal verification GitHub Actions workflow is already configured at `.github/workflows/formal-verification.yml`. Ensure these are enabled:

```yaml
# Trigger on push to protected branches
on:
  push:
    branches: [main, develop, release/*]
    paths:
      - 'ros2_ws/src/lego_mcp_safety_certified/formal/**'
```

---

## Security Hardening

### Container Image Signing

All container images must be signed using cosign before deployment:

```bash
# Sign container image
cosign sign --yes ghcr.io/lego-mcp/dashboard:v8.0.0

# Verify signature before deployment
cosign verify \
  --certificate-identity=https://github.com/lego-mcp/lego-mcp/.github/workflows/release.yml@refs/tags/v8.0.0 \
  --certificate-oidc-issuer=https://token.actions.githubusercontent.com \
  ghcr.io/lego-mcp/dashboard:v8.0.0
```

### SBOM Attestation

Generate and attach SBOM to container images:

```bash
# Generate SBOM
syft ghcr.io/lego-mcp/dashboard:v8.0.0 -o cyclonedx-json > sbom.json

# Attach SBOM attestation
cosign attest --yes --predicate sbom.json \
  --type cyclonedx \
  ghcr.io/lego-mcp/dashboard:v8.0.0
```

### Network Policies

Apply strict network policies:

```yaml
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
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-dashboard-to-mcp
  namespace: lego-mcp-prod
spec:
  podSelector:
    matchLabels:
      app: dashboard
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: mcp-server
    ports:
    - protocol: TCP
      port: 8080
```

---

## Post-Deployment Verification

### Health Checks

```bash
# Check all pods are running
kubectl get pods -n lego-mcp-prod -o wide

# Verify service mesh
linkerd check --proxy -n lego-mcp-prod

# Check database connectivity
kubectl exec -it deploy/dashboard -- python -c "
from dashboard.services.database.session import get_session
with get_session() as session:
    print('Database connected:', session.execute('SELECT 1').scalar())
"
```

### Security Validation

```bash
# Run security scanner
trivy k8s --report summary namespace lego-mcp-prod

# Verify RBAC
kubectl auth can-i --list --as=system:serviceaccount:lego-mcp-prod:dashboard

# Check certificate expiry
kubectl get certificates -n lego-mcp-prod
```

### Compliance Verification

```bash
# Run NIST 800-171 compliance check
kubectl exec -it deploy/dashboard -- python -c "
from dashboard.services.compliance import NISTComplianceChecker
checker = NISTComplianceChecker()
report = checker.run_assessment()
print(f'Compliance Score: {report.score}%')
print(f'Controls Implemented: {report.implemented}/{report.total}')
"

# Verify HSM connectivity
kubectl exec -it deploy/dashboard -- python -c "
from dashboard.services.security.hsm import KeyManager
km = KeyManager()
print(f'HSM Status: {km.get_status()}')
"
```

### Formal Verification

```bash
# Trigger formal verification workflow
gh workflow run formal-verification.yml

# Check results
gh run list --workflow=formal-verification.yml
```

---

## Disaster Recovery

### Backup Procedures

```bash
# Database backup (daily)
kubectl exec -it postgres-postgresql-0 -- pg_dump -U postgres lego_mcp | \
  gzip > backup-$(date +%Y%m%d).sql.gz

# Upload to S3 with encryption
aws s3 cp backup-*.sql.gz s3://lego-mcp-backups/ \
  --sse aws:kms \
  --sse-kms-key-id alias/lego-mcp-backup

# Vault backup
vault operator raft snapshot save vault-backup-$(date +%Y%m%d).snap
```

### Recovery Procedures

```bash
# Restore database
gunzip -c backup-*.sql.gz | \
  kubectl exec -i postgres-postgresql-0 -- psql -U postgres lego_mcp

# Restore Vault
vault operator raft snapshot restore vault-backup-*.snap

# Verify data integrity
kubectl exec -it deploy/dashboard -- python -c "
from dashboard.services.traceability import DigitalThread
thread = DigitalThread()
thread.verify_chain()
"
```

---

## Scaling Guide

### Horizontal Pod Autoscaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: dashboard-hpa
  namespace: lego-mcp-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: dashboard
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Vertical Pod Autoscaling

```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: vision-service-vpa
  namespace: lego-mcp-prod
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vision-service
  updatePolicy:
    updateMode: Auto
  resourcePolicy:
    containerPolicies:
    - containerName: vision
      minAllowed:
        cpu: 1
        memory: 2Gi
      maxAllowed:
        cpu: 16
        memory: 64Gi
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Resolution |
|-------|-------|------------|
| Pod CrashLoopBackOff | OOM or dependency failure | Check logs, increase resources |
| TLS handshake failure | Certificate mismatch | Regenerate certificates |
| Database connection timeout | Network policy blocking | Check NetworkPolicy rules |
| HSM connection refused | Firewall or credentials | Verify HSM network access |

### Debug Commands

```bash
# View pod logs
kubectl logs -f deploy/dashboard -n lego-mcp-prod

# Exec into pod for debugging
kubectl exec -it deploy/dashboard -- /bin/bash

# Check events
kubectl get events -n lego-mcp-prod --sort-by='.lastTimestamp'

# View service mesh metrics
linkerd viz stat deploy -n lego-mcp-prod
```

---

## Contact & Support

- **Security Issues:** security@lego-mcp.io (PGP key: 0x...)
- **Operations:** ops@lego-mcp.io
- **On-Call:** PagerDuty escalation policy "LEGO MCP Prod"

---

## Appendix: Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABASE_URL` | PostgreSQL connection string | Yes |
| `REDIS_URL` | Redis connection string | Yes |
| `VAULT_ADDR` | HashiCorp Vault address | Yes |
| `HSM_SLOT` | HSM slot number | Production |
| `OIDC_ISSUER` | OIDC provider URL | Yes |
| `SENTRY_DSN` | Error tracking DSN | Recommended |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OpenTelemetry collector | Recommended |
