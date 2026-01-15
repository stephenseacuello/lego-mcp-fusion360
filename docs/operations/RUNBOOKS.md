# LEGO MCP v8.0 Operations Runbooks

This document contains operational runbooks for common scenarios and incident response procedures.

## Table of Contents

1. [Daily Operations](#daily-operations)
2. [Incident Response](#incident-response)
3. [Maintenance Procedures](#maintenance-procedures)
4. [Scaling Operations](#scaling-operations)
5. [Backup and Recovery](#backup-and-recovery)
6. [Security Operations](#security-operations)

---

## Daily Operations

### Runbook: Morning Health Check

**Purpose:** Verify system health at start of shift

**Frequency:** Daily, start of each shift

**Steps:**

1. **Check Dashboard Health**
   ```bash
   curl -s https://lego-mcp.example.com/health | jq
   ```
   Expected: `{"status": "healthy"}`

2. **Review Prometheus Alerts**
   - Navigate to Grafana: https://grafana.lego-mcp.example.com
   - Check "LEGO MCP Alerts" dashboard
   - Verify no critical alerts firing

3. **Check Equipment Status**
   ```bash
   curl -s https://lego-mcp.example.com/api/v8/equipment | jq '.data[] | {id: .equipment_id, status: .status}'
   ```

4. **Review Overnight Logs**
   ```bash
   # Check for errors in last 12 hours
   kubectl logs -n lego-mcp -l app=dashboard --since=12h | grep -i error
   ```

5. **Verify Database Health**
   ```bash
   kubectl exec -n lego-mcp deploy/postgresql -- pg_isready -U lego_mcp
   ```

6. **Document Findings**
   - Log any anomalies in shift handover notes
   - Create tickets for non-critical issues

---

### Runbook: Shift Handover

**Purpose:** Transfer operational knowledge between shifts

**Frequency:** Every shift change

**Steps:**

1. **Review Active Alerts**
   - Document any ongoing incidents
   - Note alert acknowledgment status

2. **Manufacturing Status**
   - Current production orders
   - Equipment utilization
   - Quality metrics

3. **Pending Actions**
   - Unresolved issues
   - Scheduled maintenance
   - Upcoming deployments

4. **Documentation**
   - Update shift log in Confluence/Wiki
   - Brief incoming operator

---

## Incident Response

### Runbook: Service Outage (P1)

**Severity:** P1 - Critical
**Response Time:** 15 minutes

**Symptoms:**
- Dashboard returning 5xx errors
- Health check failures
- Multiple service alerts

**Steps:**

1. **Acknowledge Incident** (0-5 min)
   ```bash
   # Acknowledge in PagerDuty/Alertmanager
   # Notify incident channel: #lego-mcp-incidents
   ```

2. **Initial Assessment** (5-15 min)
   ```bash
   # Check pod status
   kubectl get pods -n lego-mcp

   # Check recent events
   kubectl get events -n lego-mcp --sort-by='.lastTimestamp' | tail -20

   # Check logs
   kubectl logs -n lego-mcp deploy/dashboard --tail=100
   ```

3. **Identify Root Cause**

   **If pods are crashlooping:**
   ```bash
   kubectl describe pod -n lego-mcp -l app=dashboard
   kubectl logs -n lego-mcp -l app=dashboard --previous
   ```

   **If database issues:**
   ```bash
   kubectl exec -n lego-mcp deploy/postgresql -- pg_isready
   kubectl logs -n lego-mcp deploy/postgresql --tail=50
   ```

   **If network issues:**
   ```bash
   kubectl get networkpolicy -n lego-mcp
   kubectl get svc -n lego-mcp
   ```

4. **Implement Fix**

   **Restart pods (if OOM or transient):**
   ```bash
   kubectl rollout restart deployment/dashboard -n lego-mcp
   ```

   **Rollback deployment (if recent change):**
   ```bash
   kubectl rollout undo deployment/dashboard -n lego-mcp
   ```

   **Scale up (if load related):**
   ```bash
   kubectl scale deployment/dashboard -n lego-mcp --replicas=5
   ```

5. **Verify Recovery**
   ```bash
   # Check pods are running
   kubectl get pods -n lego-mcp -l app=dashboard

   # Verify health
   curl -s https://lego-mcp.example.com/health

   # Run smoke tests
   ./scripts/smoke_test.sh production
   ```

6. **Post-Incident**
   - Document timeline in incident ticket
   - Schedule post-mortem within 48 hours
   - Update runbook if new scenario

---

### Runbook: Database Connection Exhaustion

**Severity:** P2 - High
**Response Time:** 30 minutes

**Symptoms:**
- "too many connections" errors
- Application timeouts
- Connection pool alerts

**Steps:**

1. **Check Connection Count**
   ```bash
   kubectl exec -n lego-mcp deploy/postgresql -- psql -U lego_mcp -c \
     "SELECT count(*) FROM pg_stat_activity;"
   ```

2. **Identify Connection Sources**
   ```bash
   kubectl exec -n lego-mcp deploy/postgresql -- psql -U lego_mcp -c \
     "SELECT client_addr, count(*) FROM pg_stat_activity GROUP BY client_addr ORDER BY count DESC;"
   ```

3. **Kill Idle Connections (if needed)**
   ```bash
   kubectl exec -n lego-mcp deploy/postgresql -- psql -U lego_mcp -c \
     "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle' AND query_start < now() - interval '10 minutes';"
   ```

4. **Increase Pool Size (temporary)**
   ```bash
   # Update deployment
   kubectl set env deployment/dashboard -n lego-mcp DB_POOL_SIZE=30
   ```

5. **Long-term Fix**
   - Review connection pool settings
   - Add PgBouncer if not present
   - Optimize query patterns

---

### Runbook: HSM Unavailable

**Severity:** P1 - Critical (Security)
**Response Time:** 15 minutes

**Symptoms:**
- "HSM unavailable" alerts
- Signing operations failing
- Audit sealing errors

**Steps:**

1. **Verify HSM Connectivity**
   ```bash
   # Check HSM endpoint
   curl -v https://hsm.internal:12345/health

   # Check from pod
   kubectl exec -n lego-mcp deploy/dashboard -- curl -s https://hsm.internal:12345/health
   ```

2. **Check HSM Service Status**
   - Log into HSM management console
   - Verify HSM is powered on and connected
   - Check HSM logs for errors

3. **Enable Fallback Mode (if configured)**
   ```bash
   # Enable software fallback (reduces security)
   kubectl set env deployment/dashboard -n lego-mcp HSM_FALLBACK_ENABLED=true
   ```

4. **Contact HSM Vendor**
   - Open support ticket with HSM vendor
   - Provide error logs and timestamps

5. **Document Impact**
   - Note which operations were affected
   - Verify audit chain integrity after recovery

---

## Maintenance Procedures

### Runbook: Database Backup Verification

**Purpose:** Verify database backups are valid
**Frequency:** Weekly

**Steps:**

1. **List Recent Backups**
   ```bash
   kubectl exec -n lego-mcp deploy/postgresql -- ls -la /backups/
   ```

2. **Verify Backup Integrity**
   ```bash
   # Check backup file
   kubectl exec -n lego-mcp deploy/postgresql -- pg_restore --list /backups/latest.dump | head -20
   ```

3. **Test Restore (in staging)**
   ```bash
   # Restore to staging
   kubectl exec -n lego-mcp-staging deploy/postgresql -- \
     pg_restore -U lego_mcp -d lego_mcp_test /backups/latest.dump
   ```

4. **Document Results**
   - Record backup size and timestamp
   - Note any errors during verification
   - Update backup verification log

---

### Runbook: Certificate Renewal

**Purpose:** Renew TLS certificates before expiry
**Frequency:** Before expiry (typically 30 days)

**Steps:**

1. **Check Certificate Expiry**
   ```bash
   echo | openssl s_client -servername lego-mcp.example.com -connect lego-mcp.example.com:443 2>/dev/null | openssl x509 -noout -dates
   ```

2. **If Using cert-manager (automatic):**
   ```bash
   # Check certificate status
   kubectl get certificate -n lego-mcp

   # Force renewal if needed
   kubectl delete secret lego-mcp-tls -n lego-mcp
   ```

3. **If Manual Renewal:**
   ```bash
   # Renew with certbot
   certbot renew --force-renewal

   # Update Kubernetes secret
   kubectl create secret tls lego-mcp-tls \
     --cert=/etc/letsencrypt/live/lego-mcp.example.com/fullchain.pem \
     --key=/etc/letsencrypt/live/lego-mcp.example.com/privkey.pem \
     -n lego-mcp --dry-run=client -o yaml | kubectl apply -f -
   ```

4. **Verify New Certificate**
   ```bash
   curl -v https://lego-mcp.example.com 2>&1 | grep "expire date"
   ```

---

## Scaling Operations

### Runbook: Scale for High Load

**Purpose:** Scale services during high-traffic periods
**Trigger:** CPU > 70% sustained for 10 minutes

**Steps:**

1. **Assess Current Load**
   ```bash
   kubectl top pods -n lego-mcp
   kubectl get hpa -n lego-mcp
   ```

2. **Manual Scale (if HPA not responding)**
   ```bash
   # Scale dashboard
   kubectl scale deployment/dashboard -n lego-mcp --replicas=10

   # Scale workers
   kubectl scale deployment/celery-worker -n lego-mcp --replicas=8
   ```

3. **Scale Database Read Replicas**
   ```bash
   # If using managed database
   # Increase replica count via cloud console

   # If self-managed
   kubectl scale statefulset/postgresql-replica -n lego-mcp --replicas=3
   ```

4. **Monitor Scaling**
   ```bash
   watch kubectl get pods -n lego-mcp
   ```

5. **Scale Down After Load**
   ```bash
   # Reduce replicas gradually
   kubectl scale deployment/dashboard -n lego-mcp --replicas=3
   ```

---

## Backup and Recovery

### Runbook: Full System Recovery

**Purpose:** Recover from complete system failure
**RTO:** 4 hours
**RPO:** 1 hour

**Steps:**

1. **Assess Damage**
   - Identify affected components
   - Determine data loss extent
   - Document timeline

2. **Provision Infrastructure**
   ```bash
   # If cluster destroyed, recreate
   terraform apply -target=module.eks_cluster

   # Apply base manifests
   kubectl apply -f k8s/base/
   ```

3. **Restore Database**
   ```bash
   # Deploy PostgreSQL
   helm install postgresql bitnami/postgresql -n lego-mcp

   # Restore from backup
   kubectl exec -n lego-mcp deploy/postgresql -- \
     pg_restore -U lego_mcp -d lego_mcp /backups/latest.dump
   ```

4. **Deploy Applications**
   ```bash
   helm upgrade --install lego-mcp ./helm/lego-mcp \
     -n lego-mcp \
     -f helm/values-production.yaml
   ```

5. **Verify Recovery**
   ```bash
   ./scripts/smoke_test.sh production
   ```

6. **Verify Data Integrity**
   - Check audit chain integrity
   - Verify recent transactions
   - Confirm equipment status

---

## Security Operations

### Runbook: Security Incident Response

**Purpose:** Respond to potential security breach
**Severity:** P1 - Critical

**Steps:**

1. **Contain (0-15 minutes)**
   ```bash
   # Isolate affected pods
   kubectl label pod <pod-name> -n lego-mcp quarantine=true

   # Apply network isolation
   kubectl apply -f k8s/emergency/quarantine-networkpolicy.yaml
   ```

2. **Preserve Evidence**
   ```bash
   # Capture pod state
   kubectl describe pod <pod-name> -n lego-mcp > incident-$(date +%s).log
   kubectl logs <pod-name> -n lego-mcp > incident-logs-$(date +%s).log

   # Capture network traffic (if pcap available)
   kubectl exec -n lego-mcp <pod-name> -- tcpdump -w /tmp/capture.pcap &
   ```

3. **Assess Impact**
   - Review audit logs for unauthorized access
   - Check for data exfiltration
   - Identify compromised credentials

4. **Eradicate**
   ```bash
   # Rotate affected credentials
   kubectl delete secret <compromised-secret> -n lego-mcp

   # Regenerate API keys
   # Rotate database passwords
   ```

5. **Recover**
   ```bash
   # Deploy clean images
   kubectl rollout restart deployment -n lego-mcp

   # Re-enable normal operations
   kubectl delete -f k8s/emergency/quarantine-networkpolicy.yaml
   ```

6. **Report**
   - Notify CISO and security team
   - File incident report
   - Update threat model

---

## Contact Information

| Role | Contact | Escalation |
|------|---------|------------|
| On-Call Engineer | PagerDuty | Automatic |
| Platform Lead | platform-lead@lego-mcp.io | After 30 min |
| Security Team | security@lego-mcp.io | Immediately for security |
| CISO | ciso@lego-mcp.io | P1 security incidents |
| Database Admin | dba@lego-mcp.io | Database issues |

---

**Document Version:** 8.0.0
**Last Updated:** 2024-01-15
**Review Frequency:** Quarterly
