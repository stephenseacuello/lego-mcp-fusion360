# LEGO MCP v8.0 Troubleshooting Guide

This guide helps diagnose and resolve common issues with the LEGO MCP v8.0 DoD/ONR-class manufacturing system.

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Common Issues](#common-issues)
3. [Service-Specific Issues](#service-specific-issues)
4. [Performance Issues](#performance-issues)
5. [Security Issues](#security-issues)
6. [Integration Issues](#integration-issues)
7. [Recovery Procedures](#recovery-procedures)

---

## Quick Diagnostics

### System Health Check

```bash
# Check all services
docker-compose ps

# Check service health endpoints
curl -s http://localhost:5000/health | jq
curl -s http://localhost:3000/health | jq
curl -s http://localhost:8081/health | jq

# Check database connectivity
docker-compose exec db pg_isready -U lego_mcp

# Check Redis connectivity
docker-compose exec redis redis-cli ping
```

### Log Collection

```bash
# Collect all logs
docker-compose logs > logs/all_services_$(date +%Y%m%d_%H%M%S).log

# Collect specific service logs
docker-compose logs dashboard > logs/dashboard.log
docker-compose logs mcp-server > logs/mcp-server.log

# Follow logs in real-time
docker-compose logs -f --tail=100 dashboard
```

### Resource Usage

```bash
# Check container resource usage
docker stats --no-stream

# Check disk usage
df -h

# Check memory
free -m
```

---

## Common Issues

### Issue: Services Won't Start

**Symptoms:**
- `docker-compose up` fails
- Services crash immediately after starting
- Health checks failing

**Diagnosis:**
```bash
# Check logs for startup errors
docker-compose logs dashboard 2>&1 | grep -i error

# Check if ports are available
netstat -tlnp | grep -E "(5000|3000|8081|5432|6379)"

# Verify environment variables
docker-compose config
```

**Solutions:**

1. **Port Conflicts:**
   ```bash
   # Find process using port
   lsof -i :5000

   # Kill conflicting process
   kill -9 <PID>

   # Or change port in docker-compose.yml
   ```

2. **Missing Environment Variables:**
   ```bash
   # Copy example env file
   cp config/examples/development.env.example .env

   # Edit and set required values
   nano .env
   ```

3. **Database Not Ready:**
   ```bash
   # Wait for database
   docker-compose up -d db
   sleep 10
   docker-compose up -d
   ```

---

### Issue: Database Connection Errors

**Symptoms:**
- `FATAL: password authentication failed`
- `connection refused`
- `too many connections`

**Diagnosis:**
```bash
# Check database logs
docker-compose logs db

# Check connection count
docker-compose exec db psql -U lego_mcp -c "SELECT count(*) FROM pg_stat_activity;"

# Test connection manually
docker-compose exec db psql -U lego_mcp -d lego_mcp -c "SELECT 1;"
```

**Solutions:**

1. **Wrong Credentials:**
   ```bash
   # Verify credentials in .env
   grep DATABASE_URL .env

   # Reset password if needed
   docker-compose exec db psql -U postgres -c "ALTER USER lego_mcp WITH PASSWORD 'new_password';"
   ```

2. **Connection Pool Exhaustion:**
   ```python
   # Increase pool size in config
   DB_POOL_SIZE=30
   DB_MAX_OVERFLOW=20
   ```

3. **Database Not Initialized:**
   ```bash
   # Run migrations
   docker-compose exec dashboard alembic upgrade head

   # Or recreate database
   docker-compose down -v
   docker-compose up -d
   ```

---

### Issue: Redis Connection Issues

**Symptoms:**
- `NOAUTH Authentication required`
- `Connection timed out`
- Cache misses

**Diagnosis:**
```bash
# Check Redis logs
docker-compose logs redis

# Test Redis connection
docker-compose exec redis redis-cli -a ${REDIS_PASSWORD} ping

# Check memory usage
docker-compose exec redis redis-cli -a ${REDIS_PASSWORD} info memory
```

**Solutions:**

1. **Authentication Failed:**
   ```bash
   # Verify password in .env
   grep REDIS_PASSWORD .env

   # Test with correct password
   docker-compose exec redis redis-cli -a your_password ping
   ```

2. **Memory Full:**
   ```bash
   # Flush cache (careful in production!)
   docker-compose exec redis redis-cli -a ${REDIS_PASSWORD} FLUSHALL

   # Or increase memory limit
   # In redis.conf: maxmemory 1gb
   ```

---

### Issue: API Returning 500 Errors

**Symptoms:**
- Internal server error responses
- Stack traces in logs
- Intermittent failures

**Diagnosis:**
```bash
# Check dashboard logs
docker-compose logs dashboard | grep -A 10 "500\|Error\|Exception"

# Check specific endpoint
curl -v http://localhost:5000/api/v8/equipment

# Check for Python tracebacks
docker-compose logs dashboard 2>&1 | grep -A 20 "Traceback"
```

**Solutions:**

1. **Import Errors:**
   ```bash
   # Rebuild container with fresh dependencies
   docker-compose build --no-cache dashboard
   docker-compose up -d dashboard
   ```

2. **Configuration Errors:**
   ```bash
   # Validate configuration
   docker-compose exec dashboard python -c "from dashboard.app import create_app; app = create_app()"
   ```

3. **Database Schema Mismatch:**
   ```bash
   # Check migration status
   docker-compose exec dashboard alembic current

   # Apply pending migrations
   docker-compose exec dashboard alembic upgrade head
   ```

---

## Service-Specific Issues

### Dashboard Service

**Issue: Templates Not Rendering**

```bash
# Check template syntax
docker-compose exec dashboard python -c "
from jinja2 import Environment, FileSystemLoader
env = Environment(loader=FileSystemLoader('templates'))
env.get_template('pages/index.html')
print('Templates OK')
"
```

**Issue: Static Files Not Loading**

```bash
# Verify static files exist
docker-compose exec dashboard ls -la static/

# Check nginx proxy (if using)
curl -I http://localhost/static/css/style.css
```

### MCP Server

**Issue: Tools Not Registering**

```bash
# Check MCP server logs
docker-compose logs mcp-server | grep -i "tool\|register"

# Verify tool definitions
docker-compose exec mcp-server cat /app/tools.json
```

**Issue: Fusion 360 Connection Failed**

```bash
# Check HTTP server status
curl -v http://localhost:3000/api/fusion360/status

# Verify Fusion 360 add-in is running
# Check Fusion 360 > Scripts and Add-ins > LegoMCP
```

### Slicer Service

**Issue: Slicing Fails**

```bash
# Check slicer logs
docker-compose logs slicer-service | grep -i error

# Verify profiles exist
docker-compose exec slicer-service ls -la /app/profiles/

# Test with sample STL
curl -X POST http://localhost:8081/slice \
  -F "file=@sample.stl" \
  -F "profile=prusa_mk4"
```

---

## Performance Issues

### Issue: Slow API Responses

**Diagnosis:**
```bash
# Enable SQL query logging
export SQLALCHEMY_ECHO=true

# Check response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:5000/api/v8/equipment

# Check for slow queries
docker-compose exec db psql -U lego_mcp -c "
SELECT query, calls, mean_time, total_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;
"
```

**Solutions:**

1. **Add Database Indexes:**
   ```sql
   -- Check missing indexes
   SELECT schemaname, tablename, indexname
   FROM pg_indexes
   WHERE tablename = 'equipment';

   -- Add index if missing
   CREATE INDEX IF NOT EXISTS ix_equipment_status ON equipment(status);
   ```

2. **Enable Caching:**
   ```python
   # Verify cache is working
   from dashboard.services.caching import cache
   cache.set('test', 'value', timeout=60)
   print(cache.get('test'))
   ```

3. **Increase Resources:**
   ```yaml
   # In docker-compose.yml
   services:
     dashboard:
       deploy:
         resources:
           limits:
             cpus: '2.0'
             memory: 2G
   ```

### Issue: High Memory Usage

**Diagnosis:**
```bash
# Check container memory
docker stats --no-stream --format "table {{.Name}}\t{{.MemUsage}}"

# Check for memory leaks
docker-compose exec dashboard python -c "
import tracemalloc
tracemalloc.start()
# Run suspect operation
snapshot = tracemalloc.take_snapshot()
for stat in snapshot.statistics('lineno')[:10]:
    print(stat)
"
```

**Solutions:**

1. **Reduce Worker Count:**
   ```bash
   # In gunicorn config
   workers = 2  # Instead of auto-scaling
   ```

2. **Enable Memory Limits:**
   ```yaml
   # In docker-compose.yml
   deploy:
     resources:
       limits:
         memory: 1G
   ```

---

## Security Issues

### Issue: HSM Connection Failed

**Symptoms:**
- `HSM unavailable` alerts
- Signing operations failing
- Audit sealing errors

**Diagnosis:**
```bash
# Check HSM connectivity
docker-compose exec dashboard python -c "
from dashboard.services.security.hsm import KeyManager
km = KeyManager()
print(f'HSM Status: {km.get_status()}')
"

# Check HSM logs
docker-compose logs dashboard | grep -i hsm
```

**Solutions:**

1. **Verify HSM Configuration:**
   ```bash
   # Check environment variables
   grep HSM .env

   # Test with simulation mode
   HSM_TYPE=simulator docker-compose up -d dashboard
   ```

2. **Network Issues:**
   ```bash
   # Check HSM endpoint connectivity
   curl -v https://hsm.internal:12345/health
   ```

### Issue: Certificate Expiring

**Diagnosis:**
```bash
# Check certificate expiry
openssl x509 -in /path/to/cert.pem -noout -enddate

# Check all certificates
docker-compose exec dashboard python -c "
import ssl
import socket
context = ssl.create_default_context()
with socket.create_connection(('localhost', 5000)) as sock:
    with context.wrap_socket(sock, server_hostname='localhost') as ssock:
        cert = ssock.getpeercert()
        print(f'Expires: {cert[\"notAfter\"]}')
"
```

**Solutions:**

1. **Renew Certificates:**
   ```bash
   # Using cert-manager (Kubernetes)
   kubectl delete secret lego-mcp-tls -n lego-mcp
   # cert-manager will auto-renew

   # Manual renewal
   certbot renew --force-renewal
   ```

### Issue: Authentication Failures

**Diagnosis:**
```bash
# Check auth logs
docker-compose logs dashboard | grep -i "auth\|login\|401\|403"

# Verify JWT configuration
docker-compose exec dashboard python -c "
from dashboard.services.security import verify_jwt
# Test token verification
"
```

**Solutions:**

1. **Token Expired:**
   ```bash
   # Increase token lifetime (development only)
   JWT_ACCESS_TOKEN_EXPIRES=7200
   ```

2. **Secret Mismatch:**
   ```bash
   # Regenerate JWT secret
   python -c "import secrets; print(secrets.token_hex(32))"
   # Update in .env and restart
   ```

---

## Integration Issues

### Issue: ROS2 Bridge Not Working

**Symptoms:**
- Equipment data not updating
- Commands not reaching equipment
- ROS2 topics empty

**Diagnosis:**
```bash
# Check ROS2 nodes
ros2 node list

# Check topics
ros2 topic list
ros2 topic echo /equipment/status

# Check bridge logs
docker-compose logs ros2-bridge
```

**Solutions:**

1. **Domain ID Mismatch:**
   ```bash
   # Verify domain ID matches
   export ROS_DOMAIN_ID=42
   ros2 topic list
   ```

2. **DDS Configuration:**
   ```bash
   # Check CycloneDDS config
   export CYCLONE_DDS_URI=/etc/cyclonedds/cyclonedds.xml
   ```

### Issue: OPC-UA Connection Failed

**Diagnosis:**
```bash
# Check OPC-UA server
docker-compose exec dashboard python -c "
from asyncua import Client
import asyncio

async def check():
    client = Client('opc.tcp://localhost:4840')
    await client.connect()
    print('Connected!')
    await client.disconnect()

asyncio.run(check())
"
```

**Solutions:**

1. **Security Mode Mismatch:**
   ```python
   # Try different security modes
   client.set_security_string("Basic256Sha256,SignAndEncrypt,cert.der,key.pem")
   ```

---

## Recovery Procedures

### Database Recovery

```bash
# Stop services
docker-compose stop dashboard mcp-server

# Restore from backup
docker-compose exec db psql -U postgres -c "DROP DATABASE IF EXISTS lego_mcp;"
docker-compose exec db psql -U postgres -c "CREATE DATABASE lego_mcp OWNER lego_mcp;"
docker-compose exec db psql -U lego_mcp -d lego_mcp < backups/db_backup_latest.sql

# Run migrations
docker-compose exec dashboard alembic upgrade head

# Restart services
docker-compose up -d
```

### Full System Reset

```bash
# CAUTION: This destroys all data!

# Stop everything
docker-compose down -v

# Remove all data
rm -rf ./data/*

# Rebuild
docker-compose build --no-cache

# Start fresh
docker-compose up -d

# Initialize database
docker-compose exec dashboard alembic upgrade head
```

### Rollback Deployment

```bash
# Kubernetes rollback
kubectl rollout undo deployment/dashboard -n lego-mcp

# Docker rollback
docker-compose pull dashboard:previous-tag
docker-compose up -d dashboard
```

---

## Getting Help

If you're unable to resolve an issue:

1. **Check Documentation:**
   - [User Guide](./USER_GUIDE.md)
   - [Developer Guide](./DEVELOPER.md)
   - [API Reference](./API_V8.md)

2. **Collect Diagnostic Info:**
   ```bash
   ./scripts/collect_diagnostics.sh > diagnostics.tar.gz
   ```

3. **Contact Support:**
   - Create GitHub Issue with diagnostic info
   - Email: support@lego-mcp.io
   - Include: version, environment, steps to reproduce

---

**Document Version:** 8.0.0
**Last Updated:** 2024-01-15
