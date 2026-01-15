# ADR-007: HSM-Backed Audit Trail Sealing

## Status

Accepted

## Date

2026-01-14

## Context

The LEGO MCP system maintains tamper-evident audit trails for:
- Security events (authentication, authorization)
- Manufacturing operations (work orders, quality results)
- Configuration changes
- Equipment commands

Current implementation uses SHA-256 hash chains for tamper detection. However, this has limitations:
1. **No External Trust**: Chain can be rebuilt if attacker has full access
2. **No Time Attestation**: Timestamps can be backdated
3. **Regulatory Gap**: DoD requires hardware-backed integrity (CMMC AU.2.041-044)
4. **Recovery Uncertainty**: Hard to prove chain wasn't modified pre-detection

## Decision

We will implement **HSM-Backed Daily Audit Sealing** that cryptographically binds the audit chain to hardware timestamps:

### 1. Sealing Architecture

```
Daily at 00:00 UTC:
  1. Compute hash of all audit entries since last seal
  2. Include: entry_count, first_hash, last_hash, timestamp
  3. Send to HSM for signing with non-exportable key
  4. Store seal in append-only storage
  5. Broadcast seal hash to external witness services
```

### 2. Seal Structure

```python
@dataclass
class DailySeal:
    seal_date: date                    # UTC date being sealed
    entry_count: int                   # Number of entries in period
    chain_hash: str                    # SHA-256 of entire day's entries
    first_entry_hash: str              # Hash of first entry
    last_entry_hash: str               # Hash of last entry
    previous_seal_hash: str            # Link to previous seal
    hsm_signature: bytes               # Hardware-backed signature
    hsm_key_id: str                    # Identifier of signing key
    timestamp: datetime                # HSM-attested timestamp
    witness_receipts: List[str]        # External attestations
```

### 3. HSM Configuration

**Supported Hardware:**
- YubiHSM 2 (FIPS 140-2 Level 3)
- AWS CloudHSM (FIPS 140-2 Level 3)
- Azure Dedicated HSM
- TPM 2.0 (FIPS 140-2 Level 1+)

**Key Properties:**
- Algorithm: ECDSA P-384 or ML-DSA-65 (post-quantum)
- Non-exportable: Key never leaves HSM
- Audit: HSM logs all signing operations
- Rotation: Annual key rotation with overlap

### 4. Verification Process

```
Verification Steps:
  1. Retrieve seal for date range
  2. Verify HSM signature using public key
  3. Recompute chain_hash from audit entries
  4. Compare computed vs sealed hash
  5. Verify seal chain integrity (previous_seal_hash)
  6. Check external witness receipts
```

### 5. External Witnessing

**Witness Services:**
- RFC 3161 Time Stamping Authority
- Blockchain anchoring (optional)
- Cross-organizational witness exchange

### Implementation

- `dashboard/services/traceability/hsm_sealer.py`
- `dashboard/services/traceability/audit_chain.py` (modified)
- `dashboard/services/security/hsm/key_manager.py`

## Consequences

### Positive

- **Hardware Trust**: Signatures bound to tamper-resistant hardware
- **Time Attestation**: HSM provides trusted timestamps
- **Regulatory Compliance**: Meets CMMC AU.2.041-044
- **Forensic Value**: Can prove data existed at specific time
- **Non-Repudiation**: Can't deny creating audit entries

### Negative

- **HSM Dependency**: System requires HSM availability
- **Operational Complexity**: HSM requires careful key management
- **Cost**: HSM hardware/service has ongoing cost
- **Latency**: HSM operations add ~10-50ms per seal

### Risks

- HSM unavailability prevents sealing
- Key compromise (unlikely but catastrophic)
- Seal chain corruption

### Mitigations

- Redundant HSM deployment (primary + backup)
- HSM key ceremony with multi-person authorization
- Daily verification of seal chain
- Immediate alerting on any verification failure

## Seal Chain Diagram

```
Day 1          Day 2          Day 3          Day 4
+--------+     +--------+     +--------+     +--------+
|Seal_001|<----|Seal_002|<----|Seal_003|<----|Seal_004|
|        |     |        |     |        |     |        |
|entries:|     |entries:|     |entries:|     |entries:|
| 15,420 |     | 18,230 |     | 12,890 |     | 16,450 |
|        |     |        |     |        |     |        |
|HSM sig:|     |HSM sig:|     |HSM sig:|     |HSM sig:|
| 0xABC..|     | 0xDEF..|     | 0x123..|     | 0x456..|
+--------+     +--------+     +--------+     +--------+
    |              |              |              |
    v              v              v              v
+--------------------------------------------------------+
|              Audit Entry Hash Chain                     |
|  E1 -> E2 -> E3 -> ... -> E15420 -> E15421 -> ...      |
+--------------------------------------------------------+
```

## Implementation Notes

```python
from dashboard.services.traceability.hsm_sealer import HSMSealer, create_hsm_sealer
from dashboard.services.traceability.audit_chain import DigitalThread

# Initialize HSM sealer
sealer = create_hsm_sealer(
    hsm_type="yubihsm",
    key_id="audit-signing-key-2026",
)

# Get audit chain
audit = DigitalThread()

# Create daily seal
seal = sealer.create_daily_seal(
    entries=audit.get_entries_for_date(date.today() - timedelta(days=1)),
    previous_seal=sealer.get_latest_seal(),
)

# Verify seal
result = sealer.verify_seal(seal)
assert result.valid
print(f"Verified {result.entry_count} entries")

# Verify entire chain
chain_result = sealer.verify_seal_chain(
    start_date=date(2026, 1, 1),
    end_date=date.today(),
)
assert chain_result.all_valid
```

## Compliance Mapping

| Requirement | Control | Implementation |
|-------------|---------|----------------|
| CMMC AU.2.041 | Audit storage integrity | HSM-signed seals |
| CMMC AU.2.042 | Audit tampering alerting | Verification + alerts |
| CMMC AU.2.043 | Audit review capability | Chain verification API |
| CMMC AU.2.044 | Audit correlation | Trace ID in entries |
| NIST 800-171 3.3.1 | Audit record creation | DigitalThread |
| NIST 800-171 3.3.2 | Unique audit records | UUID + hash |

## References

- [FIPS 140-3: Security Requirements for Cryptographic Modules](https://csrc.nist.gov/publications/detail/fips/140/3/final)
- [RFC 3161: Time-Stamp Protocol](https://tools.ietf.org/html/rfc3161)
- [CMMC Assessment Guide](https://www.acq.osd.mil/cmmc/)
- [YubiHSM 2 Documentation](https://developers.yubico.com/YubiHSM2/)
