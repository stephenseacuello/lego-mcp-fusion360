# ADR-001: Post-Quantum Cryptography

## Status

Accepted

## Date

2026-01-10

## Context

The LEGO MCP system handles sensitive manufacturing data, intellectual property, and potentially classified DoD/ONR information. With the advancement of quantum computing, current cryptographic algorithms (RSA, ECDSA, ECDH) will become vulnerable to Shor's algorithm attacks.

NIST has standardized three post-quantum cryptographic algorithms:
- **FIPS 203**: ML-KEM (Kyber) for key encapsulation
- **FIPS 204**: ML-DSA (Dilithium) for digital signatures
- **FIPS 205**: SLH-DSA (SPHINCS+) for hash-based signatures

The system must:
1. Protect data confidentiality for 20+ years (cryptographic longevity)
2. Meet CMMC Level 3 requirements for DoD contracts
3. Support regulatory compliance (NIST 800-171, IEC 62443)
4. Enable gradual migration without breaking existing systems

## Decision

We will implement **hybrid post-quantum cryptography** that combines classical and PQ algorithms:

1. **Key Encapsulation**: ML-KEM-768 + X25519
   - ML-KEM-768 provides NIST Security Level 3
   - X25519 provides fallback if PQ fails

2. **Digital Signatures**: ML-DSA-65 + Ed25519
   - ML-DSA-65 for new signatures
   - Ed25519 for backward compatibility

3. **Hash-Based Signatures**: SLH-DSA-SHA2-128s
   - For long-term archival signatures
   - Code signing and firmware verification

4. **Implementation**: Python wrapper using liboqs
   - `dashboard/services/security/pq_crypto.py`

## Consequences

### Positive

- **Quantum-Safe**: Protected against future quantum attacks
- **Standards-Compliant**: Uses NIST-approved algorithms
- **Hybrid Approach**: No single point of failure
- **Audit-Ready**: Demonstrates security due diligence

### Negative

- **Larger Keys/Signatures**: ML-DSA signatures are ~2.4KB vs 64B for Ed25519
- **Performance Impact**: ~10x slower than classical crypto
- **Dependency**: Requires liboqs library
- **Complexity**: Must manage two key types

### Risks

- PQ algorithms are relatively new (standardized 2024)
- Implementation bugs in cryptographic libraries
- Key management complexity increases

### Mitigations

- Use well-tested libraries (liboqs, PQShield)
- Implement extensive test suite for crypto operations
- Hybrid mode provides fallback
- Regular security audits

## Implementation Notes

```python
# Key generation
pq = PostQuantumCrypto()
signing_key = pq.generate_signing_keypair(PQAlgorithm.ML_DSA_65)
kem_key = pq.generate_kem_keypair(PQAlgorithm.ML_KEM_768)

# Hybrid encryption
result = pq.encrypt_hybrid(plaintext, mode=EncryptionMode.HYBRID)
```

## References

- [NIST FIPS 203 (ML-KEM)](https://csrc.nist.gov/pubs/fips/203/final)
- [NIST FIPS 204 (ML-DSA)](https://csrc.nist.gov/pubs/fips/204/final)
- [NIST FIPS 205 (SLH-DSA)](https://csrc.nist.gov/pubs/fips/205/final)
- [liboqs - Open Quantum Safe](https://github.com/open-quantum-safe/liboqs)
