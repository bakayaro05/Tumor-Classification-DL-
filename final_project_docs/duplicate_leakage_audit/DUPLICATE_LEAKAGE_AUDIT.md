# Duplicate / Leakage Audit

Status: **MATERIAL_DUPLICATION**

Images hashed: 7023
- Train: 4569
- Validation: 1143
- Test: 1311

Exact/pixel duplicate cross-split pairs: 464
High-confidence near-duplicate pairs: 98
Unresolved review pairs: 481

Comparisons performed: Train↔Validation, Train↔Test, Validation↔Test.
The audit uses file SHA-256, decoded-pixel SHA-256, 64-bit pHash, 64-bit dHash, and grayscale pixel similarity for perceptual candidates.

This image-level audit cannot establish patient-level independence because reliable patient identifiers are not available in the project dataset.
