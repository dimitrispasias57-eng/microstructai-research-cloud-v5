"""Prospective decision/audit rules. No image access or segmentation code."""
import hashlib

VERSION = 'V4_M2_RESERVED_VALIDATION_RULES_V1'
AUDIT_SEED = 'RBC_V4_M2_RESERVED50_AUDIT_20261025_V1'

def system_status(source_status, mask_available, ai_class=None, qc_flags=(), ai_flags=(), fallback=False, technical_failure=False):
    if technical_failure:
        return 'HUMAN_REVIEW_REQUIRED_TECHNICAL'
    if source_status == 'SOURCE_UNUSABLE' and not mask_available:
        return 'PROVISIONAL_REJECT_SOURCE'
    if source_status == 'UNCERTAIN' or not mask_available:
        return 'HUMAN_REVIEW_REQUIRED_UNCERTAIN'
    if source_status != 'SEGMENTATION_PRODUCED':
        raise ValueError('Unrecognized source/mask state')
    if qc_flags or fallback:
        return 'HUMAN_REVIEW_REQUIRED_QC'
    if ai_class == 'UNUSABLE':
        # Proposer supplied a complete boundary but reviewer disputes eligibility.
        return 'HUMAN_REVIEW_REQUIRED_SOURCE_CONFLICT'
    if ai_class == 'MAJOR_ERROR':
        return 'PROVISIONAL_REJECT_MASK'
    if ai_class in ['MINOR_ERROR', 'REVIEWER_UNCERTAIN', None]:
        return 'HUMAN_REVIEW_REQUIRED_QUALITY'
    if ai_class == 'GOOD':
        return 'HUMAN_REVIEW_REQUIRED_FLAGS' if ai_flags else 'PROVISIONAL_AUTO_ACCEPT'
    raise ValueError('Unrecognized AI class')

def audit_plan(records):
    """Records have unique ID, original_sha256, and frozen system status only."""
    assert len({r['id'] for r in records}) == len(records)
    groups = {}
    for r in records:
        status = r['status']
        stratum = ('AUTO_ACCEPT' if status == 'PROVISIONAL_AUTO_ACCEPT' else
                   'REJECT_SOURCE' if status == 'PROVISIONAL_REJECT_SOURCE' else
                   'REJECT_MASK' if status == 'PROVISIONAL_REJECT_MASK' else
                   'REQUIRED' if status.startswith('HUMAN_REVIEW_REQUIRED_') else None)
        assert stratum is not None
        groups.setdefault(stratum, []).append(r)
    selected = []
    for stratum, rr in sorted(groups.items()):
        limit = len(rr) if stratum == 'REQUIRED' else 20 if stratum == 'AUTO_ACCEPT' else 5
        n = min(len(rr), limit)
        def key(r):
            payload = '|'.join([AUDIT_SEED, stratum, r['id'], r['original_sha256']])
            return hashlib.sha256(payload.encode('utf-8')).hexdigest(), r['id']
        order = sorted(rr, key=key)
        selected.extend(dict(r, audit_stratum=stratum, inclusion_probability=n/len(rr),
                             selected=i<n, audit_rank=i+1)
                        for i,r in enumerate(order))
    return selected
