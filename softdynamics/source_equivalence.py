"""Explicit, receipt-bound equivalence of validated execution implementations.

Physical inputs never participate in aliases. Keep actual source hashes in the
receipt, and resolve only the validated transfer implementation when constructing
material pairs. Frozen replay continues to pin the complete receipt hash.
"""


def physical_sources(record):
    sources=dict(record['source_hashes'])
    proof=record.get('source_equivalence')
    if proof is None:
        return sources
    if (proof.get('schema')!='softdynamics.transfer-equivalence.v1' or
            not proof.get('validation') or set(proof.get('aliases',{}))!={'transfer'}):
        raise ValueError('Invalid execution equivalence receipt')
    alias=proof['aliases']['transfer']
    if sources.get('transfer')!=alias['executed']:
        raise ValueError('Execution equivalence does not match the actual transfer')
    sources['transfer']=alias['reference']
    return sources
