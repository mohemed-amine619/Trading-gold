"""
core/config — runtime configuration plane.

One declarative registry drives:

* validation (type + range + enum) for every dashboard write,
* the schema the dashboard renders,
* .env persistence keys,
* the audit trail,
* the "is this safe to change while a position is open?" flag.

This exists because ``run.py``'s CONFIG and ``core/settings.py`` had drifted
into two different sets of defaults, and the dashboard could only edit a
hand-maintained subset of them. A key that is not in this registry cannot be
written from the API, and a key that is in it is guaranteed to be validated and
audited.
"""

from core.config.schema import (
    ParamSpec,
    PARAMS,
    GROUP_ORDER,
    schema_payload,
    validate_payload,
    coerce_value,
)
from core.config.audit import record_changes, recent_changes

__all__ = [
    "ParamSpec",
    "PARAMS",
    "GROUP_ORDER",
    "schema_payload",
    "validate_payload",
    "coerce_value",
    "record_changes",
    "recent_changes",
]
