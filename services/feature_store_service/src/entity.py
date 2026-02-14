"""
Entity Definitions for Feast Feature Store

Entities represent the primary keys for feature lookups.
In our case, each washing machine is uniquely identified by Machine_ID.
"""

from feast import Entity
from feast.value_type import ValueType

# ============================================================================
# MACHINE ENTITY
# ============================================================================
# Represents a unique washing machine in the system
# All features are associated with a specific machine via this entity
machine = Entity(
    name="machine",
    join_keys=["Machine_ID"],  # Primary key for joining features
    value_type=ValueType.INT64,
    description="Unique identifier for each washing machine in the facility"
)