from feast import Entity
from feast.types import ValueType

# Define the entity (unique identifier for each machine)
machine = Entity(name="machine", 
                 join_keys=["Machine_ID"], 
                 value_type=ValueType.STRING, 
                 description="Machine ID")