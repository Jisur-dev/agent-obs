from .repository import ConfigurationDiff
from dataclasses import fields

class VersionDifferenceCalculator:

    def compare(self,version_a,version_b):
        differences = {}

        config_a=version_a.configuration
        config_b=version_b.configuration
        for field in fields(config_a):
            name=field.name
            value_a=getattr(config_a,name)
            value_b=getattr(config_b,name)
            if value_a!=value_b:
                differences[name]=(value_a,value_b)

        return ConfigurationDiff(
            version_a=version_a.version_id,
            version_b=version_b.version_id,
            changed_fields=differences
        )
        