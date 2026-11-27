from .repository import Version, ConfigurationDiff
from dataclasses import fields
from agent_obs.Exeptions.type_errors import type_validator


class VersionDifferenceCalculator:
    """Calculates differences between the configurations of two Version objects"""
    
    @staticmethod
    def compare(version_a: Version, version_b: Version) -> ConfigurationDiff:
        """Compare the configurations of two versions and identify changed fields

        
            version_a (Version): The first version object to compare
            version_b (Version): The second version object to compare

            ConfigurationDiff: An object containing both version IDs and a dictionary
            mapping each changed field name (str) to a tuple of (old_value, new_value)
        """
        #create an instance pf the validator
        validator = type_validator()
        
        #validate the input versions
        validator.version_type_non_valid(version_a)
        validator.version_type_non_valid(version_b)
        
        #validate the configurations
        validator.config_type_non_valid(version_a)
        validator.config_type_non_valid(version_b)

        
        differences = {}

        config_a = version_a.configuration
        config_b = version_b.configuration

        for field in fields(config_a):
            name = field.name
            value_a = getattr(config_a, name)
            value_b = getattr(config_b, name)

            if value_a != value_b:
                differences[name] = (value_a, value_b)

        return ConfigurationDiff(
            version_a=version_a.version_id,
            version_b=version_b.version_id,
            changed_fields=differences,
        )