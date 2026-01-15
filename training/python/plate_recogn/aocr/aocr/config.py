import os, pathlib, yaml, logging
from collections import namedtuple

class ConfigObject:
    """ https://stackoverflow.com/a/73184811 """
    def __init__(self, nested_dict):
        self.nested_dict = nested_dict

    def _parse(self):
        nested_dict = self.nested_dict
        obj_type = type(nested_dict)
        if obj_type is not dict:
            raise TypeError(f"Expected 'dict' but found '{obj_type}'")
        return self._transform_to_named_tuples("root", nested_dict)

    def _transform_to_named_tuples(self, tuple_name, possibly_nested_obj):
        if type(possibly_nested_obj) is dict:
            named_tuple_def = namedtuple(tuple_name, possibly_nested_obj.keys())
            transformed_value = named_tuple_def(
                *[
                    self._transform_to_named_tuples(key, value)
                    for key, value in possibly_nested_obj.items()
                ]
            )
        elif type(possibly_nested_obj) is list:
            transformed_value = [
                self._transform_to_named_tuples(f"{tuple_name}_{i}", possibly_nested_obj[i])
                for i in range(len(possibly_nested_obj))
            ]
        else:
            transformed_value = possibly_nested_obj

        return transformed_value
    
class Config:
    @staticmethod
    def parse(file_path :str) -> namedtuple:
        with open(file_path, 'r') as file:
            return ConfigObject(yaml.safe_load(file))._parse()
        
        
    