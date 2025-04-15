import os
import yaml  
import argparse

def load_conf(task: str, imbtype: str, algorithm: str, to_parser: bool = True):
    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config"))
    config_path = os.path.join(config_dir, task, imbtype, algorithm + ".yml")
    
    print(f"Load config file from: {config_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)  
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

    if not to_parser:
        return config

    parser = argparse.ArgumentParser(description=f"Configuration for {algorithm}")
    
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                parser.add_argument(f"--{key}", action="store_true", default=True, 
                                    help=f"Enable {key} (default: {value})")
                parser.add_argument(f"--no-{key}", dest=key, action="store_false", 
                                    help=f"Disable {key}")
            else:
                parser.add_argument(f"--{key}", action="store_true", default=False, 
                                    help=f"Enable {key} (default: {value})")
                parser.add_argument(f"--no-{key}", dest=key, action="store_false", 
                                    help=f"Disable {key} (default: {value})")
        else:
            parser.add_argument(f"--{key}", type=type(value), default=value, 
                                help=f"{key} (default: {value})")

    return parser.parse_args()