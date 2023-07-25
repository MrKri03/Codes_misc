import argparse
import subprocess
import os 
import yaml
from conda.base.context import context
from conda.cli import common

def create_conda_environment(env_path, requirements):
    
    if requirements.endswith(".yml"):
    # Create Conda environment with requirements
        with open(os.path.join(env_path,requirements), "r") as yml_file:
            yaml_data = yaml.safe_load(yml_file)

        if not isinstance(yaml_data, dict):
            print("\n\n\t\t--Invalid YAML file format--\n")
            return
        name_value = yaml_data.get("name")
          
        packages = yaml_data.get("dependencies")
    

        if name_value is None or packages is None:
            print("\n\n\t\t--Missing 'name' or 'dependencies' in the YAML file--\n")
            return
        

        conda_cmd = [
            'conda', 'create', '--prefix', os.path.join(env_path, name_value)
        ]
        conda_cmd.extend(packages)

        # Execute the conda command
        subprocess.run(conda_cmd)

    else:
        print("\n\n\t\t--You should provide a .yml file for the requirements--\n")
        pass


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Create a Conda environment and install packages.')
parser.add_argument('--requirements',dest='requirements', required=True, help='Requirements to create the environment')

args = parser.parse_args()

# Call function to create the Conda environment and install packages
create_conda_environment(os.getcwd(), args.requirements)

## ACTIVATE YOUR ENVIRONMENT USING ANACONDA PROMPT
