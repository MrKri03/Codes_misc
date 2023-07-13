import argparse
import subprocess
import os 

def create_conda_environment(env_path,requirements):
    
    if requirements.endswith(".yml"):
    # Create Conda environment with requirements
        create_command = f'conda env create -f {os.path.join(env_path,requirements)}'
    
        subprocess.call(create_command)
    else:
        print("\n\n\t\t--You should provide a .yml file for the requirements--\n")
        pass


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Create a Conda environment and install packages.')
parser.add_argument('--path', dest='env_path', required=True, help='Path where the Conda environment will be created')
#parser.add_argument('--name', dest='env_name', required=True, help='Name of the Conda environment')
parser.add_argument('--requirements',dest='requirements', required=True, help='Requirements to create the environment')
args = parser.parse_args()

# Call function to create the Conda environment and install packages
create_conda_environment(args.env_path, args.requirements)

## ACTIVATE YOUR ENVIRONMENT USING ANACONDA PROMPT