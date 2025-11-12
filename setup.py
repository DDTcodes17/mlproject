# Bulding Tool/Application as a Package/Library in Pypi
# find_packages => Lists all packages/directories with any folder with __init__.py

from setuptools import find_packages, setup
from typing import List

HYPHEN_CONST = "-e ."    # used for running setup.py simultaneously with requirements.txt

def get_requirements(file_path:str)->List[str]:
    '''Returns list of Required Packages''' 
    requirement = []
    with open(file_path) as file:
        requirement = file.readlines()    # Reads \n also
        requirement = [req.replace("\n", "") for req in requirement]

        if HYPHEN_CONST in requirement:          #For not reading -e . as package
            requirement.remove(HYPHEN_CONST)
    return requirement        

setup(
    name='mlproject',
    version='0.0.1',
    author='Dhruv Tiwari',
    author_email='tiwaridhruv15@gmail.com',
    packages = find_packages(),              #Installs directories/Source_code of Parent Package(__init__.py())
    #Efficient installation of packages
    install_requires=get_requirements('requirements.txt')     #Installs libraries when pip install parent package
)

##Parent Package mlproject.egg-info created