from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT='-e .'

def get_requirements(file_path: str) -> list[str]:
    """This function will return the list of requirements."""
    
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]  # Strip to remove any additional whitespace
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements


setup(
name= 'ML Project using Regression', 
version='0.0.1',
author='Vijay Kumar Kodam',
author_email='vijay.kodam98@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')
)