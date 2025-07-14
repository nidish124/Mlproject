from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path:str)->List[str]:
    ''' 
    this function will return list of requirements
    '''

    HYPEN_e_dot = "-e ."
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        if HYPEN_e_dot in requirements:
            requirements.remove(HYPEN_e_dot)
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='Nidish',
    author_email='nidish124@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
