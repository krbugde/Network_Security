from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."  # Fixed typo: "HYPEN" → "HYPHEN"

def get_requirements(file_path: str) -> List[str]:
    """
    This function returns a list of packages required to be installed.
    """
    requirements = []

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]  # strip() handles \n, \r, spaces

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    print(requirements)
    return requirements


setup(
    name="NetworkSecurity",
    author="Kumudini",
    version="0.0.1",
    author_email="kumudinibugde@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)