"""Setup-script for energy_fault_detector package."""
from setuptools import setup, find_packages

PACKAGE_NAME = 'energy_fault_detector'  # name used by PIP
IMPORT_NAME = 'energy_fault_detector'  # name used for imports


with open('requirements.txt', 'r') as f:
    requirements_install = f.readlines()

with open('requirements-dev.txt', 'r') as f:
    requirements_dev = f.readlines()

with open('requirements-docs.txt', 'r') as f:
    requirements_docs = f.readlines()

# All the requirements
requirements = {
    'install': [
        r.replace('\n', '') for r in requirements_install if not r.startswith('-r ') and not r.startswith('--extra')
    ],
    'docs': [
        r.replace('\n', '') for r in requirements_docs if not r.startswith('-r ')
    ],
    'dev': [
        r.replace('\n', '') for r in requirements_dev if not r.startswith('-r ')
    ]
}

with open(f'{PACKAGE_NAME}/VERSION', 'r') as f:
    VERSION = f.readlines()[0].strip()


if __name__ == '__main__':
    setup(
        name=PACKAGE_NAME,
        version=VERSION,

        packages=find_packages(exclude=['docs', 'tests', 'tests.*', 'notebooks']),

        install_requires=requirements['install'],
        extras_require={k: r for k, r in requirements.items() if k != 'install'},
    )
