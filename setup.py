from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='anchor-topic',
    version='0.1.2',
    description='A package for anchor-based topic models.',
    long_description=readme,
    author='Michelle Yuan',
    author_email='myuan@cs.umd.edu',
    url='https://github.com/forest-snow/anchor-topic',
    license=license,
    packages=find_packages(exclude=('tests')),
    install_requires=[
        'numpy',
        'scipy',
        'numba'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ]
)