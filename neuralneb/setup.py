from setuptools import find_packages, setup


setup(
    author= "Mathias Schreiner",
    name ='neuralneb',
    packages=find_packages(),
    install_requires = ['torch', 'ase', 'pytest', 'progressbar']
)
