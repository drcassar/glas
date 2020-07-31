import setuptools

setuptools.setup(
    name='glas',
    version='1.0',
    author='Daniel Roberto Cassar',
    author_email='contact@danielcassar.com.br',
    description='Python module for solving inverse design of glasses',
    url="https://github.com/drcassar/glas",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy>=1.16',
        'pandas>=1.0.0',
        'deap',
        'mendeleev',
    ],
    keywords='glass, non-crystalline materials, inverse design, genetic algorithm',
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "Environment :: Console",
    ],
    license='GPL',
    python_requires='>=3.6',
)
