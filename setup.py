from setuptools import setup, find_packages

setup(
    name='qcbm', 
    version='0.2.0alpha', 
    author='Mohammad Ghazivakili',
    author_email='m.ghazivakili@gmail.com',
    description='A Quantum Circuit Born Machine Generator',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',  # This tells setuptools to expect Markdown content in the README file
    url='https://github.com/aspuru-guzik-group/qcbm',  # URL to the GitHub repository or package website
    package_dir={'': 'src'},  # Root package directory
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Choose the appropriate license for your project
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Specify the minimum Python version required
    install_requires=[
        'numpy',  # List your package dependencies here
        'requests',
        'qiskit',
        'qiskit-ibm-runtime',
        'qiskit[visualization]',
        'ipykernel',
        'torch',
        'qulacs',
        'tqdm',
        'scipy',
        
    ],
    # entry_points={
    #     'console_scripts': [
    #         'your_command=your_package.module:function',  # Example entry point for a command line script
    #     ],
    # },
    # include_package_data=True,  # Include additional files specified in MANIFEST.in
    # package_data={
    #     '': ['*.txt', '*.rst'],  # Example of including additional files
    #     'your_package': ['data/*.dat'],  # Example of including data files within a package
    # },
)
