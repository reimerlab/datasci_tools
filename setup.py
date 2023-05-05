from setuptools import setup, find_packages

setup(
   name='python_tools', # the name of the package, which can be different than the folder when using pip instal
   version='1.0',
   description='Modules to help faciliatate basic python programming (specifically for data science) and wrappers for common packages ',
   author='Brendan Celii',
   author_email='brendanacelii@gmail.com',
   packages=find_packages(),  #teslls what packages to be included for the install
   install_requires=[
        'colour>=0.1.5',
        'datajoint>=0.12.9',
        'ipyvolume>=0.5.2',
        'ipywebrtc>=0.5.0',
        'ipywidgets>=7.5.1',
        'matplotlib>=3.3.4',
        "modin[all]",
        'networkx>=2.5',
        'numpy>=1.19.5',
        'pandas>=1.1.5',
        'pandasql>=0.7.3',
         #'pydot>=1.4.2',
        'pykdtree>=1.3.1',
        'scikit_learn>=0.23.1',
        'scipy>=1.5.4',
        'seaborn>=0.11.2',
        'Shapely>=1.7.0',
        'six>=1.11.0',
        'tqdm>=4.62.2',
        'trimesh>=3.9.0',
        'webcolors>=1.11.1',
   ], #external packages as dependencies
    
    # if wanted to install with the extra requirements use pip install -e ".[interactive]"
    extras_require={
        #'interactive': ['matplotlib>=2.2.0', 'jupyter'],
    },
    
    # if have a python script that wants to be run from the command line
    entry_points={
        #'console_scripts': ['pipeline_download=Applications.Eleox_Data_Fetch.Eleox_Data_Fetcher_vp1:main']
    },
    scripts=[], 
    
)