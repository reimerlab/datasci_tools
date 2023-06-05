# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
#sys.path.insert(0, os.path.abspath('../python_tools'))
#import python_tools



project = 'python_tools'
copyright = '2023, Brendan Celii'
author = 'Brendan Celii'
release = '1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_static_path = ['_static']

import sphinx_rtd_theme

extensions = [
  "sphinx_rtd_theme",
  "sphinx.ext.autodoc",#converts the doc string into documentation
  "sphinx.ext.viewcode",
  "sphinx.ext.napoleon"
]

html_theme = "sphinx_rtd_theme"

# autodoc_mock_imports = [
#    "external_library"
# ]

exclude_patterns = ['*.ipynb_checkpoints*']

def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip

def setup(app):
    app.connect("autodoc-skip-member", skip)
