import os
import sys
from unittest.mock import MagicMock

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'TransiFlow'
copyright = '2024 Utrecht University'
author = 'Sven Baars'
release = '0.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Allow apidoc to import the code. External dependencies need to be mocked to
# make that possible without installing them, which is hard.
sys.path.insert(0, os.path.abspath('..'))

sys.modules['PyTrilinos'] = MagicMock()
sys.modules['HYMLS'] = MagicMock()
sys.modules['jadapy'] = MagicMock()
sys.modules['jadapy.orthogonalization'] = MagicMock()
sys.modules['petsc4py'] = MagicMock()


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.mathjax',
    'sphinxcontrib.apidoc',
    'numpydoc',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

add_module_names = False

apidoc_module_dir = '../transiflow'
apidoc_output_dir = 'reference'
apidoc_excluded_paths = [
    'interface',
    'interface/**',
    'BoundaryConditions.py',
    'Continuation.py',
    'CrsMatrix.py',
    'CylindricalDiscretization.py',
    'TimeIntegration.py',
    'Discretization.py',
]
apidoc_separate_modules = True
apidoc_toc_file = False
apidoc_module_first = True

autodoc_default_options = {
    'inherited-members': True
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Options for MathJax -----------------------------------------------------
mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'macros': {
            'd': '\\textrm{d}',
            'Re': '\\textrm{Re}',
        }
    }
}
