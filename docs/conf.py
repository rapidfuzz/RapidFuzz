# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import datetime


# -- Project information -----------------------------------------------------

project = 'RapidFuzz'
copyright = str(datetime.datetime.now().year) + ", RapidFuzz Team"
author = 'RapidFuzz Team'
# html_logo = "_static/RapidFuzz.svg"
master_doc = "index"

version = ""
# release = ""
with open("../VERSION", "r") as version_file:
    # release = version_file.read().strip()
    version = version_file.read().strip()
release = version


html_theme_options = {
    "fixed_sidebar": True,

    'logo': 'RapidFuzz.svg',
    "description": ("Rapid fuzzy string matching using the "
                    "Levenshtein Distance"),
    "sidebar_collapse": True,
    "show_relbar_bottom": True,
    "show_powered_by": False,

     'github_user': "maxbachmann",
     'github_repo': "rapidfuzz",
    # "github_button": True,
    # "github_type": "count",
    # "github_banner": True,
}

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_tabs.tabs'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
