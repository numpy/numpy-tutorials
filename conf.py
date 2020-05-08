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
import os
import re

# -- Project information -----------------------------------------------------

project = 'NumPy Tutorials'
copyright = '2008-2020, The SciPy community'

# The default replacements for |version| and |release|, also used in various
# other places throughout the built documents.
#
import numpy
# The short X.Y version (including .devXXXX, rcX, b1 suffixes if present)
version = re.sub(r'(\d+\.\d+)\.\d+(.*)', r'\1\2', numpy.__version__)
version = re.sub(r'(\.dev\d+).*?$', r'\1', version)
# The full version, including alpha/beta/rc tags.
release = numpy.__version__
print("%s %s" % (version, release))

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'nbsphinx',
    'numpydoc',
    'matplotlib.sphinxext.plot_directive',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'scipy-sphinx-theme']

master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

themedir = os.path.join('scipy-sphinx-theme', '_theme')
if not os.path.isdir(themedir):
    raise RuntimeError("Get the scipy-sphinx-theme first, "
                       "via git submodule init && git submodule update")

html_theme = 'scipy'
html_theme_path = [themedir]

if 'scipyorg' in tags:
    # Build for the scipy.org website
    html_theme_options = {
        "edit_link": True,
        "sidebar": "right",
        "scipy_org_logo": True,
        "rootlinks": [("https://scipy.org/", "Scipy.org"),
                      ("https://docs.scipy.org/", "Docs")]
    }
else:
    # Default build
    html_theme_options = {
        "edit_link": False,
        "sidebar": "left",
        "scipy_org_logo": False,
        "rootlinks": [("https://numpy.org/", "NumPy.org"),
                      ("https://numpy.org/doc", "Docs"),
                     ]
    }
    html_sidebars = {'index': ['indexsidebar.html', 'searchbox.html'],
                     '**': ['customsidebar.html', 'globaltoc.html']}

#html_additional_pages = {
#    'index': 'indexcontent.html',
#}

html_title = "%s" % project
html_static_path = ['_static']
html_last_updated_fmt = '%b %d, %Y'

#html_use_modindex = True
html_copy_source = True
html_domain_indices = False
html_file_suffix = '.html'
html_sourcelink_suffix = '.ipynb'

htmlhelp_basename = 'numpy'

if 'sphinx.ext.pngmath' in extensions:
    pngmath_use_preview = True
    pngmath_dvipng_args = ['-gamma', '1.5', '-D', '96', '-bg', 'Transparent']

plot_html_show_formats = False
plot_html_show_source_link = False

