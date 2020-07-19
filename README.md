# NumPy tutorials

This set of tutorials and educational materials is being developed,
IT IS NOT YET INTEGRATED IN THE HTML DOCS AT https://www.numpy.org/devdocs/

The goal of this repository is to provide high-quality resources by the
NumPy project, both for self-learning and for teaching classes with.

## Contributing

We very much welcome contributions! If you have an idea or proposal for a new
tutorial, please open an issue with an outline.

## Jupyter Notebooks

The choice of Jupyter Notebook in this repo instead of the usual format 
([reStructuredText, through Sphinx](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html))
used in the main NumPy documentation has two reasons:

 * Jupyter notebooks are a common format for communicating scientific
   information.
 * rST may present a barrier for some people who might otherwise be very
   interested in contributing tutorial material.

## Generating the "site"

Sphinx is configured with the appropriate extensions to execute the notebooks
and generated webpages from them. To accomplish this from a fresh repo:

1. Install the dependencies: from a terminal, run
   
   ```
   pip install -r requirements.txt
   ```

   To execute the notebooks, you'll also need to install the dependencies for
   the tutorial(s) themselves:

   ```
   pip install -r content/requirements.txt
   ```

2. Build and view: from your terminal, run

   ```
   make html && <your_browser> _build/html/index.html
   ```

## Adding your own tutorials

If you have your own tutorial in the form of a Jupyter notebook and you'd like
to try it out on the site:

1. Add your notebook to the `content/` directory
2. Update `content/requirements.txt` with the dependencies for your tutorial
3. Update the `toctree` in `index.rst` to include your new entry
4. Update the attribution section (below) to credit the original tutorial 
   author.

## Attribution

 - The [cs231n][cs231] tutorial is by [@jcjohnson][jj]. The full tutorial in 
   its original form is linked via [numpy.org][learn].
 - The SVD tutorial is by [@melissawm][mwm]. The full tutorial is available
   via the [tutorials page][np_tutorials] of the official NumPy documentation.

[jj]: https://github.com/jcjohnson
[mwm]: https://github.com/melissawm
[np_tutorials]: https://numpy.org/devdocs/user/tutorials_index.html

## Useful links

The following may be useful:

- [NumPy documentation team meetings](https://hackmd.io/oB_boakvRqKR-_2jRV-Qjg?both)
- [NEP 44 - Restructuring the NumPy documentation](https://numpy.org/neps/nep-0044-restructuring-numpy-docs.html)
- [Blog post - Documentation as a way to build Community](https://labs.quansight.org/blog/2020/03/documentation-as-a-way-to-build-community/)
- Note that regular documentation issues for NumPy can be found in the
  [main NumPy repository](https://github.com/numpy/numpy/issues) (see the
  `Documentation` labels (2x) there)

