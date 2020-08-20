# NumPy tutorials

[![Binder](http://mybinder.org/badge_logo.svg)](http://mybinder.org/v2/gh/numpy/numpy-tutorials/master?urlpath=lab/tree/content)
[![Gitpod ready-to-code](https://img.shields.io/badge/Gitpod-ready--to--code-blue?logo=gitpod)](https://gitpod.io/#https://github.com/numpy/numpy-tutorials)

This set of tutorials and educational materials is being developed,
IT IS NOT INTEGRATED IN THE HTML DOCS AT https://www.numpy.org/devdocs/

The goal of this repository is to provide high-quality resources by the
NumPy project, both for self-learning and for teaching classes with. If you're
interested in adding your own content, check the [Contributing](#contributing)
section.

To open a live version of the content, click the **launch Binder** button above.
To download a local copy of the `.ipynb` files, you can either
[clone this repository](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository)
or navigate to any of the documents listed below and download it individually.

## Content

1. [Tutorial: Linear algebra on n-dimensional arrays](content/tutorial-svd.ipynb)
2. [Tutorial: CS231n Python Tutorial With Google Colab](content/cs231_tutorial.ipynb)

## Contributing

We very much welcome contributions! If you have an idea or proposal for a new
tutorial, please [open an issue](https://github.com/numpy/numpy-tutorials/issues)
with an outline. 

Don’t worry if English is not your first language, or if you can only come up
with a rough draft. Open source is a community effort. Do your best – we’ll help
fix issues.

Images and real-life data make text more engaging and powerful, but be sure what
you use is appropriately licensed and available. Here again, even a rough idea
for artwork can be polished by others.

### Why Jupyter Notebooks?

The choice of Jupyter Notebook in this repo instead of the usual format 
([reStructuredText, through Sphinx](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html))
used in the main NumPy documentation has two reasons:

 * Jupyter notebooks are a common format for communicating scientific
   information.
 * rST may present a barrier for some people who might otherwise be very
   interested in contributing tutorial material.

### Adding your own tutorials

If you have your own tutorial in the form of a Jupyter notebook (a `.ipynb`
file) and you'd like to add it to the repository:

#### Create an issue

Go to [https://github.com/numpy/numpy-tutorials/issues](https://github.com/numpy/numpy-tutorials/issues) and create a new issue with your proposal. Give as much detail as you can about what kind of content you would like to write (tutorial, how-to) and what you plan to cover. We will try to respond as quickly as possible with comments, if applicable.

Next, after writing the document, you can add it to this repository by following **one of the following** guides.

#### Using Gitpod

This workflow lets you add your content using a cloud workspace, and is recommended if you have less experience with git.

1. Click the **Gitpod** link in your issue, if it exists. If not, you can prepend `https://gitpod.io/#` to the issue URL. This will open a complete workspace in your browser, allowing you to do all tasks related to adding your content in the same place. Using this method also creates an automatic branch for your new content, which is preferrable so we can add it to the main branch after the Pull Request is reviewed and accepted.
2. In the Gitpod menu, select the `content` directory by clicking on in. 
3. In the top menu, go to **File -> Upload Files** and select your `.ipynb` file to upload.
4. Once this is done, remember to update this `README.md` file to include your new entry in the **Content** section.
5. Update the attribution section (below) to credit the original content author.
6. Save your work by going to **File -> Save All**
7. Commit your changes by clicking on the **Source control** icon on the left panel:
    - Adding a commit message to the text box
    - Click on the *plus* sign next to each file that was changed
    - Click the "Commit" checkmark.
8. Submit your [pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests) by clicking on the "Pull Request" button on the right of the Gitpod interface. There may be a message about your branch not existing yet - that's normal and you can continue with your pull request anyway.
9. Wait for review!

#### Using git locally
1. Clone this repository.
2. In your terminal, create a new environment (we recommend using conda, and doing `conda env create -f environment.yml` from the root of the repository files. This will automatically install all necessary dependencies to run all notebooks in the repo.)
3. Create a new branch and add your notebook to the `content/` directory in this new branch.
4. Update the `environment.yml` file with the dependencies for your tutorial
(only if you add new dependencies)
5. Update this `README.md` file to include your new entry in the **Content** section.
6. Update the attribution section (below) to credit the original content
author.
7. Submit your [pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests) and wait for review.

For more information about GitHub and its workflow, you can see
[this document](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests).

### Attribution

 - The cs231n tutorial is by [@jcjohnson][jj]. The full tutorial in 
   its original form is linked via [numpy.org][learn].
 - The SVD tutorial is by [@melissawm][mwm]. The full tutorial is available
   via the [tutorials page][np_tutorials] of the official NumPy documentation.

[jj]: https://github.com/jcjohnson
[mwm]: https://github.com/melissawm
[np_tutorials]: https://numpy.org/devdocs/user/tutorials_index.html

## Useful links and resources

The following links may be useful:

- [NumPy Code of Conduct](https://numpy.org/doc/stable/dev/conduct/code_of_conduct.html)
- [Main NumPy documentation](https://numpy.org/doc/stable/)
- [NumPy documentation team meeting notes](https://hackmd.io/oB_boakvRqKR-_2jRV-Qjg?both)
- [NEP 44 - Restructuring the NumPy documentation](https://numpy.org/neps/nep-0044-restructuring-numpy-docs.html)
- [Blog post - Documentation as a way to build Community](https://labs.quansight.org/blog/2020/03/documentation-as-a-way-to-build-community/)

Note that regular documentation issues for NumPy can be found in the [main NumPy
repository](https://github.com/numpy/numpy/issues) (see the `Documentation`
labels there). 
