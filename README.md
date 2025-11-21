# NumPy tutorials

_For the rendered tutorials, see https://numpy.org/numpy-tutorials/._

The goal of this repository is to provide high-quality resources by the
NumPy project, both for self-learning and for teaching classes with. If you're
interested in adding your own content, check the [Contributing](#contributing)
section. This set of tutorials and educational materials is not a part of the
NumPy source tree.

To download a local copy of the `.ipynb` files, you can either
[clone this repository](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository)
or navigate to any of the documents listed below and download it individually.

## Content

0. [Learn to write a NumPy tutorial](content/tutorial-style-guide.md): our style guide for writing tutorials.
1. [Tutorial: Linear algebra on n-dimensional arrays](content/tutorial-svd.md)
2. [Tutorial: Determining Moore's Law with real data in NumPy](content/mooreslaw-tutorial.md)
3. [Tutorial: Saving and sharing your NumPy arrays](content/save-load-arrays.md)
4. [Tutorial: NumPy deep learning on MNIST from scratch](content/tutorial-deep-learning-on-mnist.md)
5. [Tutorial: X-ray image processing](content/tutorial-x-ray-image-processing.md)
6. [Tutorial: Masked Arrays](content/tutorial-ma.md)
7. [Tutorial: Static Equilibrium](content/tutorial-static_equilibrium.md)
8. [Tutorial: Plotting Fractals](content/tutorial-plotting-fractals.ipynb)
9. [Tutorial: Analysing the impact of the lockdown on air quality in Delhi, India](content/tutorial-air-quality-analysis.md)


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

### Building the website

```{note}
The NumPy tutorials are powered by [`jupyter-book`][jb-docs] and the
[`MyST` document engine][mystmd].
See the linked documentation for further details.
```

[jb-docs]: https://jupyterbook.org/stable/
[mystmd]:

#### Quickstart

Set up a development environment with the dependencies listed in
`requirements.txt` and `site/requirements.txt`.
For example, using the built-in [`venv`][venv] module:

```bash
python -m venv np-tutorials
source np-tutorials/bin/activate
python -m pip install -r requirements.txt -r site/requirements.txt
```

[venv]: https://docs.python.org/3/library/venv.html

The site can then be built with:

```bash
jupyter-book start --execute
```

This will execute all the notebooks and start a web server to view the rendered
content locally.
View the rendered site by opening the ``localhost:3000`` in your preferred browser.

### Adding your own tutorials

If you have your own tutorial in the form of a Jupyter notebook (a `.ipynb`
file) and you'd like to add it to the repository, follow the steps below.


#### Create an issue

Go to [https://github.com/numpy/numpy-tutorials/issues](https://github.com/numpy/numpy-tutorials/issues)
and create a new issue with your proposal. Give as much detail as you can about
what kind of content you would like to write (tutorial, how-to) and what you
plan to cover. We will try to respond as quickly as possible with comments, if
applicable.

#### Check out our suggested template

You can use our [Tutorial Style Guide](content/tutorial-style-guide.md) to make
your content consistent with our existing tutorials.

#### Upload your content

<ul>
<details>
    <summary>
        <b>Fork this repository</b> (if you haven't before).
    </summary>
    <img src="site/_static/01-fork.gif" width=80% height=80%>
</details>

<details>
    <summary>
        <b>In your own fork, create a new branch for your content.</b>
    </summary>
    <img src="site/_static/02-create_new_branch.gif" width=80% height=80%>
</details>

<details>
    <summary>
        <b>Add your notebook to the <code>content/</code> directory.</b>
    </summary>
    <img src="site/_static/03-upload.gif" width=80% height=80%>
</details>

<b>Update the <code>environment.yml</code> file with the dependencies for your
tutorial</b> (only if you add new dependencies).

<details>
    <summary>
        <b>Update this <code>README.md</code> to include your new entry.</b>
    </summary>
    <img src="site/_static/04-add_to_readme.gif" width=80% height=80%>
</details>

<b>Update the attribution section (below) to credit the original tutorial
author, if applicable.</b>

<details>
    <summary>
        <b>Create a <a href="https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests">pull request.</a>
        Make sure the "Allow edits and access to secrets by maintainers" option
        is selected so we can properly review your submission.</b>
    </summary>
    <img src="site/_static/05-create_PR.gif" width=80% height=80%>
</details>

:tada: <b>Wait for review!</b>
</ul>

For more information about GitHub and its workflow, you can see
[this document](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests).


### Building the Sphinx site locally

Building the tutorials website, which is published at
https://github.com/numpy/numpy-tutorials, locally isn't necessary before making
a contribution, but may be helpful:

```bash
conda env create -f environment.yml
conda activate numpy-tutorials
cd site
make html
```

## Translations

While we don't have the capacity to translate and maintain translated versions
of these tutorials, you are free to use and translate them to other languages.

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

