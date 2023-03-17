# Contributing

We very much welcome contributions! If you have an idea or proposal for a new
tutorial, please [open an issue](https://github.com/numpy/numpy-tutorials/issues)
with an outline.

Don’t worry if English is not your first language, or if you can only come up
with a rough draft. Open source is a community effort. Do your best – we’ll help
fix issues.

Images and real-life data make text more engaging and powerful, but be sure what
you use is appropriately licensed and available. Here again, even a rough idea
for artwork can be polished by others.

The NumPy tutorials are a curated collection of
[MyST-NB](https://myst-nb.readthedocs.io/) notebooks. These notebooks are used
to produce static websites and can be opened as notebooks in Jupyter using
[Jupytext](https://jupytext.readthedocs.io).

> __Note:__ You should use [CommonMark](https://commonmark.org) markdown
> cells. Jupyter only renders CommonMark.

## Why Jupyter Notebooks?

The choice of Jupyter Notebook in this repo instead of the usual format
([reStructuredText][rst])
used in the main NumPy documentation has two reasons:


 * Jupyter notebooks are a common format for communicating scientific
   information.
 * Jupyter notebooks can be launched in [Binder](https://www.mybinder.org), so that users can interact
   with tutorials
 * rST may present a barrier for some people who might otherwise be very
   interested in contributing tutorial material.

[rst]: https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html

### Note

You may notice our content is in markdown format (`.md` files). We review and
host notebooks in the [MyST-NB](https://myst-nb.readthedocs.io/) format. We
accept both Jupyter notebooks (`.ipynb`) and MyST-NB notebooks (`.md`).
If you want to sync your `.ipynb` to your `.md` file follow the [pairing
tutorial](content/pairing.md).

```{toctree}
:hidden:

content/pairing
```

## Adding your own tutorials

If you have your own tutorial in the form of a Jupyter notebook (an `.ipynb`
file) and you'd like to try add it out to the repository, follow the steps below.

### Create an issue

Go to <https://github.com/numpy/numpy-tutorials/issues> and create a new issue
with your proposal.
Give as much detail as you can about what kind of content you would like to
write (tutorial, how-to) and what you plan to cover.
We will try to respond as quickly as possible with comments, if applicable.

### Check out our suggested template

You can use this template to make your content consistent with our existing
tutorials:

```{toctree}
---
maxdepth: 1
---
content/tutorial-style-guide
```

### Upload your content

Remember to clear all outputs on your notebook before uploading it.

<ul>
<details>
    <summary>
        <b>Fork this repository</b> (if you haven't before).
    </summary>
    <img src="_static/01-fork.gif" width=80% height=80%>
</details>

<details>
    <summary>
        <b>In your own fork, create a new branch for your content.</b>
    </summary>
    <img src="_static/02-create_new_branch.gif" width=80% height=80%>
</details>

<details>
    <summary>
        <b>Add your notebook to the <code>content/</code> directory.</b>
    </summary>
    <img src="_static/03-upload.gif" width=80% height=80%>
</details>

<b>Update the <code>environment.yml</code> file with the dependencies for your tutorial</b>
(only if you add new dependencies).

<details>
    <summary>
        <b>Update this <code>README.md</code> to include your new entry.</b>
    </summary>
    <img src="_static/04-add_to_readme.gif" width=80% height=80%>
</details>

<details>
    <summary>
        <b>Create a <a href="https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests">pull request.</a> Make sure the "Allow edits and access to secrets by maintainers" option is selected so we can properly review your submission.</b>
    </summary>
    <img src="_static/05-create_PR.gif" width=80% height=80%>
</details>

🎉 <b>Wait for review!</b>
</ul>

For more information about GitHub and its workflow, you can see
[this document][collab].

[collab]: https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests
