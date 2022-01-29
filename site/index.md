# NumPy tutorials

[![Binder](https://mybinder.org/badge_logo.svg)][launch_binder]

[launch_binder]: https://mybinder.org/v2/gh/numpy/numpy-tutorials/main?urlpath=lab/tree/content

This set of tutorials and educational materials is being developed in the
[numpy-tutorials](https://github.com/numpy/numpy-tutorials) repository, and is
not a part of the NumPy source tree. The goal of this repository is to provide
high-quality resources by the NumPy project, both for self-learning and for
teaching classes with. If you're interested in adding your own content, check
the [Contributing](contributing.md) section.

To open a live version of the content, click the **launch Binder** button above.
To open each of the `.md` files, right click and select "Open with -> Notebook".
You can also launch individual tutorials on Binder by clicking on the rocket
icon that appears in the upper-right corner of each tutorial. To download a
local copy of the `.ipynb` files, you can either
[clone this repository](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository)
or use the download icon in the upper-right corner of each tutorial.

```{toctree}
---
hidden: true
---

features
applications
contributing
```

## NumPy Features

````{panels}

{doc}`content/tutorial-svd`
^^^^^^^^^^^^^^^^^^^^^^^^^^^

```{glue:} thumb_svd
```
---

{doc}`content/tutorial-ma`
^^^^^^^^^^^^^^^^^^^^^^^^^^

```{glue:} thumb_ma
```

---

{doc}`content/save-load-arrays`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

![Default thumbnail: NumPy logo](_static/numpylogo.svg)
````

## NumPy Applications

````{panels}

{doc}`content/mooreslaw-tutorial`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

```{glue:} thumb_mooreslaw
```

---

{doc}`content/tutorial-deep-learning-on-mnist`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

```{glue:} thumb_mnist
```

---

{doc}`content/tutorial-deep-reinforcement-learning-with-pong-from-pixels`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

![Diagram showing the component operations of reinforcement learning detailed
in this tutorial](content/_static/tutorial-deep-reinforcement-learning-with-pong-from-pixels.png)

---

{doc}`content/tutorial-nlp-from-scratch`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

![Overview of the model architecture, showing a series of animated boxes.
There are five identical boxes labeled A and receiving as input one of the
words in the phrase "life's a box of chocolates". Each box is highlighted in
turn, representing the memory blocks of the LSTM network as information passes
through them, ultimately reaching a "Positive" output value.](content/_static/lstm.gif)

---

{doc}`content/tutorial-x-ray-image-processing`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

```{glue:} thumb_xray
```

---

{doc}`content/tutorial-static_equilibrium`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

```{glue:} thumb_static_eq
```

---

{doc}`content/tutorial-plotting-fractals`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

![An example of a fractal visualization from this tutorial](content/_static/fractal.png)

---

{doc}`content/tutorial-air-quality-analysis`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

![A grid showing the India Gate in smog above and clear air below](content/_static/11-delhi-aqi.jpg)

````

## Useful links and resources

The following links may be useful:

- [NumPy Code of Conduct](https://numpy.org/code-of-conduct/)
- [Main NumPy documentation](https://numpy.org/doc/stable/)
- [NumPy documentation team meeting notes](https://hackmd.io/oB_boakvRqKR-_2jRV-Qjg?both)
- [NEP 44 - Restructuring the NumPy documentation](https://numpy.org/neps/nep-0044-restructuring-numpy-docs.html)
- [Blog post - Documentation as a way to build Community](https://labs.quansight.org/blog/2020/03/documentation-as-a-way-to-build-community/)

Note that regular documentation issues for NumPy can be found in the [main NumPy
repository](https://github.com/numpy/numpy/issues) (see the `Documentation`
labels there).
