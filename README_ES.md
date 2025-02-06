# Tutoriales de NumPy

_Para ver los tutoriales renderizados, visita https://numpy.org/numpy-tutorials/_.

El objetivo de este repositorio es proporcionar recursos de alta calidad por parte del
proyecto NumPy, tanto para el autoaprendizaje como para la enseñanza en clases. Si estás
interesado en agregar tu propio contenido, consulta la sección [Contribuir](#contribuir).
Este conjunto de tutoriales y materiales educativos no forma parte del árbol de fuentes de NumPy.

Para descargar una copia local de los archivos `.ipynb`, puedes
[clonar este repositorio](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository)
o navegar a cualquiera de los documentos enumerados a continuación y descargarlos individualmente.

## Contenido

0. [Aprende a escribir un tutorial de NumPy](content/tutorial-style-guide.md): nuestra guía de estilo para escribir tutoriales.
1. [Tutorial: Álgebra lineal en matrices n-dimensionales](content/tutorial-svd.md)
2. [Tutorial: Determinando la Ley de Moore con datos reales en NumPy](content/mooreslaw-tutorial.md)
3. [Tutorial: Guardando y compartiendo tus matrices NumPy](content/save-load-arrays.md)
4. [Tutorial: Aprendizaje profundo con NumPy en MNIST desde cero](content/tutorial-deep-learning-on-mnist.md)
5. [Tutorial: Procesamiento de imágenes de rayos X](content/tutorial-x-ray-image-processing.md)
6. [Tutorial: Aprendizaje profundo de refuerzo con NumPy y Pong desde píxeles](content/tutorial-deep-reinforcement-learning-with-pong-from-pixels.md)
7. [Tutorial: Arrays enmascarados](content/tutorial-ma.md)
8. [Tutorial: Equilibrio estático](content/tutorial-static_equilibrium.md)
9. [Tutorial: Trazado de fractales](content/tutorial-plotting-fractals.ipynb)
10. [Tutorial: Procesamiento de lenguaje natural con NumPy desde cero con un enfoque en ética](content/tutorial-nlp-from-scratch.md)
11. [Tutorial: Análisis del impacto del confinamiento en la calidad del aire en Delhi, India](content/tutorial-air-quality-analysis.md)

## Contribuir

¡Damos una muy calurosa bienvenida a las contribuciones! Si tienes una idea o propuesta para un nuevo
tutorial, por favor [abre un issue](https://github.com/numpy/numpy-tutorials/issues)
con un esquema.

No te preocupes si el inglés no es tu primer idioma, o si solo puedes hacer un
borrador aproximado. El código abierto es un esfuerzo comunitario. Haz tu mejor esfuerzo, nosotros
ayudaremos a corregir los problemas.

Las imágenes y los datos de la vida real hacen que el texto sea más atractivo y poderoso, pero asegúrate de que lo que
utilices tenga la licencia apropiada y esté disponible. Nuevamente, incluso una idea aproximada
de arte puede ser mejorada por otros.

Los tutoriales de NumPy son una colección curada de
notebooks [MyST-NB](https://myst-nb.readthedocs.io/). Estos notebooks se utilizan
para producir sitios web estáticos y pueden abrirse como notebooks en Jupyter usando
[Jupytext](https://jupytext.readthedocs.io).

> __Nota:__ Deberías usar celdas en markdown [CommonMark](https://commonmark.org).
> Jupyter solo renderiza CommonMark.

### ¿Por qué Jupyter Notebooks?

La elección de Jupyter Notebook en este repositorio en lugar del formato usual
([reStructuredText, a través de Sphinx](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html))
utilizado en la documentación principal de NumPy tiene dos razones:

  * Los notebooks de Jupyter son un formato común para comunicar información científica.
  * Los notebooks de Jupyter pueden ser lanzados en [Binder](https://www.mybinder.org), para que los usuarios puedan interactuar
    con los tutoriales.
  * rST puede representar una barrera para algunas personas que podrían estar muy
    interesadas en contribuir con material tutorial.

#### Nota

Puedes notar que nuestro contenido está en formato markdown (archivos `.md`). Revisamos y
alojamos notebooks en el formato [MyST-NB](https://myst-nb.readthedocs.io/). Aceptamos tanto notebooks de Jupyter (`.ipynb`) como notebooks MyST-NB (`.md`). Si deseas
sincronizar tu `.ipynb` con tu archivo `.md`, sigue el [tutorial de emparejamiento](content/pairing.md).

### Agregar tus propios tutoriales

Si tienes tu propio tutorial en forma de un notebook de Jupyter (un archivo `.ipynb`)
y deseas agregarlo al repositorio, sigue los pasos a continuación.

#### Crear un issue

Ve a [https://github.com/numpy/numpy-tutorials/issues](https://github.com/numpy/numpy-tutorials/issues)
y crea un nuevo issue con tu propuesta. Proporciona tanta información como puedas sobre
el tipo de contenido que te gustaría escribir (tutorial, cómo hacer) y lo que
planeas cubrir. Intentaremos responder lo más rápido posible con comentarios, si
es aplicable.

#### Consulta nuestra plantilla sugerida

Puedes usar nuestra [Guía de Estilo de Tutoriales](content/tutorial-style-guide.md) para hacer
tu contenido consistente con nuestros tutoriales existentes.

#### Sube tu contenido

<ul>
<details>
    <summary>
        <b>Haz un fork de este repositorio</b> (si no lo has hecho antes).
    </summary>
    <img src="site/_static/01-fork.gif" width=80% height=80%>
</details>

<details>
    <summary>
        <b>En tu propio fork, crea una nueva rama para tu contenido.</b>
    </summary>
    <img src="site/_static/02-create_new_branch.gif" width=80% height=80%>
</details>

<details>
    <summary>
        <b>Agrega tu notebook al directorio <code>content/</code>.</b>
    </summary>
    <img src="site/_static/03-upload.gif" width=80% height=80%>
</details>

<b>Actualiza el archivo <code>environment.yml</code> con las dependencias para tu
tutorial</b> (solo si agregas nuevas dependencias).

<details>
    <summary>
        <b>Actualiza este <code>README.md</code> para incluir tu nueva entrada.</b>
    </summary>
    <img src="site/_static/04-add_to_readme.gif" width=80% height=80%>
</details>

<b>Actualiza la sección de atribución (a continuación) para acreditar al autor original del tutorial,
si es aplicable.</b>

<details>
    <summary>
        <b>Crea un <a href="https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests">pull request</a>.
        Asegúrate de que la opción "Permitir ediciones y acceso a secretos por parte de los mantenedores"
        esté seleccionada para que podamos revisar adecuadamente tu envío.</b>
    </summary>
    <img src="site/_static/05-create_PR.gif" width=80% height=80%>
</details>

:tada: <b>¡Espera la revisión!</b>
</ul>

Para más información sobre GitHub y su flujo de trabajo, puedes ver
[este documento](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests).

### Construyendo el sitio Sphinx localmente

Construir el sitio web de tutoriales, que se publica en
https://github.com/numpy/numpy-tutorials, localmente no es necesario antes de hacer
una contribución, pero puede ser útil:

```bash
conda env create -f environment.yml
conda activate numpy-tutorials
cd site
make html

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

