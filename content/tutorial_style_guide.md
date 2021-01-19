---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.6.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Learn to write a NumPy tutorial

![image](https://documentation.divio.com/_images/overview.png)
<p style="text-align:right;font-style:italic;">Image credit: Daniele Procida's <a href="https://documentation.divio.com/">The documentation system</a></p>

+++ 

## What you'll do

Guided by a template, you'll write a NumPy tutorial.

## What you'll learn

- You'll be able to craft a tutorial that follows a standard format and reflects good teaching practice.

- You'll learn the three standard headings that open a NumPy tutorial -- **What you'll do,** **What you'll learn,** and **What you'll need** -- and some optional headings for the bottom -- **On your own,** **In practice,** **Further reading.**

- You'll know what makes **What you'll learn** different from **What you'll do.**

- You'll be able to distinguish a **tutorial** from a **how-to**.

- You'll learn what not to put in a **What you'll learn** section.

## What you'll need

- This template.

- A portrait of your intended reader. 
    - Just as schools list prerequisites for higher-level courses, you can assume readers know some things (which you must list, as noted in the next bullet). Overexplaining bogs down the tutorial and obscures the main points.
    - But also put yourself in the reader's place and consider what to explain along the way.
    
    
- "What you'll need" is a list of:

    - packages that must be present on the user's machine before they begin. Don't include `numpy`.
    - what you assumed the reader knew in the bullet above. Don't say `Python`;  `familiarity with Python iterators` is fine.


- Informality and enthusiasm. Imagine your reader not out in the audience but next to you.

- Willingness to write incomplete sentences for the **What you'll need** bullets. They don't begin with the words "You'll need."

- **Not** required are native English skills. Others can help.


***

## After a horizontal rule, start your own headings

Your tutorial steps begin here, using headings of your choice. At the end of the tutorial you'll place another horizontal rule and return to standard headings.


## Titles have verbs

In general, include a verb in the title; thus **Learn to write a NumPy tutorial** rather than "Rules for NumPy tutorials." Consider putting verbs in the headings as well.


## Titles are lowercase

Capitalize the first word, and after that only words that are ordinarily capitalized (so not "Titles Are Lowercase").


## What to say in "What you'll learn"

Avoid abstraction. "About" is a tipoff: Rather than writing "You'll learn about NumPy I/O," write "You'll learn how to read a comma-delimited text file into a NumPy array."


## Why are "What you'll do" and "What you'll learn" different?

**What you'll do** is typically one sentence listing an end product: "You'll bake a cake." This makes the endpoint clear. **What you'll learn** lists the payoffs, and there may be many: "You'll learn to follow a recipe. You'll get practice measuring ingredients. You'll learn how to tell when a cake is ready to come out of the oven."  


## Avoid asides

As explained by master documentation writer [Daniele Procida](https://documentation.divio.com/tutorials):

> Don’t explain anything the learner doesn’t need to know in order to complete the tutorial. 

Because tutorial steps are chosen to be clear and easy, they may fall short of
production-grade. Yes, you should share this, but not during the tutorial, which should be straightforward and assured. The `In practice` section is the place for details, exceptions, alternatives, and similar fine print.


## Use plots and illustrations

Figures are a double win; they amplify your points and make the page inviting.  Like English skills, artistic skills (or graphic-toolset skills) aren't required. Even if you only scan a hand illustration, somebody can polish it.

An illustration below the title, even if it's only decorative, makes your page distinctive.


## Use real datasets when possible

Readers are likelier to be engaged by a real use case. Be sure you have rights to the data.


## Tutorials and how-to's  -- similar but different

Tutorial readers are out-of-towners who want a feel for the place. Pick any single destination and explain sights along the way.

Unlike how-to readers, who know what they need, tutorial readers don't know what it is they don't know. So while tutorials need headings like **What you'll do** and **What you'll learn**, these headings would never appear in a how-to.

## Make use of the Google doc style guide

NumPy docs follow the [Google developer documentation style guide](https://developers.google.com/style/). In addition to providing answers to recurring questions ("crossreference" or "cross-reference"?) the guide is filled with suggestions that will strengthen your doc writing.

## The notebook must be fully executable

`Run all cells` should execute all cells to the bottom of the file. If you're demonstrating a bad expression and want to show the traceback, comment
the expression and put the traceback in a text cell.

(Note that triple backquotes won't be enough for a traceback that contains `<text inside angle brackets>`, 
the angle brackets must be replaced by `&lt;` and `&gt;` as shown in the text cell markdown below.)

```{code-cell} ipython3
#100/0
```

<div style="background-color:#fcdcdc">
<code style="background-color:#fcdcdc;font-size:90%">    
---------------------------------------------------------------------------
ZeroDivisionError                         Traceback (most recent call last)
&lt;ipython-input-10-bbe761e74a70&gt; in &lt;module&gt;
----> 1 100/0

ZeroDivisionError: division by zero
</code>
</div>

+++ 

***

## On your own

Close the tutorial section with a horizontal rule. You're free to take any direction now, but here are three suggested sections.

In an optional `On your own` section, you can offer an assignment for readers to exercise their new skills. If it's a question with an answer, provide it -- perhaps in a footnote to keep it from being a spoiler.

## In practice...

- The fine print that you avoided can go in this section.



- Don't just say it's usually done another way; explain why. 
 

## Further reading

- Ideally, rather than giving bare links, **Further reading** describes the references: [The Documentation System](https://documentation.divio.com/) is the inspiration for this tutorial, and describes three other kinds of documentation.
- The Google guide is long; there's also [a summary](https://developers.google.com/style/highlights?hl=pt-br).
- NumPy's website includes a [documentation how-to](https://numpy.org/devdocs/dev/howto-docs.html).
