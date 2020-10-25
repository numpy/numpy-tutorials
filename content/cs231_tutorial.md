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

+++ {"colab_type": "text", "id": "dzNng6vCL9eP"}

# CS231n Python Tutorial With Google Colab

+++ {"colab_type": "text", "id": "0vJLt3JRL9eR"}

This tutorial was originally written by [Justin Johnson](https://web.eecs.umich.edu/~justincj/) for cs231n. It was adapted as a Jupyter notebook for cs228 by [Volodymyr Kuleshov](http://web.stanford.edu/~kuleshov/) and [Isaac Caswell](https://symsys.stanford.edu/viewing/symsysaffiliate/21335).

This version has been adapted for Colab by Kevin Zakka for the Spring 2020 edition of [cs231n](https://cs231n.github.io/). It runs Python3 by default.

+++ {"colab_type": "text", "id": "qVrTo-LhL9eS"}

## Introduction

+++ {"colab_type": "text", "id": "9t1gKp9PL9eV"}

Python is a great general-purpose programming language on its own, but with the help of a few popular libraries (numpy, scipy, matplotlib) it becomes a powerful environment for scientific computing.

We expect that many of you will have some experience with Python and numpy; for the rest of you, this section will serve as a quick crash course both on the Python programming language and on the use of Python for scientific computing.

Some of you may have previous knowledge in Matlab, in which case we also recommend the numpy for Matlab users page (https://docs.scipy.org/doc/numpy-dev/user/numpy-for-matlab-users.html).

+++ {"colab_type": "text", "id": "U1PvreR9L9eW"}

In this tutorial, we will cover:

* Basic Python: Basic data types (Containers, Lists, Dictionaries, Sets, Tuples), Functions, Classes
* Numpy: Arrays, Array indexing, Datatypes, Array math, Broadcasting
* Matplotlib: Plotting, Subplots, Images
* IPython: Creating notebooks, Typical workflows

+++ {"colab_type": "text", "id": "nxvEkGXPM3Xh"}

## A Brief Note on Python Versions

As of Janurary 1, 2020, Python has [officially dropped support](https://www.python.org/doc/sunset-python-2/) for `python2`. We'll be using Python 3.7 for this iteration of the course. You can check your Python version at the command line by running `python --version`. In Colab, we can enforce the Python version by clicking `Runtime -> Change Runtime Type` and selecting `python3`. Note that as of April 2020, Colab uses Python 3.6.9 which should run everything without any errors.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 34
colab_type: code
id: 1L4Am0QATgOc
outputId: bb5ee3ac-8683-44ab-e599-a2077510f327
---
!python --version
```

+++ {"colab_type": "text", "id": "JAFKYgrpL9eY"}

## Basics of Python

+++ {"colab_type": "text", "id": "RbFS6tdgL9ea"}

Python is a high-level, dynamically typed multiparadigm programming language. Python code is often said to be almost like pseudocode, since it allows you to express very powerful ideas in very few lines of code while being very readable. As an example, here is an implementation of the classic quicksort algorithm in Python:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 34
colab_type: code
id: cYb0pjh1L9eb
outputId: 9a8e37de-1dc1-4092-faee-06ad4ff2d73a
---
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print(quicksort([3,6,8,10,1,2,1]))
```

+++ {"colab_type": "text", "id": "NwS_hu4xL9eo"}

### Basic data types

+++ {"colab_type": "text", "id": "DL5sMSZ9L9eq"}

#### Numbers

+++ {"colab_type": "text", "id": "MGS0XEWoL9er"}

Integers and floats work as you would expect from other languages:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 52
colab_type: code
id: KheDr_zDL9es
outputId: 1db9f4d3-2e0d-4008-f78a-161ed52c4359
---
x = 3
print(x, type(x))
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 86
colab_type: code
id: sk_8DFcuL9ey
outputId: dd60a271-3457-465d-e16a-41acf12a56ab
---
print(x + 1)   # Addition
print(x - 1)   # Subtraction
print(x * 2)   # Multiplication
print(x ** 2)  # Exponentiation
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 52
colab_type: code
id: U4Jl8K0tL9e4
outputId: 07e3db14-3781-42b7-8ba6-042b3f9f72ba
---
x += 1
print(x)
x *= 2
print(x)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 52
colab_type: code
id: w-nZ0Sg_L9e9
outputId: 3aa579f8-9540-46ef-935e-be887781ecb4
---
y = 2.5
print(type(y))
print(y, y + 1, y * 2, y ** 2)
```

+++ {"colab_type": "text", "id": "r2A9ApyaL9fB"}

Note that unlike many languages, Python does not have unary increment (x++) or decrement (x--) operators.

Python also has built-in types for long integers and complex numbers; you can find all of the details in the [documentation](https://docs.python.org/3.7/library/stdtypes.html#numeric-types-int-float-long-complex).

+++ {"colab_type": "text", "id": "EqRS7qhBL9fC"}

#### Booleans

+++ {"colab_type": "text", "id": "Nv_LIVOJL9fD"}

Python implements all of the usual operators for Boolean logic, but uses English words rather than symbols (`&&`, `||`, etc.):

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 34
colab_type: code
id: RvoImwgGL9fE
outputId: 1517077b-edca-463f-857b-6a8c386cd387
---
t, f = True, False
print(type(t))
```

+++ {"colab_type": "text", "id": "YQgmQfOgL9fI"}

Now we let's look at the operations:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 86
colab_type: code
id: 6zYm7WzCL9fK
outputId: f3cebe76-5af4-473a-8127-88a1fd60560f
---
print(t and f) # Logical AND;
print(t or f)  # Logical OR;
print(not t)   # Logical NOT;
print(t != f)  # Logical XOR;
```

+++ {"colab_type": "text", "id": "UQnQWFEyL9fP"}

#### Strings

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 34
colab_type: code
id: AijEDtPFL9fP
outputId: 2a6b0cd7-58f1-43cf-e6b7-bf940d532549
---
hello = 'hello'   # String literals can use single quotes
world = "world"   # or double quotes; it does not matter
print(hello, len(hello))
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 34
colab_type: code
id: saDeaA7hL9fT
outputId: 2837d0ab-9ae5-4053-d087-bfa0af81c344
---
hw = hello + ' ' + world  # String concatenation
print(hw)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 34
colab_type: code
id: Nji1_UjYL9fY
outputId: 0149b0ca-425a-4a34-8e24-8dff7080922e
---
hw12 = '{} {} {}'.format(hello, world, 12)  # string formatting
print(hw12)
```

+++ {"colab_type": "text", "id": "bUpl35bIL9fc"}

String objects have a bunch of useful methods; for example:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 121
colab_type: code
id: VOxGatlsL9fd
outputId: ab009df3-8643-4d3e-f85f-a813b70db9cb
---
s = "hello"
print(s.capitalize())  # Capitalize a string
print(s.upper())       # Convert a string to uppercase; prints "HELLO"
print(s.rjust(7))      # Right-justify a string, padding with spaces
print(s.center(7))     # Center a string, padding with spaces
print(s.replace('l', '(ell)'))  # Replace all instances of one substring with another
print('  world '.strip())  # Strip leading and trailing whitespace
```

+++ {"colab_type": "text", "id": "06cayXLtL9fi"}

You can find a list of all string methods in the [documentation](https://docs.python.org/3.7/library/stdtypes.html#string-methods).

+++ {"colab_type": "text", "id": "p-6hClFjL9fk"}

### Containers

+++ {"colab_type": "text", "id": "FD9H18eQL9fk"}

Python includes several built-in container types: lists, dictionaries, sets, and tuples.

+++ {"colab_type": "text", "id": "UsIWOe0LL9fn"}

#### Lists

+++ {"colab_type": "text", "id": "wzxX7rgWL9fn"}

A list is the Python equivalent of an array, but is resizeable and can contain elements of different types:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 52
colab_type: code
id: hk3A8pPcL9fp
outputId: b545939a-580c-4356-db95-7ad3670b46e4
---
xs = [3, 1, 2]   # Create a list
print(xs, xs[2])
print(xs[-1])     # Negative indices count from the end of the list; prints "2"
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 34
colab_type: code
id: YCjCy_0_L9ft
outputId: 417c54ff-170b-4372-9099-0f756f8e48af
---
xs[2] = 'foo'    # Lists can contain elements of different types
print(xs)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 34
colab_type: code
id: vJ0x5cF-L9fx
outputId: a97731a3-70e1-4553-d9e0-2aea227cac80
---
xs.append('bar') # Add a new element to the end of the list
print(xs)  
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 34
colab_type: code
id: cxVCNRTNL9f1
outputId: 508fbe59-20aa-48b5-a1b2-f90363e7a104
---
x = xs.pop()     # Remove and return the last element of the list
print(x, xs)
```

+++ {"colab_type": "text", "id": "ilyoyO34L9f4"}

As usual, you can find all the gory details about lists in the [documentation](https://docs.python.org/3.7/tutorial/datastructures.html#more-on-lists).

+++ {"colab_type": "text", "id": "ovahhxd_L9f5"}

#### Slicing

+++ {"colab_type": "text", "id": "YeSYKhv9L9f6"}

In addition to accessing list elements one at a time, Python provides concise syntax to access sublists; this is known as slicing:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 139
colab_type: code
id: ninq666bL9f6
outputId: c3c2ed92-7358-4fdb-bbc0-e90f82e7e941
---
nums = list(range(5))    # range is a built-in function that creates a list of integers
print(nums)         # Prints "[0, 1, 2, 3, 4]"
print(nums[2:4])    # Get a slice from index 2 to 4 (exclusive); prints "[2, 3]"
print(nums[2:])     # Get a slice from index 2 to the end; prints "[2, 3, 4]"
print(nums[:2])     # Get a slice from the start to index 2 (exclusive); prints "[0, 1]"
print(nums[:])      # Get a slice of the whole list; prints ["0, 1, 2, 3, 4]"
print(nums[:-1])    # Slice indices can be negative; prints ["0, 1, 2, 3]"
nums[2:4] = [8, 9] # Assign a new sublist to a slice
print(nums)         # Prints "[0, 1, 8, 9, 4]"
```

+++ {"colab_type": "text", "id": "UONpMhF4L9f_"}

#### Loops

+++ {"colab_type": "text", "id": "_DYz1j6QL9f_"}

You can loop over the elements of a list like this:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 69
colab_type: code
id: 4cCOysfWL9gA
outputId: 560e46c7-279c-409a-838c-64bea8d321c4
---
animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print(animal)
```

+++ {"colab_type": "text", "id": "KxIaQs7pL9gE"}

If you want access to the index of each element within the body of a loop, use the built-in `enumerate` function:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 69
colab_type: code
id: JjGnDluWL9gF
outputId: 81421905-17ea-4c5a-bcc0-176de19fd9bd
---
animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print('#{}: {}'.format(idx + 1, animal))
```

+++ {"colab_type": "text", "id": "arrLCcMyL9gK"}

#### List comprehensions:

+++ {"colab_type": "text", "id": "5Qn2jU_pL9gL"}

When programming, frequently we want to transform one type of data into another. As a simple example, consider the following code that computes square numbers:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 34
colab_type: code
id: IVNEwoMXL9gL
outputId: d571445b-055d-45f0-f800-24fd76ceec5a
---
nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
print(squares)
```

+++ {"colab_type": "text", "id": "7DmKVUFaL9gQ"}

You can make this code simpler using a list comprehension:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 34
colab_type: code
id: kZxsUfV6L9gR
outputId: 4254a7d4-58ba-4f70-a963-20c46b485b72
---
nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]
print(squares)
```

+++ {"colab_type": "text", "id": "-D8ARK7tL9gV"}

List comprehensions can also contain conditions:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 34
colab_type: code
id: yUtgOyyYL9gV
outputId: 1ae7ab58-8119-44dc-8e57-fda09197d026
---
nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print(even_squares)
```

+++ {"colab_type": "text", "id": "H8xsUEFpL9gZ"}

#### Dictionaries

+++ {"colab_type": "text", "id": "kkjAGMAJL9ga"}

A dictionary stores (key, value) pairs, similar to a `Map` in Java or an object in Javascript. You can use it like this:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 52
colab_type: code
id: XBYI1MrYL9gb
outputId: 8e24c1da-0fc0-4b4c-a3e6-6f758a53b7da
---
d = {'cat': 'cute', 'dog': 'furry'}  # Create a new dictionary with some data
print(d['cat'])       # Get an entry from a dictionary; prints "cute"
print('cat' in d)     # Check if a dictionary has a given key; prints "True"
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 34
colab_type: code
id: pS7e-G-HL9gf
outputId: feb4bf18-c0a3-42a2-eaf5-3fc390f36dcf
---
d['fish'] = 'wet'    # Set an entry in a dictionary
print(d['fish'])      # Prints "wet"
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 165
colab_type: code
id: tFY065ItL9gi
outputId: 7e42a5f0-1856-4608-a927-0930ab37a66c
tags: [raises-exception]
---
print(d['monkey'])  # KeyError: 'monkey' not a key of d
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 52
colab_type: code
id: 8TjbEWqML9gl
outputId: ef14d05e-401d-4d23-ed1a-0fe6b4c77d6f
---
print(d.get('monkey', 'N/A'))  # Get an element with a default; prints "N/A"
print(d.get('fish', 'N/A'))    # Get an element with a default; prints "wet"
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 34
colab_type: code
id: 0EItdNBJL9go
outputId: 652a950f-b0c2-4623-98bd-0191b300cd57
---
del d['fish']        # Remove an element from a dictionary
print(d.get('fish', 'N/A')) # "fish" is no longer a key; prints "N/A"
```

+++ {"colab_type": "text", "id": "wqm4dRZNL9gr"}

You can find all you need to know about dictionaries in the [documentation](https://docs.python.org/2/library/stdtypes.html#dict).

+++ {"colab_type": "text", "id": "IxwEqHlGL9gr"}

It is easy to iterate over the keys in a dictionary:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 69
colab_type: code
id: rYfz7ZKNL9gs
outputId: 155bdb17-3179-4292-c832-8166e955e942
---
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal, legs in d.items():
    print('A {} has {} legs'.format(animal, legs))
```

+++ {"colab_type": "text", "id": "17sxiOpzL9gz"}

Dictionary comprehensions: These are similar to list comprehensions, but allow you to easily construct dictionaries. For example:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 34
colab_type: code
id: 8PB07imLL9gz
outputId: e9ddf886-39ed-4f35-dd80-64a19d2eec9b
---
nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print(even_num_to_square)
```

+++ {"colab_type": "text", "id": "V9MHfUdvL9g2"}

#### Sets

+++ {"colab_type": "text", "id": "Rpm4UtNpL9g2"}

A set is an unordered collection of distinct elements. As a simple example, consider the following:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 52
colab_type: code
id: MmyaniLsL9g2
outputId: 8f152d48-0a07-432a-cf98-8de4fd57ddbb
---
animals = {'cat', 'dog'}
print('cat' in animals)   # Check if an element is in a set; prints "True"
print('fish' in animals)  # prints "False"
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 52
colab_type: code
id: ElJEyK86L9g6
outputId: b9d7dab9-5a98-41cd-efbc-786d0c4377f7
---
animals.add('fish')      # Add an element to a set
print('fish' in animals)
print(len(animals))       # Number of elements in a set;
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 52
colab_type: code
id: 5uGmrxdPL9g9
outputId: e644d24c-26c6-4b43-ab15-8aa81fe884d4
---
animals.add('cat')       # Adding an element that is already in the set does nothing
print(len(animals))       
animals.remove('cat')    # Remove an element from a set
print(len(animals))       
```

+++ {"colab_type": "text", "id": "zk2DbvLKL9g_"}

_Loops_: Iterating over a set has the same syntax as iterating over a list; however since sets are unordered, you cannot make assumptions about the order in which you visit the elements of the set:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 69
colab_type: code
id: K47KYNGyL9hA
outputId: 4477f897-4355-4816-b39b-b93ffbac4bf0
---
animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print('#{}: {}'.format(idx + 1, animal))
```

+++ {"colab_type": "text", "id": "puq4S8buL9hC"}

Set comprehensions: Like lists and dictionaries, we can easily construct sets using set comprehensions:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 34
colab_type: code
id: iw7k90k3L9hC
outputId: 72d6b824-6d31-47b2-f929-4cf434590ee5
---
from math import sqrt
print({int(sqrt(x)) for x in range(30)})
```

+++ {"colab_type": "text", "id": "qPsHSKB1L9hF"}

#### Tuples

+++ {"colab_type": "text", "id": "kucc0LKVL9hG"}

A tuple is an (immutable) ordered list of values. A tuple is in many ways similar to a list; one of the most important differences is that tuples can be used as keys in dictionaries and as elements of sets, while lists cannot. Here is a trivial example:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 69
colab_type: code
id: 9wHUyTKxL9hH
outputId: cdc5f620-04fe-4b0b-df7a-55b061d23d88
---
d = {(x, x + 1): x for x in range(10)}  # Create a dictionary with tuple keys
t = (5, 6)       # Create a tuple
print(type(t))
print(d[t])       
print(d[(1, 2)])
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 165
colab_type: code
id: HoO8zYKzL9hJ
outputId: 28862bfc-0298-40d7-f8c4-168e109d2d93
tags: [raises-exception]
---
t[0] = 1
```

+++ {"colab_type": "text", "id": "AXA4jrEOL9hM"}

### Functions

+++ {"colab_type": "text", "id": "WaRms-QfL9hN"}

Python functions are defined using the `def` keyword. For example:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 69
colab_type: code
id: kiMDUr58L9hN
outputId: 9f53bf9a-7b2a-4c51-9def-398e4677cd6c
---
def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

for x in [-1, 0, 1]:
    print(sign(x))
```

+++ {"colab_type": "text", "id": "U-QJFt8TL9hR"}

We will often define functions to take optional keyword arguments, like this:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 52
colab_type: code
id: PfsZ3DazL9hR
outputId: 6e6af832-67d8-4d8c-949b-335927684ae3
---
def hello(name, loud=False):
    if loud:
        print('HELLO, {}'.format(name.upper()))
    else:
        print('Hello, {}!'.format(name))

hello('Bob')
hello('Fred', loud=True)
```

+++ {"colab_type": "text", "id": "ObA9PRtQL9hT"}

### Classes

+++ {"colab_type": "text", "id": "hAzL_lTkL9hU"}

The syntax for defining classes in Python is straightforward:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 52
colab_type: code
id: RWdbaGigL9hU
outputId: 4f6615c5-75a7-4ce4-8ea1-1e7f5e4e9fc3
---
class Greeter:

    # Constructor
    def __init__(self, name):
        self.name = name  # Create an instance variable

    # Instance method
    def greet(self, loud=False):
        if loud:
          print('HELLO, {}'.format(self.name.upper()))
        else:
          print('Hello, {}!'.format(self.name))

g = Greeter('Fred')  # Construct an instance of the Greeter class
g.greet()            # Call an instance method; prints "Hello, Fred"
g.greet(loud=True)   # Call an instance method; prints "HELLO, FRED!"
```

+++ {"colab_type": "text", "id": "3cfrOV4dL9hW"}

## Numpy

+++ {"colab_type": "text", "id": "fY12nHhyL9hX"}

Numpy is the core library for scientific computing in Python. It provides a high-performance multidimensional array object, and tools for working with these arrays. If you are already familiar with MATLAB, you might find this [tutorial](http://wiki.scipy.org/NumPy_for_Matlab_Users) useful to get started with Numpy.

+++ {"colab_type": "text", "id": "lZMyAdqhL9hY"}

To use Numpy, we first need to import the `numpy` package:

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: 58QdX8BLL9hZ

import numpy as np
```

+++ {"colab_type": "text", "id": "DDx6v1EdL9hb"}

### Arrays

+++ {"colab_type": "text", "id": "f-Zv3f7LL9hc"}

A numpy array is a grid of values, all of the same type, and is indexed by a tuple of nonnegative integers. The number of dimensions is the rank of the array; the shape of an array is a tuple of integers giving the size of the array along each dimension.

+++ {"colab_type": "text", "id": "_eMTRnZRL9hc"}

We can initialize numpy arrays from nested Python lists, and access elements using square brackets:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 52
colab_type: code
id: -l3JrGxCL9hc
outputId: 8d9dad18-c734-4a8a-ca8c-44060a40fb79
---
a = np.array([1, 2, 3])  # Create a rank 1 array
print(type(a), a.shape, a[0], a[1], a[2])
a[0] = 5                 # Change an element of the array
print(a)                  
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 52
colab_type: code
id: ma6mk-kdL9hh
outputId: 0b54ff2f-e7f1-4b30-c653-9bf81cb8fbb0
---
b = np.array([[1,2,3],[4,5,6]])   # Create a rank 2 array
print(b)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 52
colab_type: code
id: ymfSHAwtL9hj
outputId: 5bd292d8-c751-43b9-d480-f357dde52342
---
print(b.shape)
print(b[0, 0], b[0, 1], b[1, 0])
```

+++ {"colab_type": "text", "id": "F2qwdyvuL9hn"}

Numpy also provides many functions to create arrays:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 52
colab_type: code
id: mVTN_EBqL9hn
outputId: d267c65f-ba90-4043-cedb-f468ab1bcc5d
---
a = np.zeros((2,2))  # Create an array of all zeros
print(a)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 34
colab_type: code
id: skiKlNmlL9h5
outputId: 7d1ec1b5-a1fe-4f44-cbe3-cdeacad425f1
---
b = np.ones((1,2))   # Create an array of all ones
print(b)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 52
colab_type: code
id: HtFsr03bL9h7
outputId: 2688b157-2fad-4fc6-f20b-8633207f0326
---
c = np.full((2,2), 7) # Create a constant array
print(c)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 52
colab_type: code
id: -QcALHvkL9h9
outputId: 5035d6fe-cb7e-4222-c972-55fe23c9d4c0
---
d = np.eye(2)        # Create a 2x2 identity matrix
print(d)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 52
colab_type: code
id: RCpaYg9qL9iA
outputId: 25f0b387-39cf-42f3-8701-de860cc75e2e
---
e = np.random.random((2,2)) # Create an array filled with random values
print(e)
```

+++ {"colab_type": "text", "id": "jI5qcSDfL9iC"}

### Array indexing

+++ {"colab_type": "text", "id": "M-E4MUeVL9iC"}

Numpy offers several ways to index into arrays.

+++ {"colab_type": "text", "id": "QYv4JyIEL9iD"}

Slicing: Similar to Python lists, numpy arrays can be sliced. Since arrays may be multidimensional, you must specify a slice for each dimension of the array:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 52
colab_type: code
id: wLWA0udwL9iD
outputId: 99f08618-c513-4982-8982-b146fc72dab3
---
import numpy as np

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2):
# [[2 3]
#  [6 7]]
b = a[:2, 1:3]
print(b)
```

+++ {"colab_type": "text", "id": "KahhtZKYL9iF"}

A slice of an array is a view into the same data, so modifying it will modify the original array.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 52
colab_type: code
id: 1kmtaFHuL9iG
outputId: ee3ab60c-4064-4a9e-b04c-453d3955f1d1
---
print(a[0, 1])
b[0, 0] = 77    # b[0, 0] is the same piece of data as a[0, 1]
print(a[0, 1]) 
```

+++ {"colab_type": "text", "id": "_Zcf3zi-L9iI"}

You can also mix integer indexing with slice indexing. However, doing so will yield an array of lower rank than the original array. Note that this is quite different from the way that MATLAB handles array slicing:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 69
colab_type: code
id: G6lfbPuxL9iJ
outputId: a225fe9d-2a29-4e14-a243-2b7d583bd4bc
---
# Create the following rank 2 array with shape (3, 4)
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(a)
```

+++ {"colab_type": "text", "id": "NCye3NXhL9iL"}

Two ways of accessing the data in the middle row of the array.
Mixing integer indexing with slices yields an array of lower rank,
while using only slices yields an array of the same rank as the
original array:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 69
colab_type: code
id: EOiEMsmNL9iL
outputId: ab2ebe48-9002-45a8-9462-fd490b467f40
---
row_r1 = a[1, :]    # Rank 1 view of the second row of a  
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
row_r3 = a[[1], :]  # Rank 2 view of the second row of a
print(row_r1, row_r1.shape)
print(row_r2, row_r2.shape)
print(row_r3, row_r3.shape)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 104
colab_type: code
id: JXu73pfDL9iN
outputId: 6c589b85-e9b0-4c13-a39d-4cd9fb2f41ac
---
# We can make the same distinction when accessing columns of an array:
col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)
print()
print(col_r2, col_r2.shape)
```

+++ {"colab_type": "text", "id": "VP3916bOL9iP"}

Integer array indexing: When you index into numpy arrays using slicing, the resulting array view will always be a subarray of the original array. In contrast, integer array indexing allows you to construct arbitrary arrays using the data from another array. Here is an example:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 52
colab_type: code
id: TBnWonIDL9iP
outputId: c29fa2cd-234e-4765-c70a-6889acc63573
---
a = np.array([[1,2], [3, 4], [5, 6]])

# An example of integer array indexing.
# The returned array will have shape (3,) and 
print(a[[0, 1, 2], [0, 1, 0]])

# The above example of integer array indexing is equivalent to this:
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 52
colab_type: code
id: n7vuati-L9iR
outputId: c3e9ba14-f66e-4202-999e-2e1aed5bd631
---
# When using integer array indexing, you can reuse the same
# element from the source array:
print(a[[0, 0], [1, 1]])

# Equivalent to the previous integer array indexing example
print(np.array([a[0, 1], a[0, 1]]))
```

+++ {"colab_type": "text", "id": "kaipSLafL9iU"}

One useful trick with integer array indexing is selecting or mutating one element from each row of a matrix:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 86
colab_type: code
id: ehqsV7TXL9iU
outputId: de509c40-4ee4-4b7c-e75d-1a936a3350e7
---
# Create a new array from which we will select elements
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
print(a)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 34
colab_type: code
id: pAPOoqy5L9iV
outputId: f812e29b-9218-4767-d3a8-e9854e754e68
---
# Create an array of indices
b = np.array([0, 2, 0, 1])

# Select one element from each row of a using the indices in b
print(a[np.arange(4), b])  # Prints "[ 1  6  7 11]"
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 86
colab_type: code
id: 6v1PdI1DL9ib
outputId: 89f50f82-de1b-4417-e55c-edbc0ee07584
---
# Mutate one element from each row of a using the indices in b
a[np.arange(4), b] += 10
print(a)
```

+++ {"colab_type": "text", "id": "kaE8dBGgL9id"}

Boolean array indexing: Boolean array indexing lets you pick out arbitrary elements of an array. Frequently this type of indexing is used to select the elements of an array that satisfy some condition. Here is an example:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 69
colab_type: code
id: 32PusjtKL9id
outputId: 8782e8ec-b78d-44d7-8141-23e39750b854
---
import numpy as np

a = np.array([[1,2], [3, 4], [5, 6]])

bool_idx = (a > 2)  # Find the elements of a that are bigger than 2;
                    # this returns a numpy array of Booleans of the same
                    # shape as a, where each slot of bool_idx tells
                    # whether that element of a is > 2.

print(bool_idx)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 52
colab_type: code
id: cb2IRMXaL9if
outputId: 5983f208-3738-472d-d6ab-11fe85b36c95
---
# We use boolean array indexing to construct a rank 1 array
# consisting of the elements of a corresponding to the True values
# of bool_idx
print(a[bool_idx])

# We can do all of the above in a single concise statement:
print(a[a > 2])
```

+++ {"colab_type": "text", "id": "CdofMonAL9ih"}

For brevity we have left out a lot of details about numpy array indexing; if you want to know more you should read the documentation.

+++ {"colab_type": "text", "id": "jTctwqdQL9ih"}

### Datatypes

+++ {"colab_type": "text", "id": "kSZQ1WkIL9ih"}

Every numpy array is a grid of elements of the same type. Numpy provides a large set of numeric datatypes that you can use to construct arrays. Numpy tries to guess a datatype when you create an array, but functions that construct arrays usually also include an optional argument to explicitly specify the datatype. Here is an example:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 34
colab_type: code
id: 4za4O0m5L9ih
outputId: 2ea4fb80-a4df-43f9-c162-5665895c13ae
---
x = np.array([1, 2])  # Let numpy choose the datatype
y = np.array([1.0, 2.0])  # Let numpy choose the datatype
z = np.array([1, 2], dtype=np.int64)  # Force a particular datatype

print(x.dtype, y.dtype, z.dtype)
```

+++ {"colab_type": "text", "id": "RLVIsZQpL9ik"}

You can read all about numpy datatypes in the [documentation](http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html).

+++ {"colab_type": "text", "id": "TuB-fdhIL9ik"}

### Array math

+++ {"colab_type": "text", "id": "18e8V8elL9ik"}

Basic mathematical functions operate elementwise on arrays, and are available both as operator overloads and as functions in the numpy module:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 86
colab_type: code
id: gHKvBrSKL9il
outputId: a8a924b1-9d60-4b68-8fd3-e4657ae3f08b
---
x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# Elementwise sum; both produce the array
print(x + y)
print(np.add(x, y))
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 86
colab_type: code
id: 1fZtIAMxL9in
outputId: 122f1380-6144-4d6c-9d31-f62d839889a2
---
# Elementwise difference; both produce the array
print(x - y)
print(np.subtract(x, y))
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 86
colab_type: code
id: nil4AScML9io
outputId: 038c8bb2-122b-4e59-c0a8-a091014fe68e
---
# Elementwise product; both produce the array
print(x * y)
print(np.multiply(x, y))
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 86
colab_type: code
id: 0JoA4lH6L9ip
outputId: 12351a74-7871-4bc2-97ce-a508bf4810da
---
# Elementwise division; both produce the array
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print(x / y)
print(np.divide(x, y))
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 52
colab_type: code
id: g0iZuA6bL9ir
outputId: 29927dda-4167-4aa8-fbda-9008b09e4356
---
# Elementwise square root; produces the array
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print(np.sqrt(x))
```

+++ {"colab_type": "text", "id": "a5d_uujuL9it"}

Note that unlike MATLAB, `*` is elementwise multiplication, not matrix multiplication. We instead use the dot function to compute inner products of vectors, to multiply a vector by a matrix, and to multiply matrices. dot is available both as a function in the numpy module and as an instance method of array objects:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 52
colab_type: code
id: I3FnmoSeL9iu
outputId: 46f4575a-2e5e-4347-a34e-0cc5bd280110
---
x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11, 12])

# Inner product of vectors; both produce 219
print(v.dot(w))
print(np.dot(v, w))
```

+++ {"colab_type": "text", "id": "vmxPbrHASVeA"}

You can also use the `@` operator which is equivalent to numpy's `dot` operator.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 34
colab_type: code
id: vyrWA-mXSdtt
outputId: a9aae545-2c93-4649-b220-b097655955f6
---
print(v @ w)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 69
colab_type: code
id: zvUODeTxL9iw
outputId: 4093fc76-094f-4453-a421-a212b5226968
---
# Matrix / vector product; both produce the rank 1 array [29 67]
print(x.dot(v))
print(np.dot(x, v))
print(x @ v)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 121
colab_type: code
id: 3V_3NzNEL9iy
outputId: af2a89f9-af5d-47a6-9ad2-06a84b521b94
---
# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print(x.dot(y))
print(np.dot(x, y))
print(x @ y)
```

+++ {"colab_type": "text", "id": "FbE-1If_L9i0"}

Numpy provides many useful functions for performing computations on arrays; one of the most useful is `sum`:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 69
colab_type: code
id: DZUdZvPrL9i0
outputId: 99cad470-d692-4b25-91c9-a57aa25f4c6e
---
x = np.array([[1,2],[3,4]])

print(np.sum(x))  # Compute sum of all elements; prints "10"
print(np.sum(x, axis=0))  # Compute sum of each column; prints "[4 6]"
print(np.sum(x, axis=1))  # Compute sum of each row; prints "[3 7]"
```

+++ {"colab_type": "text", "id": "ahdVW4iUL9i3"}

You can find the full list of mathematical functions provided by numpy in the [documentation](http://docs.scipy.org/doc/numpy/reference/routines.math.html).

Apart from computing mathematical functions using arrays, we frequently need to reshape or otherwise manipulate data in arrays. The simplest example of this type of operation is transposing a matrix; to transpose a matrix, simply use the T attribute of an array object:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 104
colab_type: code
id: 63Yl1f3oL9i3
outputId: c75ac7ba-4351-42f8-a09c-a4e0d966ab50
---
print(x)
print("transpose\n", x.T)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 104
colab_type: code
id: mkk03eNIL9i4
outputId: 499eec5a-55b7-473a-d4aa-9d023d63885a
---
v = np.array([[1,2,3]])
print(v )
print("transpose\n", v.T)
```

+++ {"colab_type": "text", "id": "REfLrUTcL9i7"}

### Broadcasting

+++ {"colab_type": "text", "id": "EygGAMWqL9i7"}

Broadcasting is a powerful mechanism that allows numpy to work with arrays of different shapes when performing arithmetic operations. Frequently we have a smaller array and a larger array, and we want to use the smaller array multiple times to perform some operation on the larger array.

For example, suppose that we want to add a constant vector to each row of a matrix. We could do it like this:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 86
colab_type: code
id: WEEvkV1ZL9i7
outputId: 3896d03c-3ece-4aa8-f675-aef3a220574d
---
# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)   # Create an empty matrix with the same shape as x

# Add the vector v to each row of the matrix x with an explicit loop
for i in range(4):
    y[i, :] = x[i, :] + v

print(y)
```

+++ {"colab_type": "text", "id": "2OlXXupEL9i-"}

This works; however when the matrix `x` is very large, computing an explicit loop in Python could be slow. Note that adding the vector v to each row of the matrix `x` is equivalent to forming a matrix `vv` by stacking multiple copies of `v` vertically, then performing elementwise summation of `x` and `vv`. We could implement this approach like this:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 86
colab_type: code
id: vS7UwAQQL9i-
outputId: 8621e502-c25d-4a18-c973-886dbfd1df36
---
vv = np.tile(v, (4, 1))  # Stack 4 copies of v on top of each other
print(vv)                # Prints "[[1 0 1]
                         #          [1 0 1]
                         #          [1 0 1]
                         #          [1 0 1]]"
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 86
colab_type: code
id: N0hJphSIL9jA
outputId: def6a757-170c-43bf-8728-732dfb133273
---
y = x + vv  # Add x and vv elementwise
print(y)
```

+++ {"colab_type": "text", "id": "zHos6RJnL9jB"}

Numpy broadcasting allows us to perform this computation without actually creating multiple copies of v. Consider this version, using broadcasting:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 86
colab_type: code
id: vnYFb-gYL9jC
outputId: df3bea8a-ad72-4a83-90bb-306b55c6fb93
---
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  # Add v to each row of x using broadcasting
print(y)
```

+++ {"colab_type": "text", "id": "08YyIURKL9jH"}

The line `y = x + v` works even though `x` has shape `(4, 3)` and `v` has shape `(3,)` due to broadcasting; this line works as if v actually had shape `(4, 3)`, where each row was a copy of `v`, and the sum was performed elementwise.

Broadcasting two arrays together follows these rules:

1. If the arrays do not have the same rank, prepend the shape of the lower rank array with 1s until both shapes have the same length.
2. The two arrays are said to be compatible in a dimension if they have the same size in the dimension, or if one of the arrays has size 1 in that dimension.
3. The arrays can be broadcast together if they are compatible in all dimensions.
4. After broadcasting, each array behaves as if it had shape equal to the elementwise maximum of shapes of the two input arrays.
5. In any dimension where one array had size 1 and the other array had size greater than 1, the first array behaves as if it were copied along that dimension

If this explanation does not make sense, try reading the explanation from the [documentation](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) or this [explanation](http://wiki.scipy.org/EricsBroadcastingDoc).

Functions that support broadcasting are known as universal functions. You can find the list of all universal functions in the [documentation](http://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs).

Here are some applications of broadcasting:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 69
colab_type: code
id: EmQnwoM9L9jH
outputId: f59e181e-e2d4-416c-d094-c4d003ce8509
---
# Compute outer product of vectors
v = np.array([1,2,3])  # v has shape (3,)
w = np.array([4,5])    # w has shape (2,)
# To compute an outer product, we first reshape v to be a column
# vector of shape (3, 1); we can then broadcast it against w to yield
# an output of shape (3, 2), which is the outer product of v and w:

print(np.reshape(v, (3, 1)) * w)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 52
colab_type: code
id: PgotmpcnL9jK
outputId: 567763d3-073a-4e3c-9ebe-6c7d2b6d3446
---
# Add a vector to each row of a matrix
x = np.array([[1,2,3], [4,5,6]])
# x has shape (2, 3) and v has shape (3,) so they broadcast to (2, 3),
# giving the following matrix:

print(x + v)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 52
colab_type: code
id: T5hKS1QaL9jK
outputId: 5f14ac5c-7a21-4216-e91d-cfce5720a804
---
# Add a vector to each column of a matrix
# x has shape (2, 3) and w has shape (2,).
# If we transpose x then it has shape (3, 2) and can be broadcast
# against w to yield a result of shape (3, 2); transposing this result
# yields the final result of shape (2, 3) which is the matrix x with
# the vector w added to each column. Gives the following matrix:

print((x.T + w).T)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 52
colab_type: code
id: JDUrZUl6L9jN
outputId: 53e99a89-c599-406d-9fe3-7aa35ae5fb90
---
# Another solution is to reshape w to be a row vector of shape (2, 1);
# we can then broadcast it directly against x to produce the same
# output.
print(x + np.reshape(w, (2, 1)))
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 52
colab_type: code
id: VzrEo4KGL9jP
outputId: 53c9d4cc-32d5-46b0-d090-53c7db57fb32
---
# Multiply a matrix by a constant:
# x has shape (2, 3). Numpy treats scalars as arrays of shape ();
# these can be broadcast together to shape (2, 3), producing the
# following array:
print(x * 2)
```

+++ {"colab_type": "text", "id": "89e2FXxFL9jQ"}

Broadcasting typically makes your code more concise and faster, so you should strive to use it where possible.

+++ {"colab_type": "text", "id": "iF3ZtwVNL9jQ"}

This brief overview has touched on many of the important things that you need to know about numpy, but is far from complete. Check out the [numpy reference](http://docs.scipy.org/doc/numpy/reference/) to find out much more about numpy.

+++ {"colab_type": "text", "id": "tEINf4bEL9jR"}

## Matplotlib

+++ {"colab_type": "text", "id": "0hgVWLaXL9jR"}

Matplotlib is a plotting library. In this section give a brief introduction to the `matplotlib.pyplot` module, which provides a plotting system similar to that of MATLAB.

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: cmh_7c6KL9jR

import matplotlib.pyplot as plt
```

+++ {"colab_type": "text", "id": "jOsaA5hGL9jS"}

By running this special iPython command, we will be displaying plots inline:

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: ijpsmwGnL9jT

%matplotlib inline
```

+++ {"colab_type": "text", "id": "U5Z_oMoLL9jV"}

### Plotting

+++ {"colab_type": "text", "id": "6QyFJ7dhL9jV"}

The most important function in `matplotlib` is plot, which allows you to plot 2D data. Here is a simple example:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 282
colab_type: code
id: pua52BGeL9jW
outputId: 9ac3ee0f-7ff7-463b-b901-c33d21a2b10c
---
# Compute the x and y coordinates for points on a sine curve
x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

# Plot the points using matplotlib
plt.plot(x, y)
```

+++ {"colab_type": "text", "id": "9W2VAcLiL9jX"}

With just a little bit of extra work we can easily plot multiple lines at once, and add a title, legend, and axis labels:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 312
colab_type: code
id: TfCQHJ5AL9jY
outputId: fdb9c033-0f06-4041-a69d-a0f3a54c7206
---
y_sin = np.sin(x)
y_cos = np.cos(x)

# Plot the points using matplotlib
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
```

+++ {"colab_type": "text", "id": "R5IeAY03L9ja"}

### Subplots 

+++ {"colab_type": "text", "id": "CfUzwJg0L9ja"}

You can plot different things in the same figure using the subplot function. Here is an example:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 281
colab_type: code
id: dM23yGH9L9ja
outputId: 14dfa5ea-f453-4da5-a2ee-fea0de8f72d9
---
# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(2, 1, 1)

# Make the first plot
plt.plot(x, y_sin)
plt.title('Sine')

# Set the second subplot as active, and make the second plot.
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

# Show the figure.
plt.show()
```

+++ {"colab_type": "text", "id": "gLtsST5SL9jc"}

You can read much more about the `subplot` function in the [documentation](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.subplot).

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: eJXA5AWSL9jc


```
