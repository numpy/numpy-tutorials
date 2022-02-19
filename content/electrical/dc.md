---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Tutorial: Solve DC electric circuits using Numpy

## Overview

### What you'll do

* Plot [Ohm's Law](https://en.wikipedia.org/wiki/Ohm%27s_law)
* Find the value of resistance that ensures optimal power flow
* Solve [DC circuits](https://en.wikipedia.org/wiki/Direct_current) using *mesh analysis*

### What you'll learn

* You'll learn how to create a NumPy array and manipulate it
* You'll learn how to make simple plots out of NumPy arrays
* You'll learn how to form matrices in NumPy
* You'll learn how to perform simple linear algebra operations with these matrices

### What you'll need
* A "can-do" attitude
* Basic electrical knowledge (Ohm's Law, KCL, KVL)
* Basic linear algebra knowledge (Matrices, Inversion and Determinants)
* Basic Python knowledge (know how to run code and import modules - see the 
[Python Tutorial](https://docs.python.org/3/tutorial/) to review)
* The [Matplotlib](https://matploblib.org) package if you want to reproduce the plots.

## Part 1: Plotting Ohm's Law

### Part 1a: Calculating Ohm's Law in pure python

<img src="imgs/circuit1.png" width="400" height="200"/>

Solving for $I$ in *Figure 1* is a pretty trivial matter, it can be done in a
few lines of code. Using `R` to denote the resistance, `V` to denote the 
voltage and `I` to denote the current, we have the following:
```python
R = 10
V = 5 
I = V/R 
print(f"I = {I} A")
```


### Part 1b: Generating voltages and currents in pure python

<img src="imgs/circuit2.png" width="400" height="200"/>

Now, we would like to plot the variation in the current `I` with respect to changes in the voltage `V`, given a fixed value of the resistance `R`.
Before a plot of Ohm's Law can be made, you first need to generate a series of values 
for the voltages and currents. Not using NumPy, and choosing `V` and `I` to denote 
the lists containing those values, respectively, we would end up with the following 
code:


```python
import numpy as np
# V and I are initialized as empty lists
R = 10
V = []
I = []

for value in range(-5, 6): 
    V.append(value)
for voltage in V: 
    I.append(voltage/R)
```

The code above is **inefficient**. You generally want to avoid running *for loops* in Python 
since they are statically typed and interpreted. In this simple example it won't make much of a difference
but with larger and more complex code it will noticably run worse. 

### Part 1c: Generating voltages and currents using numpy

Let's see how to repeat what was done in the previous section using `NumPy`. Using
the method [`np.linspace`](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html)
to create a [`numpy.ndarray`](https://numpy.org/doc/stable/reference/arrays.ndarray.html)
you can specify:

1. The starting value of the array;
2. The ending value of the array;
3. The number of points between the starting and ending.

```python
R = 10
V = np.linspace(-5, 5, 11)

I = V/R
```

### Part 1d: Plotting using matplotlib

You will use the `matplotlib` library to plot $V$ and $I$. You can check out the 
[introductory](https://matplotlib.org/3.1.1/tutorials/introductory/pyplot.html)
tutorial on matplotlib's `pyplot` interface to learn more.

> If you are executing the commands below in the IPython shell,
> it might be necessary to use the command `plt.show()` to show the image window.

```python
import matplotlib.pyplot as plt
plt.plot(V, I)
```
The code below will plot the exact same graph, but also add:

1. An x-axis label;
2. A y-axis label;
3. A title;
4. A grid.

```python
plt.clf() # Clear the plot if there was anything on it
plt.plot(V, I)
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.title('Ohms Law')
plt.grid()
```

## Part 2: Finding the conditions for maximum power transfer

### Part 2a: Plotting maximum power transfer

<img src="imgs/circuit3.png" width="400" height="200"/>


[The maximum power transfer theorem](https://en.wikipedia.org/wiki/Maximum_power_transfer_theorem)
states that maximum power transfer will occur when the resistance of the load (Rload)
equals the resistance of the source (Rth). 
You will write some Python code using NumPy arrays to test this. (The values for `V` 
and `Rth` are arbitrary, as long as they are positive you can choose them to be whatever 
value you wish). 

```python
V = 8
Rth = 5

Rload = np.linspace(0.1, 50, 20)
Iload = V/(Rth + Rload) # Element wise division
Vload = Iload * Rload # Element wise multiplication
Pload = Iload * Vload

plt.plot(Rload, Pload)
```


### Part 2b: Using NumPy to find Rload

Looking at the plot, it does seem that when `Rload = Rth` there is maximum power
transfer. But it is not 100% clear. You can use NumPy to find the exact value of `Rload`
that makes `Pload` maximum.

```python
Pmax = np.max(Pload) # Finding the maximum value of Pload
print(f"Pmax = {Pmax}")

Pmax_idx = np.argmax(Pload) # Finding the index in which Pload is maximum
print(f"Pmax_idx = {Pmax_idx}")

Rload[Pmax_idx]
print(f"Rload[Pmax_idx] = {Rload[Pmax_idx]}")
```

As can be seen, the value of `Rload` isn't exactly 5. This is because the 
linear spacing that was used for `np.linspace` in *Part 2a* was too small. 
If we were to increase the spacing, then `Rload` would be closer to 5.

## Part 3: Solving circuits using mesh analysis

### Part 3a: 2x2 Mesh Matrix

<img src="imgs/circuit4.png" height="200" width="400"/>

The circuit in *Figure 4* can be solved via KVL and KCL, to speed up the 
process a [mesh analysis](https://en.wikipedia.org/wiki/Mesh_analysis) can be
preformed to yield 
*two [linear equations](https://en.wikipedia.org/wiki/System_of_linear_equations)*:


\begin{equation*}
I_1 (R_1 + R_2) + I_2(-R_2) = V_1 \\
I_1 (-R_2) + I_2(R_2 + R_3) = V_2
\end{equation*}

These two equations can be written in *matrix format* and inputted into NumPy:

\begin{equation*}
\begin{bmatrix} R1+R2 & -R2 \\ -R2 & R2+R3 \end{bmatrix} 
* 
\begin{bmatrix} I1 \\ I2  \end{bmatrix}
=
\begin{bmatrix} V1 \\ -V2  \end{bmatrix}
\end{equation*}


```python
R1 = 10
R2 = 5
R3 = 5
V1 = 5
V2 = -15

a11 = R1+R2
a12 = -R2
a21 = -R2
a22 = R2+R3

b1 = V1
b2 = -V2

A = np.array([[a11, a12],
              [a21, a22]])

b = np.array([[b1], [b2]])

print(A)
print(b)
```

If solving the matrix using pencil and paper, you would first invert $A$ and
then multiply it by $b$ to get the current ($x$). Using NumPy, you can do this by using the 
[`numpy.linalg.inv`](https://numpy.org/devdocs/reference/generated/numpy.linalg.inv.html)
function and the `@` operator (equivalent to the 
[`numpy.matmul`](https://numpy.org/devdocs/reference/generated/numpy.matmul.html) function).

\begin{equation*}
\begin{bmatrix} I1 \\ I2  \end{bmatrix} 
= 
\begin{bmatrix} R1+R2 & -R2 \\ -R2 & R2+R3 \end{bmatrix}^{-1} 
* 
\begin{bmatrix} V1 \\ -V2  \end{bmatrix}
\end{equation*}

```python
Ainv = np.linalg.inv(A)
print(Ainv)

x = Ainv @ b
```


**NEVER** invert your matrices if you are programming on a computer (except if
you actually need the inverted matrix itself, which is rare). Instead use the 
`np.linalg.solve` function.

> Note 1: If you want to learn more about the complications arising from matrix inversion,
> check out this
> (article)[https://www.johndcook.com/blog/2010/01/19/dont-invert-that-matrix/]
> talking about it's harms. And check out this
> (paper)[http://www.netlib.org/lapack/lawnspdf/lawn27.pdf] if you want a full
> mathematical explanation.


```python
x = np.linalg.solve(A, b)
print(x)
```

### Part 3b: 3x3 Mesh Matrix

<img src="imgs/circuit5.png" width="500" height="250"/>

Repeat the same process that was done before. The only difference is that this
time you will end up with a `3x3` matrix


\begin{equation*}
\begin{bmatrix} 
R1+R2+R3 & -R2 & -R3\\ -R2 & R2+R4 & -R4 \\ -R3 & -R4 & R3+R4 
\end{bmatrix}
*
\begin{bmatrix} 
I1 \\ I2 \\ I3
\end{bmatrix}
=
\begin{bmatrix} 
0 \\ V1 \\ -V2
\end{bmatrix}
\end{equation*}

```python
R1 = 90
R2 = 30
R3 = 30
R4 = 5
V1 = 25
V2 = -65

a11 = R1 + R2 + R3
a12 = -R2
a13 = -R3
a21 = -R2
a22 = R2+R4
a23 = -R4
a31 = -R3
a32 = -R4
a33 = R3+R4

b1 = 0
b2 = V1
b3 = -V2

A = np.array([[a11, a12, a13],
              [a21, a22, a23],
              [a31, a32, a33]])

b = np.array([[b1], [b2], [b3]])

x = np.linalg.solve(A, b)
print(x)
```

### Part 3c: 3x3 Mesh Matrix with a mistake

<img src="imgs/circuit6.png" width="500" height="300"/>

For this last circuit, let us purposely make a mistake. Instead of taking a loop
through $R1$, let's take it through `Loop 1` instead. Thereby missing vital
information.

\begin{equation*}
\begin{bmatrix} 
-R2-R3 & +R2 & +R3\\ -R2 & R2+R4 & -R4 \\ -R3 & -R4 & R3+R4 
\end{bmatrix}
*
\begin{bmatrix} 
I1 \\ I2 \\ I3
\end{bmatrix}
=
\begin{bmatrix} 
V1-V2 \\ V1 \\ -V2
\end{bmatrix}
\end{equation*}


```python
R1 = 90
R2 = 30
R3 = 30
R4 = 5
V1 = 25
V2 = -65

a11 = -R2 - R3
a12 = +R2
a13 = +R3
a21 = -R2
a22 = R2+R4
a23 = -R4
a31 = -R3
a32 = -R4
a33 = R3+R4

b1 = V1-V2
b2 = V1
b3 = -V2

A = np.array([[a11, a12, a13],
              [a21, a22, a23],
              [a31, a32, a33]])

b = np.array([[b1], [b2], [b3]])
```
```python
# x = np.linalg.solve(A, b)
# print(x)
```

When you run it, you will get this as an output:

```
LinAlgError                               Traceback (most recent call last)
<ipython3-input-2-c441d734ff80> in <module>
     26 b = np.array([[b1], [b2], [b3]])
     27
---> 28 x = np.linalg.solve(A, b)
     29 print(x)

<__array_function__ internals> in solve(*args, **kwargs)

~/Documents/venv_global/lib/python3.8/site-packages/numpy/linalg/linalg.py in solve(a, b)
    397     signature = 'DD->D' if isComplexType(t) else 'dd->d'
    398     extobj = get_linalg_error_extobj(_raise_linalgerror_singular)
--> 399     r = gufunc(a, b, signature=signature, extobj=extobj)
    400
    401     return wrap(r.astype(result_t, copy=False))

~/Documents/venv_global/lib/python3.8/site-packages/numpy/linalg/linalg.py in _raise_lina
lgerror_singular(err, flag)
     95
     96 def _raise_linalgerror_singular(err, flag):
---> 97     raise LinAlgError("Singular matrix")
     98
     99 def _raise_linalgerror_nonposdef(err, flag):

LinAlgError: Singular matrix
```

An error, how unexpected!

As a programmer you need to get used to reading error outputs. Looking at the
last line it says that there is a `Singular Matrix` error.  This happened
because the rows of the $A$ matrix are linearly dependent, $Row1 = Row2 + Row3$.


\begin{equation*}
A =
\begin{bmatrix} 
-R2-R3 & +R2 & +R3\\ -R2 & R2+R4 & -R4 \\ -R3 & -R4 & R3+R4 
\end{bmatrix}
\end{equation*}

This can be easily confirmed by taking the determinant of $A$:

```
Det = np.linalg.det(A)
```

## Conclusion

### On your own

* Try doing a nodal analysis.

* In Part 2a Try using voltage division to calculate `Pload` directly 
(ie without needing to calculate `Iload`)

### In practice...

- The python code written in `Part 1b` can indeed be written without any
`for-loops` (use `range` to set the value of $V$)

- There are many occasions in which the `scipy.linalg` module is preferrable 
over the `numpy.linalg` module. For more information on this, check the 
[scipy.linalg Reference](https://docs.scipy.org/doc/scipy/reference/tutorial/linalg.html).

### Further reading

* https://numpy.org/doc/stable/user/

