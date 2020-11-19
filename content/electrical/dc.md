---
jupytext:
 formats: md:myst
 text_representation:
   format_name: myst
kernelspec:
 display_name: Python 3
 name: python3
---

# Tutorial: Solve DC electric circuits using Numpy

```{figure} imgs/pikachu_greeting.png
```

```{contents}
:depth: 3
```

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
* Basic Python knowledge (know how to run code and import modules - see the [Python Tutorial](https://docs.python.org/3/tutorial/) to review)
* The [Matplotlib](https://matploblib.org) package if you want to reproduce the plots.

## Part 1: Plotting Ohm's Law
### Part 1a: Calculating Ohm's Law in pure python

```{figure} imgs/circuit1.png
---
width: 400px
height: 200px
---
Figure 1: Simple Circuit
```

Solving for $I$ in *Figure 1* is a pretty trivial matter, it can be done in a few lines of code. Using `R` to denote the resistance, `V` to denote the voltage and `I` to denote the current, we have the following:


```{code-cell} ipython3
R = 10
V = 5
I = V/R
print(f"I = {I} A")
```

### Part 1b: Generating voltages and currents in pure python

```{figure} imgs/circuit2.png
---
width: 400px
height: 200px
---
Figure 2: Ohm's Law
```
Now, we would like to plot the variation in the current `I` with respect to changes in the voltage `V`, given a fixed value of the resistance `R`.
Before a plot of Ohm's Law can be made, you first need to generate a series of values 
for the voltages and currents. Not using NumPy, and choosing `V` and `I` to denote 
the lists containing those values, respectively, we would end up with the following 
code:

```{code-cell} ipython3
R = 10
# V and I are initialized as empty lists
V = [] 
I = [] 

for i in range(-5, 6): 
    V.append(i)
for voltage in V: 
    I.append(voltage/R)

print(f"Voltages (V): {V}")
print(f"Current (A): {I}")

There are a two issues with what was done above:
1. It is **inefficient**. You want to avoid running *for loops* in Python because
they make the code run slower. In this simple example it won't make much of a difference
but with larger and more complex code it well.
2. It is **unpythonic**. The code above works but is not clear and maintainable.

```{admonition} Challenge 1
How would you make a list of voltages that have a decimal point (V = [-5, -4.9, -4.8 ...])
```

```{admonition} Challenge 2
Try to optimize the code further by removing one of the `for-loops`
```

### Part 1c: Generating voltages and currents using numpy

Let's see how to repeat what was done in the previous section using `NumPy`. Using
the method [`np.linspace`](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html) to create a [`numpy.ndarray`](https://numpy.org/doc/stable/reference/arrays.ndarray.html) you can specify:

1. The starting value of the array;
2. The ending value of the array;
3. The number of points between the starting and ending.


```{code-cell} ipython3
R = 10
V = np.linspace(-5, 5, 11)

print(f"Voltage (V): {V}")
print(f"Length of V: {V.shape}")
print(f"Type of V: {type(V)}")

I = V/R

print(f"Current (A): {I}")
print(f"Length of I: {I.shape}")
print(f"Type of I: {type(I)}")
```

### Part 1d: Plotting using matplotlib

You will use the `matplotlib` library to plot $V$ and $I$. You can check out the [introductory](https://matplotlib.org/3.1.1/tutorials/introductory/pyplot.html) tutorial on matplotlib's `pyplot` interface to learn more.

```{code-cell} ipython3
import matplotlib.pyplot as plt
plt.plot(V, I)
plt.show() # Run this command if the plot doesn't show
```

```{figure} imgs/plot1_basic.png
---
width: 400px
---
```

```{code-cell} ipython3
plt.clf() # Clear the plot if there was anything on it
plt.plot(V, I)
plt.xlabel('Voltage (V)') # x-axis label
plt.ylabel('Current (A)') # y-axis label
plt.title('Ohms Law')
plt.grid()
```

```{figure} imgs/plot2_better.png
---
width: 400px
---
```

## Part 2: Finding the conditions for maximum power transfer

### Part 2a: Plotting maximum power transfer

```{figure} imgs/circuit3.png
---
width: 400px
height: 200px
---
Figure 3: Maximum Power Transfer Circuit
```

[The maximum power transfer theorem](https://en.wikipedia.org/wiki/Maximum_power_transfer_theorem) states that maximum power transfer will occur when Rload == Rth. You will write some Python code using NumPy arrays to test this.


```{code-cell} ipython3
V = 8
Rth = 5

Rload = np.linspace(0.1, 50, 20)
Iload = V/(Rth + Rload) # Element wise division
Vload = Iload * Rload # Element wise multiplication
Pload = Iload * Vload

plt.plot(Rload, Pload)
```

```{figure} imgs/plot2_better.png
---
width: 400px
---
```

```{figure} imgs/plot3_power.png
---
width: 400px
---
```

```{admonition} Challenge
Try using voltage division to calculate `Pload` directly
(ie without needing to calculate `Iload`)
```

### Part 2b: Using NumPy to find Rload

Looking at the plot, it does seem that when `Rload = Rth` there is maximum power
transfer. But it is not 100% clear. You can use numpy to find the exact value of `Rload`
that makes `Pload` maximum.


```{code-cell} ipython3
Pmax = np.max(Pload) # Finding the maximum value of Pload
print(f"Pmax = {Pmax}")

Pmax_idx = np.argmax(Pload) # Finding the index in which Pload is maximum
print(f"Pmax_idx = {Pmax_idx}")

Rload[Pmax_idx]
print(f"Rload[Pmax_idx] = {Rload[Pmax_idx]}")
```

```{admonition} Challenge
 Why is the value of `Rload` not exactly 5? How would you
increase the accuracy?
```

## Part 3: Solving circuits using mesh analysis

### Part 3a: 2x2 Mesh Matrix

```{figure} imgs/circuit4.png
---
width: 400px
height: 200px
---
Figure 4: Loop Circuit
```

Doing a mesh analysis on *Figure 4* yields *two linear equations*:


```{math}
I1(R1+R2) + I2(-R2) = V1

I1(-R2) + I2(R2+R3) = -V2
```

These two equations can be written in *matrix format* and inputted into NumPy:

```{math}
\begin{bmatrix} R1+R2 & -R2 \\ -R2 & R2+R3 \end{bmatrix} 
* 
\begin{bmatrix} I1 \\ I2  \end{bmatrix}
=
\begin{bmatrix} V1 \\ -V2  \end{bmatrix}
```

```{admonition} Note
Check out Wikipedia's articles on 
[mesh analysis](https://www.electronics-tutorials.ws/dccircuits/dcp_5.html),
[linear equations](https://en.wikipedia.org/wiki/System_of_linear_equations), 
and [matrices](https://en.wikipedia.org/wiki/Matrix_(mathematics))
if you would like to revise the concepts presented here
```

```{code-cell} ipython3
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
then multiply it by $b$.

```{math}
\begin{bmatrix} I1 \\ I2  \end{bmatrix} 
= 
\begin{bmatrix} R1+R2 & -R2 \\ -R2 & R2+R3 \end{bmatrix}^{-1} 
* 
\begin{bmatrix} V1 \\ -V2  \end{bmatrix}
```

```{code-cell} ipython3
Ainv = np.linalg.inv(A)
print(Ainv)

x = Ainv @ b # The @ operator multiplies two matrices
print(x)
```

```{figure} imgs/pikachu_angry.jpg
---
width: 500px
---

```

**NEVER** invert your matrices if you are programming on a computer (except if
you actually need the inverted matrix itself, which is rare). The process is
very sensitive to scaling, propagates numerical errors and will make **Pikachu**
angry! Instead use the `np.linalg.solve` function.

```{admonition} Note 1
If you want to learn more about the complications arising from matrix inversion,
check out this
(article)[https://www.johndcook.com/blog/2010/01/19/dont-invert-that-matrix/]
talking about it's harms. And check out this
(paper)[http://www.netlib.org/lapack/lawnspdf/lawn27.pdf] if you want a full
mathematical explanation.
```

```{admonition} Note 2
Check out the reference documentation for the [numpy.linalg](https://numpy.org/doc/stable/reference/routines.linalg.html) module to learn more about the linear algebra functions available in NumPy.
```

```{code-cell} ipython3
x = np.linalg.solve(A, b)
print(x)
```

### Part 3b: 3x3 Mesh Matrix

```{figure} imgs/circuit5.png
---
width: 500px
height: 250px
---
Figure 5: Three Loop Circuit
```

Repeat the same process that was done before. The only difference is that this
time you will end up with a `3x3` matrix

```{math}
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
```

```{code-cell} ipython3
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
```{figure} imgs/circuit6.png
---
width: 500px
height: 300px
---
Figure 6: Three Loop Circuit neglecting a loop
```

For this last circuit, let us purposely make a mistake. Instead of taking a loop
through $R1$, let's take it through `Loop 1` instead. Thereby missing vital
information.

```{math}
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
```

```{code-cell} ipython3
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

```{figure} imgs/pikachu_surprised.jpg
---
width: 400px
---
An error, how unexpected!
```

As a programmer you need to get used to reading error outputs. Looking at the
last line it says that there is a `Singular Matrix` error.  This happened
because the rows of the $A$ matrix are [linearly dependent](https://en.wikipedia.org/wiki/Linear_independence), $Row1 = Row2 + Row3$.

```{math}
A =
\begin{bmatrix} 
-R2-R3 & +R2 & +R3\\ -R2 & R2+R4 & -R4 \\ -R3 & -R4 & R3+R4 
\end{bmatrix}
```

This can be easily confirmed by taking the determinant of $A$:
```
Det = np.linalg.det(A)
```

## Conclusion

### On your own

* Try doing a nodal analysis.
* Try to catch a Pikachu, you can find him in Viridian Forest.

### In practice...

- The code written in `Part 1b` can indeed be written without any `for-loops` (use `range` to set the value of $V$).
- There are many occasions in which the `scipy.linalg` module is preferrable over the `numpy.linalg` module. For more information on this, check the [scipy.linalg Reference](https://docs.scipy.org/doc/scipy/reference/tutorial/linalg.html).

### Further reading

* https://numpy.org/doc/stable/user/
