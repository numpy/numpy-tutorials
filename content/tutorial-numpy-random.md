**Random Numbers in Numpy**

Random number does NOT mean a different number every time. Random means something that can not be predicted logically.

**Generate Random Number**

`from numpy import random`

`x=random.randint(100)`

`print(x)`

The above code generates a random integer from 0 to 100.

**Generate Random Float**

`x=random.rand()`

**Generate Random Array**

randint() methods takes a `size` parameter where you can
specify the shape of an array.

`x=random.randint(100, size=(5))`

the above line of code generates a 1D array containing 5 random integers from 0 to 100.

now, let's generate a 2D array with 2 rows and each row containing 4 random integers from 0 to 100.

`x=random.randint(100, size=(2, 4))`


**Floats**

`rand()` method also allows you to specify the shape of the array.

`x = random.rand(5)`

`x = random.rand(3, 5)`