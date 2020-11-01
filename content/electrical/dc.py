## Part 0: Initializing
import numpy as np
import matplotlib.pyplot as plt

## ----------------------------
## Part 1: Plotting Ohms Law
## ----------------------------
'''
To understand how electricity works, think of a bowling ball rolling through a 
bowling alley. Attributes in this thought experiment include:
==> Speed of bowling ball (analgous to electric current)
==> Force used to roll bowling ball (analgous to electric voltage)
==> Friction of bowling alley (analgous to electric resistance)

* Assuming that the friction of the bowling alley is constant, the more force used 
to roll the bowling ball the faster it goes. This is also true in electrical 
circuits, given a constant resistance the larger the voltage the larger the current.

* On the other hand, assuming that the force used to roll the bowling ball is
constant, the higher the friction the slower the speed of the bowling ball. Again,
this is also true in electrical circuits, given a constant voltage the higher the
resistance the lower the current.

This relationship between voltage and current was discovered by Georg Ohm and it 
is now known as Ohm's Law. Mathematically, it is defined as: V = IR.

V ==> Voltage measured in Volts (V)
I ==> Current measured in Amperes (A)
R ==> Resistance measured in Ohms (ohms)

Alas, this explanation will probably not be in the final tutorial. (*Shakes fist at 
documentation gods*).
'''

## -------- Part 1a: Trivial
R = 10 # Resistance
V = 5 # Voltage
I = V/R # Current
print(f"I = {I} A")

## -------- Part 1b: Pure Python
R = 10
V = [] # Declaring V to be an empty list

# Assigning V integers from -5 to 5
for i in range(-5, 6):
    V.append(i)


I = [] # Declaring I to be an empty list

# Calculating I for all values of V
for voltage in V:
    I.append(voltage/R)

print(f"Voltages (V): {V}")
print(f"Current (A): {I}")

plt.clf()
plt.plot(V, I)                              
plt.savefig('imgs/plot1_basic.png')         
                                            
## -------- Part 1c: Using NumPy            
                                            
R = 10                                      
V = np.linspace(-5, 5, 11)                  
print(f"Voltage (V): {V}")
print(f"Length of V: {V.shape}")
print(f"Type of V: {type(V)}")

I = V/R

plt.clf()
plt.plot(V, I)
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.title('Ohms Law')
plt.grid()
plt.savefig('imgs/plot2_better.png')

## ----------------------------
## Part 2: Maximum Power Flow
## ----------------------------

## -------- Part 2a: Plotting maximum power flow
V = 8
Rth = 5

Rload = np.linspace(0.1, 50, 20)
Iload = V/(Rth + Rload)
Vload = Iload * Rload # Element wise multiplication
Pload = Iload * Vload

plt.clf()
plt.plot(Rload, Pload)
plt.xlabel('Rload (ohms)')
plt.ylabel('Pload (W)')
plt.title('Optimal Power Transfer')
plt.grid()  
plt.savefig('imgs/plot3_power.png')

## -------- Part 2b: Using indexing to find maximum power flow
Pmax = np.max(Pload)
Pmax_idx = np.argmax(Pload)
Rload[Pmax_idx] # This is weird, we expect 5 ohms

## ----------------------------
## Part 3: Mesh Analysis
## ----------------------------

## -------- Part 3a: 2x2
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

# Note that the matrix is symmetrical
A = np.array([[a11, a12],
              [a21, a22]])

b = np.array([[b1], [b2]])

# Way 1
Ainv = np.linalg.inv(A)
print(Ainv)
x = np.matmul(A, b)
x = Ainv @ b

# Way 2
x = np.linalg.solve(A, b)

## -------- Part 3b: Mesh Analysis 3x3
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

# Insert buzzlight year, brackets everywhere meme
A = np.array([[a11, a12, a13],
              [a21, a22, a23],
              [a31, a32, a33]])

b = np.array([[b1], [b2], [b3]])

Ainv = np.linalg.inv(A)
x = Ainv@b
x2 = np.linalg.solve(A, b)
print(x2)

## Part 4d: Mesh Analysis + Singular Matrix
R1 = 90
R2 = 30
R3 = 30
R4 = 5
V1 = 25
V2 = -65

a11 = R2
a12 = R3
a13 = -R2 -R3
a21 = -R2
a22 = R2+R4
a23 = -R4
a31 = -R3
a32 = -R4
a33 = R3+R4

b1 = V2-V1
b2 = V1
b3 = -V2

# Insert buzzlight year, brackets everywhere meme
A = np.array([[a11, a12, a13],
              [a21, a22, a23],
              [a31, a32, a33]])

b = np.array([[b1], [b2], [b3]])

det = np.linalg.det(A)

