---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Analyzing the impact of the lockdown on air quality in Delhi, India


## What you'll do

Calculate Air Quality Indices (AQI) and perform paired t-test on them.

## What you'll learn

- You'll learn the concept of moving averages

- You'll learn how to calculate Air Quality Index (AQI)

- You'll learn how to perform a paired t-test and find the t and p values

- You'll learn to interpret these values


## What you'll need

- SciPy and Matplotlib installed in your environment

- Basic understanding of statistical terms like population, sample, mean, standard deviation etc.


***

+++

## The problem of air pollution

Air pollution is one of the most prominent types of pollution we face that has an immediate effect on our daily lives. The
COVID-19 pandemic resulted in lockdowns in different parts of the world, offering a rare opportunity to study the effect of
human activity (or lack there of) on air pollution. In this tutorial, we will study the air quality in Delhi, one of the
worst affected cities by air pollution, before and during the lockdown from March to June 2020. For this, we will first compute
the Air Quality Index for each day from the collected pollutant measurements. Next, we will sample these indices and perform
a [paired t-test](https://en.wikipedia.org/wiki/Student%27s_t-test#Dependent_t-test_for_paired_samples) on them. It will statistically show us that the air quality improved due to the lockdown, supporting our
intuition.

Let's start by importing the necessary libraries into our environment.

```{code-cell} ipython3
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt

%matplotlib inline
```

## Building the dataset

We will use a condensed version of the [Air Quality Data in India](https://www.kaggle.com/rohanrao/air-quality-data-in-india) dataset. The dataset contains air quality data and AQI (Air Quality Index) at hourly and daily level of various stations across multiple cities in India. The condensed version available with this tutorial contains hourly pollutant measurements for Delhi
from 31/05/2019 to 30/06/2020. It has data of the standard pollutants that are required for Air Quality Index calculation:
Particulate Matter (PM 10 and PM 2.5), Nitrogen dioxide (NO2), ammonia (NH3), sulfur dioxide (SO2), carbon monoxide (CO), and ozone (O3).

```{code-cell} ipython3
! head air-quality-data.csv
```

We'll initially create two sets: `pollutants_A` with PM 2.5, PM 10, NO2, NH3, and SO2, and `pollutants_B` with CO and O3. The
two sets will be processed slightly differently, as you'll see later on.

```{code-cell} ipython3
pollutants_A = np.genfromtxt("air-quality-data.csv", dtype='f8', delimiter=",", skip_header=1, usecols=(1, 2, 3, 4, 5))
pollutants_B = np.genfromtxt("air-quality-data.csv", dtype='f8', delimiter=",", skip_header=1, usecols=(6, 7))

print(pollutants_A.shape)
print(pollutants_B.shape)
```

Our dataset might contain missing values, denoted by `NaN`, so let's do a quick check with [np.isnan](https://numpy.org/devdocs/reference/generated/numpy.isnan.html).

```{code-cell} ipython3
# check for missing values

print(np.where(np.isnan(pollutants_A) == True))
print(np.where(np.isnan(pollutants_B) == True))
```

We have successfully imported the data and it is complete. Let's move on to the AQI calculations!

+++

## Calculating the Air Quality Index


We will calculate the AQI using [the method](http://safar.tropmet.res.in/AQI-47-12-Details) adopted by the [Central Pollution Control Board](https://www.cpcb.nic.in/national-air-quality-index) of India.  To summarize the steps:

- Collect 24-hourly average concentration value for the standard pollutants; 8-hourly in case of CO and O3.


- Calculate the sub-indices for these pollutants with the formula: 
$Ip = \dfrac{\text{IHi – ILo}}{\text{BPHi – BPLo}}\cdot{\text{Cp – BPLo}} + \text{ILo}$

    Where,

    `Ip` = sub-index of pollutant p\
    `Cp` = averaged concentration of pollutant p\
    `BPHi` = concentration breakpoint i.e. greater than or equal to Cp\
    `BPLo` = concentration breakpoint i.e. less than or equal to Cp\
    `IHi` = AQI value corresponding to BPHi\
    `ILo` = AQI value corresponding to BPLo
    

- The maximum sub-index at any given time is the Air Quality Index.
    
The Air Quality Index is calculated with the help of breakpoint ranges as shown in the chart below.
![Chart of the breakpoint ranges](../_static/10-breakpoints.png)


For the first step, we have to compute [moving averages](https://en.wikipedia.org/wiki/Moving_average) for `pollutants_A` over a window of 24 hours and `pollutants_B` over a
window of 8 hours. We will write a simple function `moving_mean` using [np.cumsum](https://numpy.org/devdocs/reference/generated/numpy.cumsum.html) and [sliced indexing](https://numpy.org/devdocs/user/basics.indexing.html#slicing-and-striding) to achieve this.

To make sure both the sets are of the same length, we will truncate the `pollutants_B_8hr_avg` according to the length of
`pollutants_A_24hr_avg`. This will also ensure we have concentrations for all the pollutants over the same period of time.

```{code-cell} ipython3
def moving_mean(a, n):
    ret = np.cumsum(a, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

pollutants_A_24hr_avg = moving_mean(pollutants_A, 24)
pollutants_B_8hr_avg = moving_mean(pollutants_B, 8)[-(pollutants_A_24hr_avg.shape[0]):]
```

Now we can join both sets with [np.concatenate](https://numpy.org/devdocs/reference/generated/numpy.concatenate.html) to form a single data set of all the averaged concentrations. Note that we have to join our arrays column-wise so we pass the
`axis=1` parameter.

```{code-cell} ipython3
pollutants = np.concatenate((pollutants_A_24hr_avg, pollutants_B_8hr_avg), axis=1)
```

The subindices for each pollutant are calculated according to the standard breakpoints. The final AQI is the maximum subindex for that row.

```{code-cell} ipython3
# Calculate sub_indices

AQI = np.array([0, 51, 101, 201, 301, 401, 501])

breakpoints = {
    'PM2.5': np.array([0, 31, 61, 91, 121, 251]),
    'PM10': np.array([0, 51, 101, 251, 351, 431]),
    'NO2': np.array([0, 41, 81, 181, 281, 401]),
    'NH3': np.array([0, 201, 401, 801, 1201, 1801]),
    'SO2': np.array([0, 41, 81, 381, 801, 1601]),
    'CO': np.array([0, 1.1, 2.1, 10.1, 17.1, 35]),
    'O3': np.array([0, 51, 101, 169, 209, 749])
}

def compute_indices(pol, con):
    bp = breakpoints[pol]
    
    if bp[0] <= con < bp[1]:
        Bl = bp[0]
        Bh = bp[1] - 1
        Ih = AQI[1] - 1
        Il = AQI[0]

    elif bp[1] <= con < bp[2]:
        Bl = bp[1]
        Bh = bp[2] - 1
        Ih = AQI[2] - 1
        Il = AQI[1]

    elif bp[2] <= con < bp[3]:
        Bl = bp[2]
        Bh = bp[3] - 1
        Ih = AQI[3] - 1
        Il = AQI[2]

    elif bp[3] <= con < bp[4]:
        Bl = bp[3]
        Bh = bp[4] - 1
        Ih = AQI[4] - 1
        Il = AQI[3]

    elif bp[4] <= con < bp[5]:
        Bl = bp[4]
        Bh = bp[5] - 1
        Ih = AQI[5] - 1
        Il = AQI[4]

    elif bp[5] <= con:
        Bl = bp[5]
        Bh = bp[5] + bp[4] - 2
        Ih = AQI[6]
        Il = AQI[5]

    else:
        print("Cc out of range!")
        
    return ((Ih - Il) / (Bh - Bl)) * (con - Bl) + Il

# vectorize the function

vcompute_indices = np.vectorize(compute_indices)

# calculate sub_indices

# figure out what is wrong with CO

sub_indices = np.stack((vcompute_indices('PM2.5', pollutants[..., 0]),
                        vcompute_indices('PM10', pollutants[..., 1]),
                        vcompute_indices('NO2', pollutants[..., 2]),
                        vcompute_indices('NH3', pollutants[..., 3]),
                        vcompute_indices('SO2', pollutants[..., 4]),
                        vcompute_indices('O3', pollutants[..., 6])), axis=1)

# calculate AQIs

aqi_array = np.amax(sub_indices, axis=1)
```

## Paired t-test on the AQIs

We will now import the datetime column into a *datetime64* dtype array. We will use this array to index the AQI array and
obtain sub-sets of the dataset. We will have a dataset of around 2 months before and after the lockdown.

```{code-cell} ipython3
# datetime_data is from 31/05/2019 to 30/06/2020

datetime = np.genfromtxt("air-quality-data.csv", dtype='M8[h]', delimiter=",", skip_header=1, usecols=(0, ))[-(pollutants_A_24hr_avg.shape[0]):]

# lockdown in Delhi started on 24th March 2020

# let first dataset be from 24/03/2020 to 30/06/2020

after_lock = aqi_array[np.where(datetime >= np.datetime64('2020-03-24T00'))]

# let second dataset be as many days before 24/03/2020

before_lock = aqi_array[np.where(datetime <= np.datetime64('2020-03-21T00'))][-(after_lock.shape[0]):]

print(after_lock.shape)
print(before_lock.shape)
```

We want to determine if the lockdown had any effect on the AQI in Delhi. Let's consider the null hypothesis as "There was no effect of the lockdown on AQI." Our corresponding alternative hypothesis will be "The lockdown had an effect on the AQI and
they were improved." To accept or reject our null hypothesis, we will do a one-tailed paired t-test.

```{code-cell} ipython3
# make samples from before_lock, after_lock

before_sample_A = np.random.permutation(before_lock)[:50]
after_sample_A = np.random.permutation(after_lock)[:50]

# do a paired t-test

dof = 49

def t_test(x, y):
    diff = x - y
    return np.mean(diff) / (np.std(diff) / np.sqrt(len(x)))

t_value = t_test(before_sample_A, after_sample_A)

p_value = 1 - t.cdf(np.abs(t_value), dof)

print("The t value is {} and the p value is {}.".format(t_value, p_value))
```

## What do the t and p values mean?

Since our p value is less than alpha i.e., 0.05, we can reject our null hypothesis.

+++

***

## In practice...

- The pandas library is preferable to use for time-series data analysis

- As our population was large enough we could do a t-test which is used for normal distributions. There are better tests
for non-normal data like the [Wilcoxon test](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test).
