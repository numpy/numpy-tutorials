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

# Analysis of air pollution levels before and after lockdown in Delhi, India


## What you'll do

Calculate Air Quality Indices (AQI) and perform paired t-test on them.

## What you'll learn

- You'll compute moving averages

- You'll calculate Air Quality Index (AQI)

- You'll perform a paired t-test and find the t and p values

- You'll learn to interpret these values


## What you'll need

- SciPy and Matplotlib installed in your environment

- Basic understanding of statistical terms like population, sample, mean, standard deviation etc.


***

```{code-cell} ipython3
# import necessary libraries

import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt

%matplotlib inline
```

## Building the dataset

We will import hourly measurements of standard pollutants: Particulate Matter (PM 10 and PM 2.5),
Nitrogen dioxide (NO2), ammonia (NH3), sulfur dioxide (SO2), carbon monoxide (CO), and ozone (O3), from a *.csv file
to ndarrays. This data has been collected in the period of 31/05/2019 to 30/06/2020.

```{code-cell} ipython3
! head air-quality-data.csv
```

We'll initially have two sets: *pollutants_A* and *pollutants_B* because we will need to compute their rolling
averages over different periods. Both the arrays contain 9528 rows of measured values for each pollutant.

```{code-cell} ipython3
# fetch the data

pollutants_A = np.genfromtxt("air-quality-data.csv", dtype='f8', delimiter=",", skip_header=1, usecols=(1, 2, 3, 4, 5))
pollutants_B = np.genfromtxt("air-quality-data.csv", dtype='f8', delimiter=",", skip_header=1, usecols=(6, 7))

print(pollutants_A.shape)
print(pollutants_B.shape)
```

```{code-cell} ipython3
# check for missing values

print(np.where(np.isnan(pollutants_A) == True))
print(np.where(np.isnan(pollutants_B) == True))
```

## Air Quality Index

To calculate the AQI, we have to first get the subindices that are calculated for each pollutant. To calculate
the sub_indices, we have to first have 24-hour averages for Particulate Matter (PM 10 and PM 2.5), Nitrogen dioxide (NO2), ammonia (NH3), and sulfur dioxide (SO2) and 8-hour averages for carbon monoxide (CO) and ozone (O3).

We have to be careful to truncate the first few rows of the 8-hour averages so that they match the time interval of the 24-hour averages.

After calculating the averages we will concatenate both the arrays to form a single array for all the pollutants.

```{code-cell} ipython3
# moving mean over 24 hours for Pollutants A and max of 8 hours avg. for Pollutants B

def moving_mean(a, n):
    ret = np.cumsum(a, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

pollutants_A_24hr_avg = moving_mean(pollutants_A, 24)
pollutants_B_8hr_avg = moving_mean(pollutants_B, 8)[-(pollutants_A_24hr_avg.shape[0]):]

# join both the arrays

pollutants = np.concatenate((pollutants_A_24hr_avg, pollutants_B_8hr_avg), axis=1)
```

The subindices for each pollutant are calculated according to the standard breakpoints. The final AQI is the maximum subindex for that row.

<!-- insert breakpoint table -->

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

# let the first dataset be from 24/03/2020 to 30/06/2020

after_lock = aqi_array[np.where(datetime >= np.datetime64('2020-03-24T00'))]

# let the second dataset be as many days before 24/03/2020

before_lock = aqi_array[np.where(datetime <= np.datetime64('2020-03-21T00'))][-(after_lock.shape[0]):]

print(after_lock.shape)
print(before_lock.shape)
```

We want to determine if the lockdown had any effect on the AQI in Delhi. Let's consider the null hypothesis as "There was no effect of the lockdown on AQI." Our corresponding alternative hypothesis will be "The lockdown had an effect on the AQI, and
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

- As our population was large enough, we could do a t-test which is used for normal distributions. There are better tests
for non-normal data like the Wilcoxon test.
