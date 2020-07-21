# How-to: NumPy I/O
- [Read a .csv or other text file with no missing values](#read-a-csv-or-other-text-file-with-no-missing-values)
- [Read a .csv or other text file with missing values](#read-a-csv-or-other-text-file-with-missing-values)
  * [Non-whitespace delimiters](#non-whitespace-delimiters)
    + [Masked-array output](#masked-array-output)
    + [Array output](#array-output)
    + [Array output, specified fill-in value](#array-output-specified-fill-in-value)
  * [Whitespace-delimited](#whitespace-delimited)
- [Read a file in .npy or .npz format](#read-a-file-in-npy-or-npz-format)
- [Write to a file to be read back by NumPy](#write-to-a-file-to-be-read-back-by-numpy)
  * [Binary](#binary)
  * [Human-readable](#human-readable)
  * [Large arrays](#large-arrays)
- [Read an arbitrarily formatted binary file ("binary blob")](#read-an-arbitrarily-formatted-binary-file-binary-blob)
- [Write or read very large arrays](#write-or-read-very-large-arrays)
- [Write files for reading by other (non-NumPy) tools](#write-files-for-reading-by-other-non-numpy-tools)
- [Write or read a JSON file](#write-or-read-a-json-file)
- [Save/restore using a pickle file](#saverestore-using-a-pickle-file)
- [Convert from a pandas DataFrame to a NumPy array](#convert-from-a-pandas-dataframe-to-a-numpy-array)
- [Save/restore using `numpy.tofile` and `numpy.fromfile`](#saverestore-using-numpytofile-and-numpyfromfile)

## Read a .csv or other text file with no missing values

Use [numpy.loadtxt](https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html).

## Read a .csv or other text file with missing values

Use
[numpy.genfromtxt](https://numpy.org/doc/stable/user/basics.io.genfromtxt.html).

``numpy.genfromtxt`` will either return a
[masked array](https://numpy.org/doc/stable/reference/maskedarray.generic.html)
masking out missing
values (if ``usemask=True``) or will fill in the missing value with the value
specified in ``filling_values`` (default is ``np.nan`` for float, -1 for int).

### Non-whitespace delimiters

```
$ cat csv.txt
1, 2, 3
4,, 6
7, 8, 9
```
#### Masked-array output
```
>>> np.genfromtxt("csv.txt", delimiter=",", usemask=True)
masked_array(
  data=[[1.0, 2.0, 3.0],
        [4.0, --, 6.0],
        [7.0, 8.0, 9.0]],
  mask=[[False, False, False],
        [False,  True, False],
        [False, False, False]],
  fill_value=1e+20)
```
#### Array output
```
>>> np.genfromtxt("csv.txt", delimiter=",")
array([[ 1.,  2.,  3.],
       [ 4., nan,  6.],
       [ 7.,  8.,  9.]])
```
#### Array output, specified fill-in value
```
>>> np.genfromtxt("csv.txt", delimiter=",", dtype=np.int8, filling_values=99)
array([[ 1,  2,  3],
       [ 4, 99,  6],
       [ 7,  8,  9]], dtype=int8)
```
### Whitespace-delimited

``numpy.genfromtxt`` can also parse whitespace-delimited data files
that have missing values if

 * each field has a fixed width: use the width as the `delimiter` argument
```
## File with width=4. The data does not have to be justified (for example, the
## 2 in row 1), the last column can be less than width (for example, the 6 in
## row 2), and no delimiting character is required (for instance 8888 and 9 in row 3)

$cat fixedwidth.txt
1   2      3
44      6
7   88889

## Showing spaces as '^'
$ tr ' ' '^' < fixedwidth.txt
1^^^2^^^^^^3
44^^^^^^6
7^^^88889

>>> np.genfromtxt("fixedwidth.txt",delimiter=4)
array([[1.000e+00, 2.000e+00, 3.000e+00],
       [4.400e+01,       nan, 6.000e+00],
       [7.000e+00, 8.888e+03, 9.000e+00]])
```
 * a special value (e.g. "x") indicates a missing field: use it as the `missing_values` argument
```
$ cat nan.txt
1 2 3
44 x 6
7  8888 9

>>> np.genfromtxt("nan.txt",missing_values='x')
array([[1.000e+00, 2.000e+00, 3.000e+00],
       [4.400e+01,       nan, 6.000e+00],
       [7.000e+00, 8.888e+03, 9.000e+00]])
```
 * you want to skip the rows with missing values: set `invalid_raise=False`
```
$ cat skip.txt
1 2   3
44    6
7 888 9

>>> np.genfromtxt("skip.txt",invalid_raise=False)
__main__:1: ConversionWarning: Some errors were detected !
    Line #2 (got 2 columns instead of 3)
array([[  1.,   2.,   3.],
       [  7., 888.,   9.]])
```

 * the delimiter whitespace character is different from the whitespace that
   indicates missing data. For instance, if columns are delimited by `\t`,
   then missing data will be recognized if it consists of one
   or more spaces
```
$ cat tabs.txt
1 2 3
44    6
7 888 9

## Showing the tabs (^I) and spaces
$ cat -T tabs.txt
1^I2^I3
44^I ^I6
7^I888^I9

>>> np.genfromtxt("tabs.txt",delimiter="\t",missing_values=" +")
array([[  1.,   2.,   3.],
       [ 44.,  nan,   6.],
       [  7., 888.,   9.]])
```

## Read a file in .npy or .npz format

Use [numpy.load](https://numpy.org/doc/stable/reference/generated/numpy.load.html).

## Write to a file to be read back by NumPy
### Binary
Use [numpy.save](https://numpy.org/doc/stable/reference/generated/numpy.save.html) to store a single array,
[numpy.savez](https://numpy.org/doc/stable/reference/generated/numpy.savez.html) or
[numpy.savez_compressed](https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html#numpy.savez_compressed)
to store multiple arrays. For [security and portability](#saverestore-using-a-pickle-file), set `allow_pickle=False`.

Masked arrays [can't currently be saved](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.tofile.html),
nor can other arbitrary array subclasses.

### Human-readable
`save` and `savez` create binary files. To write a human-readable file, use
[numpy.savetxt](https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html).
The array can only be 1- or 2-dimensional, and there's no `savetxtz` for multiple files.

### Large arrays
See [Write or read large arrays](#large_arrays).

## Read an arbitrarily formatted binary file ("binary blob")

Use a [structured array](https://numpy.org/doc/stable/user/basics.rec.html).

**Example:**

The `.wav` file header is a 44-byte block preceding `data_size` bytes of the
actual sound data:

```
chunk_id         "RIFF"
chunk_size       4-byte unsigned little-endian integer
format           "WAVE"
fmt_id           "fmt "
fmt_size         4-byte unsigned little-endian integer
audio_fmt        2-byte unsigned little-endian integer
num_channels     2-byte unsigned little-endian integer
sample_rate      4-byte unsigned little-endian integer
byte_rate        4-byte unsigned little-endian integer
block_align      2-byte unsigned little-endian integer
bits_per_sample  2-byte unsigned little-endian integer
data_id          "data"
data_size        4-byte unsigned little-endian integer
```
The `.wav` file header as a NumPy structured dtype:
```
wav_header_dtype = np.dtype([
    ("chunk_id", (bytes, 4)), # flexible-sized scalar type, item size 4
    ("chunk_size", "<u4"),    # little-endian unsigned 32-bit integer
    ("format", "S4"),         # 4-byte string
    ("fmt_id", "S4"),
    ("fmt_size", "<u4"),
    ("audio_fmt", "<u2"),     #
    ("num_channels", "<u2"),  # .. more of the same ...
    ("sample_rate", "<u4"),   #
    ("byte_rate", "<u4"),
    ("block_align", "<u2"),
    ("bits_per_sample", "<u2"),
    ("data_id", "S4"),
    ("data_size", "u4"),
    #
    # the sound data itself cannot be represented here:
    # it does not have a fixed size
   ])

data = np.load(f,dtype=wave_header_dtype)
```
Credit: [Pauli Virtanen](http://scipy-lectures.org/advanced/advanced_numpy/index.html)

<a name="large_arrays">

## Write or read large arrays

Arrays too large to fit in memory can be treated like ordinary in-memory arrays using
[numpy.mmap](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html).
```
import numpy as np
array = np.memmap("mydata/myarray.arr", mode="r",
                  dtype=np.int16, shape=(1024, 1024))
```
The simplicity also means it's possible to have access patterns that don't
match `memmap`'s buffering and therefore access disk repeatedly.
[Zarr](https://zarr.readthedocs.io/en/stable/) and the similar
[HDF5](https://portal.hdfgroup.org/display/support) formats allow you to
manage buffering optimally for your program's access pattern.

Itamar Turner-Trauring in
[pythonspeed.com](https://pythonspeed.com/articles/mmap-vs-zarr-hdf5/) gives
more details and describes tradeoffs among memmap, Zarr, and HDF5.

* For **HDF5**, use [h5py](https://www.h5py.org/) or [PyTables](https://www.pytables.org/).
* For **Zarr**, see [here](https://zarr.readthedocs.io/en/stable/tutorial.html#reading-and-writing-data).
* For **NetCDF**, use [scipy.io.netcdf_file](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.netcdf_file.html).

## Write files for reading by other (non-NumPy) tools

Formats for exchanging data with other tools include HDF5, Zarr, and NetCDF.

## Write or read a JSON file

NumPy arrays are not directly [JSON serializable](https://github.com/numpy/numpy/issues/12481).

## Save/restore using a pickle file

Not recommended, due to lack of security and portability.

 * security: not secure against erroneous or maliciously constructed data
 * portability: may not be loadable on different Python installations

Use `np.save` and `np.load`, setting ``allow_pickle=False``.

## Convert from a pandas DataFrame to a NumPy array
`DataFrame.to_numpy()`

## Save/restore using `numpy.tofile` and `numpy.fromfile`

In general, prefer `numpy.save` and `numpy.load`.
[numpy.tofile](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tofile.html)
and
[numpy.fromfile](https://numpy.org/doc/stable/reference/generated/numpy.fromfile.html)
lose information on endianness and precision and so are unsuitable for anything but scratch storage.
