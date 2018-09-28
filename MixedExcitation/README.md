# Multiband Excitation Vodcoder

## Running the Code

To run the code, from the command line type

```
python vocoder.py
```

This will produce a .wav file called testout.wav which can be played on programs like VLC media player

## Changing the Input file

If you want to change the input file which gets recontructed simply change the name of the file on line 9 in vocoder.py

```python
fs, datafile = wavfile.read('name_of_new_file.wav')
```

## Adding Noise

To add gaussian white noise to the original input signal uncomment the following code and modify the mean and variance.

**Line 15:**
```python
datafile = datafile + np.random.normal(mean,variance, number of data points)
```
Once you have uncommented the above lines re-run the code from the command line using
```
python vocoder.py
```

## Printing Plots

To enable the plots to print while running uncomment the following code:

**Line 97-99:**
```python
plt.plot(refinedPitches)
plt.title(Title)
plt.savefig(filename)
```

**Line 132-136:**
```python
plt.cla()
plt.clf()
plt.plot(refP_error_per_frame)
plt.title(Title)
plt.savefig(filename)
```
Once you have uncommented the above lines re-run the code from the command line using
```
python vocoder.py
```

