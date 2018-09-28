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