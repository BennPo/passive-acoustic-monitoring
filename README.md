# Passive Acoustic Monitoring

This project is a work in progress project to separate whale calls from background noise and detect them in recordings. Longer-term goals include species classification, individual animal recognition, and live hydrophone monitoring.


## Current progress

* Loads audio files and inspects sample rate, channels, and duration
* Generates a spectrogram from .wav file
* Overlays the clean whale calls with background noise for training purposes
* Clips the longer audio files into 5 second clips
* Generate dataset with clean whale and noisy whale input pairs
* Split recordings into training, validation, and test groups
* Train a whale-call seperator
* Evaluate seperator

## Next steps

* Increase the size of the seperator dataset
* Create detection dataset with different call types labeled
* Add whale detection after seperation