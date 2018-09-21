# Classifying Sounds with a One-Dimensional Convolutional Neural Network

This notebook uses a Kaggle dataset (https://www.kaggle.com/mmoreaux/audio-cats-and-dogs) and partially reproduces a related kernel to explore CNN techniques without involving mel-spectrograms (i.e. reverting two dimensional convolutional analysis).  

## Result:
Validation accuracy hovered just above 70% while training accuracy continued to climb during additional epochs.  A holdout test set also achieved ~70% accuracy in predicting whether a particular 1-second segment contains the sound of a dog or cat.  The code also votes the probability outcomes of seconds belonging to the same audio file to produce a probability that the whole clip is dog or cat sounds (also ~70% accurate).


## Methodology
The generator augments the relatively limited samples (about 250 WAV files lasting 1.5 to about 20 seconds each) by indefinitely yielding 1-second segments of the audio (vectors of length 16,000) that start at random times.  This is done by first concatenating all the cat data into one long vector, doing the same for dogs and then normalizing each of the two long vectors.  Before being included in a batch, each segment is checked to make sure it contains some relevant information (by requiring the maximum value in a segment to be at least a certain value - we used 0.4 or 0.5, but this could be tuned more.)

The CNN has 10 one-dimensional convolutional layers with 10 filters each.  Kernal size 3 and stride 2 with "same" padding cut the 16,000-wide input down by half at each step, to 16-wide.  (Bach normalization was performed at each layer to limit vanishing/exploding gradients.) At the end, each of the ten filters is reduced to one value by global average pooling.  These values are passed to a final logistic regression to predict the probability that the recording is of a dog or a cat.  The number of trainable parameters is reasonably low at 3,041, so the model runs in 1-2 minutes per epoch.  Gradient steps per epoch were set at 100, which led to quick achievement of the terminal accuracy, but this could be tuned as well.

We also provide visualizations of example cat and dog audio files and the convolutional layers derived from them in succession.  
