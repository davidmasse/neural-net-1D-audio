# Classifying Sounds and Peeling the Layers of a One-Dimensional Convolutional Neural Network


## Overview

This notebook uses a Kaggle dataset (https://www.kaggle.com/mmoreaux/audio-cats-and-dogs) and takes a related Kaggle kernel as a point of departure to explore CNNs. The methods pursued do not involve mel-spectrograms, which turn audio classification into an image classification exercise.  Once a model is trained and hyperparameters tuned, examples from the held-out test set illustrate the workings of the network's layers.

Each data file has 1-20 or so seconds of audio recorded at 16kHz (i.e a time series where a value is recorded every 16,000th of a second).  Here are plots of two dog files (labeled "1") and two cat files (labeled "0") where meows and barks can be clearly seen:

![audio_beginning](/img/audio_beginning.png)


## Generator of Audio Sample Batches to Feed into the CNN

The generator function (with its two helper functions) augments the relatively limited samples (about 250 WAV files lasting 1.5 to about 20 seconds each) by indefinitely yielding 1-second segments of the audio (vectors of length 16,000) that start at random times.  This is done by first concatenating all the cat-sound data into one long vector, doing the same for dogs and then normalizing each of the two long vectors.  Before being included in a batch, each segment is checked to make sure it contains some relevant information (by requiring the maximum value in a segment to be at least a certain value - we used 0.4 or 0.5, but this could be tuned more.)


## Choice of Architecture

The basic architecture has 10 one-dimensional convolutional layers with 10 filters each.  This is because, typically, we will use `kernel_size` and `stride` parameters that cut the size of the outputs of each layer down by half, so the 10 layers bring each 16,000-long input vector down to a 16-long vector.  This much shorter vector is averaged by a global average pooling layer, outputting 10 averages (one for each filter).  These 10 averages are fed into a single-node layer with a sigmoid function outputting the model's estimation of the probability of the sound's being from a dog (the positive or "1" case).  10 filters were used because, on the one hand, a model with 5 features had validation accuracies 20% lower than training accuracies (though final test accuracy was a respectable 68%).  On the other hand, more than 10 filters led to unreasonable training times for such a simple task.

`kernel_size` 3, 4 and 5 were tried with `stride` 1, 2 or 3.  Best performance came from `kernel_size` 4 or 5 with `stride` 2.  `padding` was kept at "same."    Batch normalization was performed at each layer to limit vanishing/exploding gradients.  The number of trainable parameters was reasonably low at 1,000-5,000, so the model runs in 1-2 minutes per epoch.  Gradient steps per epoch were generally set at 75 (only 50 for smaller validation set).  100 steps per epoch led to quick convergence to terminal accuracy while 75/50 spread the learning over 15 epochs or so.


## Result

Maximum validation accuracy hovered around 81-84% while holdout test accuracy peaked at 79% (in predicting whether a particular 1-second segment contains the sound of a dog or cat).  The code also votes the probability outcomes of seconds belonging to the same audio file to produce a probability that the whole clip is dog or cat sounds (~70% accurate).

![model_acc](/img/model_acc.png)


## Layers

It helps to understand what each layer is doing if we think about the parameters being trained in the `model.summary()`:

![model_summary_1](/img/model_summary_1.png)
![model_summary_2](/img/model_summary_2.png)

For the first convolutional layer, with `kernel_size` 4, there is only one input (of length 16,000); but there are 4 weights to calculate (the weights that the kernel applies to convolve the audio) for each of 10 filters, so 40 weights plus 10 biases (one for each filter) is 50 parameters.  From what I gather, in order to allow forward- and back-propagation, batch normalization seems to calculate 4 parameters (2 means and 2 variances?) per filter (so 40 after each convolutional layer).  Half of the batch normalization parameters (which number 400, since there are 40 for each of 10 layers) are non-trainable.

The second through tenth convolutional layers each has input of 10 convolved audio segments (output from previous layer), and these are used to train 10 filters (of width 4), so we have 400 weights plus 10 biases (one for each filter).  After global average pooling, the final dense layer has 10 weights (one for each input) and one bias parameter.

Here is an example dog 1-second segment:

![dog_main_ex](/img/dog_main_ex.png)

The model estimates 79% probability that this is a dog:

![predict_dog_test](/img/predict_dog_test.png)

Here is the activation output of the first convolutional layer (10 filters).  As can be seen, the length has shrunk from 16,000 to 8,000:

![activ_1](/img/activ_1.png)

And here are the rest of the activation outputs:

![activ_2](/img/activ_2.png)
![activ_3](/img/activ_3.png)
![activ_4](/img/activ_4.png)
![activ_5](/img/activ_5.png)
![activ_6](/img/activ_6.png)
![activ_7](/img/activ_7.png)
![activ_8](/img/activ_8.png)
![activ_9](/img/activ_9.png)
![activ_10](/img/activ_10.png)

After the global average pooling layer, these are the 10 outputs.  Typically dog results at this point have a strong component 3:

![dog_main_pooling_output](/img/dog_main_pooling_output.png)

Here's another dog sample with a strong component 3:

![global_pooling_second_dog](/img/global_pooling_second_dog.png)

This lines up with a high weight on component 3 in the sigmoid model, both pushing the probability of "dog" up.  Here's the sigmoid weights:

![sigmoid_weights](/img/sigmoid_weights.png)

The global average pooling output for cats, in contrast, tends to have strong components 1 and 6, which line up with low weights in the sigmoid layer, leading to low probability of "dog" (i.e. high probability of "cat"):

![global_pooling_cat_1](/img/global_pooling_cat_1.png)

The filter weights can also be visualized, at least for the first convolutional layer as seen below.  We can see how different intensities of value increases and decreases (i.e. edges) would be detected by these filters.  (There are a 100 sets of 4 filter weights for each layer from 2 to 10, so we won't visualize those.)

![filter_weights](/img/filter_weights.png)
