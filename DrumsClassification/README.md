# Drums classification for musical beat reconstruction

In this project, I'm trying to identify drums on a loop beat in order to reconstruct it from scratch.

## Using the dataset

The first step is to train a model to recognize different types of drums, like so :
* Kick
* Snare
(Optional)
* HHCl
* HHOp
* Tom
* Rimshot
* Cowbell
* Clap

The features used to classify these drums are :
* Total time
* Frequency peak in attack + decay
(Optional)
* Attack time
* Decay time
* Release time
* Relative power
* Entropy in freauency bands ?

## First results

I'm trying to compare a simple kNN ML model and a Perceptron model, here are the first results, using:
* 2 Classes (Kick, Snare)
* 2 features (Total time, Freq. peak in AD)

| Classes / Features  |   kNN   |  Perceptron   |
| ------------------- | ------- | ------------- |
| 2 / 2               | 56,25%  | Content Cell  |
| ------------------- | ------- | ------------- |
| Content Cell        | Conten  | Content Cell  |