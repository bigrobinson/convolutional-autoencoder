# convolutional autoencoder
A convolutional autoencoder and associated utilies to train the network. This model was developed specifically for the
compression/distillation of Flappy Bird game frames, so layers are currently sized to take inputs of 288 x 512 RGB images.
Images of different resolution will be rescaled to 288 x 512.

## System Setup
This code was developed and tested on Ubuntu 18.04, using Python 3.5 and Pytorch 1.3.1.

## Run the Code with Defaults
1. [optional] Clone the [Flappy Bird Deep Q Learning repo] (https://github.com/uvipen/Flappy-bird-deep-Q-learning-pytorch)  `git clone https://github.com/uvipen/Flappy-bird-deep-Q-learning-pytorch`
2. [optional] Run the game with the trained DQN model.  
`python3.5 test.py`
You'll notice this version has no background. If you want the background go grab a different repo or run it in pygame.
3. While the game is running, use something like [vokoscreen] (https://github.com/vkohaupt/vokoscreen) to grab frames and
and save them to the train_data/flappy_bird directory.  
`sudo apt-get install vokoscreen`
4. Add the filenames of the images to train_data/flappy_bird_images.txt.
Modify directories in train.py as necessary to point to your local directory.
5. `python3.5 train.py`

## Options
There are currently two optimizers available: Adam and SGD. The model in saved_models was trained with Adam and the default
parameters. The SGD option includes some extras for allowing a burn-in period among other things.
