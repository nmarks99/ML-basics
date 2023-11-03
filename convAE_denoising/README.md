# Image Denoising

*Dataset:* [Sports images](https://www.kaggle.com/datasets/gpiosenka/sports-classification)

In this example, we train a neural network to remove noise (synthetically generated) from an image.
For the network architecture, we have chosen to use a convolutional autoencoder. The dataset linked above
is composed of images of sports, and text labels of what sport is in the image. For our problem however,
we will be manually adding synthetic noise to the images and using these noisy images as inputs. The labels
will be the original noise free images. The model's goal is to make the input noisy image as close as possible
to the noise free label image.
