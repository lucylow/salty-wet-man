# Salty Wet Man &#x1F499;

**Not Suitable for Work (NSFW) image classification using Keras and Tensorflow.js**

<div>
  
  [![Status](https://img.shields.io/badge/status-work--in--progress-success.svg)]()
  [![GitHub Issues](https://img.shields.io/github/issues/lucylow/salty-wet-man.svg)](https://github.com/lucylow/salty-wet-man/issues)
  [![GitHub Pull Requests](https://img.shields.io/github/issues-pr/lucylow/salty-wet-man.svg)](https://github.com/lucylow/salty-wet-man/pulls)
  [![License](https://img.shields.io/bower/l/bootstrap)]()

</div>

![alt text](https://github.com/lucylow/salty-wet-man/blob/master/readme-images/salty_wet_man.png)

---
## Introduction &#x1F499; 

* Defining NSFW material is subjective and the task of identifying these images is **non-trivial**
* Identification into two categories:

  * [SFW] positively trained for **neutral images** that are safe for work
  
  * [NSFW] negatively trained for **pornographic images** involving sexually explicit images


## Machine Learning Classifier &#x1F499;

* Machine Learning methods for object recognition  
  * Labeled image-training datasets 
  1) Small image datasets
    * Order of tens of thousands of images
    * Classic example MNIST digit-recognition task - best error rate
  2) Large image datasets
    * Order of hundreds of thousants of images
    * ImageNet or LabelMe
* Convolutional Neural Networks (CNN)
  * theoretically best for large learning capcity and completixty
  * stationarity of statistics
  * locality of pixel dependencies


---

## GPU IMplementation &#x1F499;

* CNN + image datasets = heavy computation required
* current GPUs 
* higly optomized implementation of 2D convolutions
* Size of CNN network limited by GPU memory avaliabe
  * Spread network over multiple GPUs via parallel processing 

---

## VGG16 model &#x1F499;
* training images, validation images, and testing images
*
*
---


## Overfitting &#x1F499;

* Large size of network with 1.2+ million labeled image training examples == overfitting
* Data augmentation: 
  * Label-peserving transfomations: Transformed images do not need to be stored on the GPU disk. Image translation and horizontal reflections
  * RGB channel intensities: Add a transformation (covariance matrix) to each RGB image pixel. Object idenity is invariant to changes in intensity/colour of images.
* Dropout
  * ReLu neutrons 
  * dropout is used fr first two fully-connected layers
  
  
---

## Technical Installations - requires heavy computation &#x1F499;

1. Install Python dependencies and packages (**Keras, TensorFlow, and TensorFlow.js**) - best to run from [virtualenv](https://virtualenv.pypa.io/en/latest/)
   
2. Download and convert the VGG16 model to TensorFlow.js format

3. Launch Node.js script to **load converted model** and compute **maximally-activating input images** for  convnet's filters using gradient ascent in the input space. Save image files under `dist/filters` directory 
   
4. Launch Node.js script to **calculate internal convolutional layers' activations** and gradient-based **Class Activation Map (CAM)**. Save image files under `dist/activation` directory
   
5. Compile. Launch web view at **https://lucylow.github.io/salty-wet-man/**


---


## Technical Visualizations &#x1F499;


```sh
yarn visualize
```


Increase the **number of filters to visualize per convolutional layer** from default 8 to larger value (ex. 18):


```sh
yarn visualize --gpu --filters 18
```


Default image used for **internal-activation** and **CAM visualization** is **"nsfw.jpg"**. Switch to another image by using the **"--image waifu-pic.jpeg"** 👀


```sh
yarn visualize --image waifu-pic.jpeg
```


---

## References &#x1F499;

* Alex Krizhevsky. 2012. ImageNet Classification with Deep Convolutional Networks
* Innes and Yuret. Machine Learning with Julia. https://www.youtube.com/watch?v=21_wokgnNog
* Keras VGG Documentation. https://keras.io/applications/#extract-features-with-vgg16
* Tensorflow tf.keras API. https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG16
* Yahoo Engineering's Caffe DL library and CaffeOnSpark model. https://yahooeng.tumblr.com/post/151148689421/open-sourcing-a-deep-learning-solution-for
* Gabriel Goh. Image Synthesis from Yahoo's open_nsfw. https://open_nsfw.gitlab.io/
* Client-Side NSFW Classification. https://nsfwjs.com/
* Finding Naked People. 1996. http://luthuli.cs.uiuc.edu/~daf/papers/naked.pdf
* https://machinelearningmastery.com/use-pre-trained-vgg-model-classify-objects-photographs/



---


**┬┴┬┴┤ʕ•ᴥ├┬┴┬┴**





