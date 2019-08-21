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


---


## Convolutional Neural Networks (CNN) &#x1F499;
  
**Image Datasets**

  * **Theoretically CNN is best** since large learning capacity and complexity
  * Stationarity of statistics
  * Locality of pixel dependencies  

**NSFW Images**

  * Static images
  * Uncontrolled backgrounds
  * Multiple people and partial figures
  * Different camera angles


**GPU Implementation**

  * **Heavy computation required** - Size of CNN network limited by GPU memory avaliabe
  * Highly optimized implementation of 2D convolutions
  * Solution to spread network over multiple GPUs via **parallel processing**
  

---

## Object Recognition &#x1F499;

**Labeled image-training datasets**
  
  * Small image datasets (order of tens of thousands of images) -  MNIST digit-recognition with best error rate  
  * Large image datasets (order of hundreds of thousands of images) - ImageNet
 

**ImageNet used for Large Scale Object Recognition**

  * **Dataset over 15 million labeled images**
  * Variable-resolution images (256x256)
  * Training, validation, and testing images
  * Annual competition - ImageNet Large-Scale Visual Recognition Challenge (ILSVRC) 
  
---  

## NSFW Object Recognition: Content-Based Retrival &#x1F499;


**Image Location with Large Areas of Skin-colored Regions**

  * Skin region properties - image, color, and texture  
  * **Input RGB values (skin)** with log-opponent representation
  * Intensity of image (texture) smooth-ed with median filter, then subtracted from original image
  
  * **Query By Image Content (QBIC)**
    * Absraction of an image to search for colored textured regions
    * Uses image decomposition, pattern matching, and clustering algorithms
    * Find a set of images similar to a query image


![Image retrival algorithm](https://github.com/lucylow/salty-wet-man/blob/master/readme-images/QBIC%20pipeline.png)

**Elongated Regions Grouping**

  * Group **2D and 3D constraints** on body/skin regions
  * Model human body == **cylindrical parts within skeleton geometry**
  * Identify region outline


**Classify Regions into Human Limbs**

  * **Geometric grouping algorithms** - matching view to collection of images of an object
  * Make hypothesis object present, and an estimate of appearance via **future vector from compressed image**
  * Minimum distance classifer to match feature vectors


---

## Neural Network Classifier - VGG16 model &#x1F499;

* VGG16 is a CNN for large-scale image recognition 
* **Model achieves 92.7% top-5 test accuracy on ImageNet**
* Implemented with Keras and Tensorflow backend in this project


<p align="center"><img  src="https://user-images.githubusercontent.com/17570785/50308873-e2eb6e00-049c-11e9-9587-9da6bdec011b.png" ></p>

[Image of VGG16 architecture](https://github.com/lucylow/PlotNeuralNet/blob/master/examples/VGG16/vgg16.pdf)

**VGG16 Architecture**

* **Fixed input of 224 x 224 RGB image**
* Three fully-connected (FC) layers 
  * 4096, 4096, and 1000 chanels respectively
* Max pooling layers
* Hidden layers have ReLu Retification
* Final layer is soft-max layer
* **Total 16 layers**

**VGG16 Disadvantages**

* Super slow - takes weeks to train
* **Large disk/bandwidth network achitecture with +533MB**
* Consider varient VGG19 classifer


**VGG16 Keras Implementation**

> keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

[Full Keras VGG Code:](https://keras.io/applications/#extract-features-with-vgg16)


---


## Neural Network Errors and Overfitting &#x1F499;

**Data Augmentation**

  * **Label-peserving transfomations**
    * Transformed images do not need to be stored on GPU disk to save space
    * **Image translation and horizontal reflections**
    
  * **RGB channel intensities**
    * Add transformation **(covariance matrix) to each RGB image pixel**
    * Object idenity invariant to changes in intensity/colour of images
  
**Dropout Rates**

  * ReLu neutrons 
  * Dropout is used for first two fully-connected (FC) layers (4096 and 4096)
  

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

![Image retrival](https://github.com/lucylow/salty-wet-man/blob/master/readme-images/waifu.png)


Default image used for **internal-activation** and **CAM visualization** is **"nsfw.jpg"**. Switch to another image by using the **"--image waifu-pic.jpeg"** 👀


```sh
yarn visualize --image waifu-pic.jpeg
```


---

## References &#x1F499;

* Karen Simonyan, Andrew Zisserman. Very Deep Convolutional Networks for Large-Scale Image Recognition . https://arxiv.org/abs/1409.1556
* Alex Krizhevsky. 2012. ImageNet Classification with Deep Convolutional Networks
* Yahoo Engineering's Caffe DL library and CaffeOnSpark model. https://yahooeng.tumblr.com/post/151148689421/open-sourcing-a-deep-learning-solution-for
* Gabriel Goh. Image Synthesis from Yahoo's open_nsfw. https://open_nsfw.gitlab.io/
* Client-Side NSFW Classification. https://nsfwjs.com/
* Margaret M. FleckDavid A. Forsyth Chris Bregler. Finding Naked People. 1996. http://luthuli.cs.uiuc.edu/~daf/papers/naked.pdf




---


**┬┴┬┴┤ʕ•ᴥ├┬┴┬┴**





