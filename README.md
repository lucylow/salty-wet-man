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

  * **Theoretically CNN is best** since large learning capcity and complexitty
  * Stationarity of statistics
  * Locality of pixel dependencies  


**CNN GPU Implementation**

  * **Heavy computation required** - Size of CNN network limited by GPU memory avaliabe
  * Highly optomized implementation of 2D convolutions
  * Solution to spread network over multiple GPUs via **parallel processing**
  
---

## Object Recognition &#x1F499;

**Labeled image-training datasets**
  
  * Small image datasets (order of tens of thousands of images) - classic MNIST digit-recognition task with best error rate  
  * Large image datasets (order of hundreds of thousands of images) - examples are ImageNet or LabelMe
 

**Large Dataset ImageNet**

  * **Dataset over 15 million labeled images**
  * Variable-resolution images (256x256)
  * Training, validation, and testing images
  * Annual competition - ImageNet Large-Scale Visual Recognition Challenge (ILSVRC) 
  

---

## NSFW Images  &#x1F499;

  * Static images
  * Uncontrolled backgrounds
  * Multiple people and partial figures
  * Different camera angles
  
  
---  

## NSFW Object Recognition: Content-Based Retrival &#x1F499;


**Image Location with Large Areas of Skin-colored Regions**

  * Skin regions in an image and color and texture properties 
  * **Input RGB values (skin)** with log-opponent representation
  * Intensity of image (texture) smoothered with median filter subtracted with original image
  * **QBIC search - operator uses absraction of an image to search for colored textured regions**

[Insert image of QBIC algorithm]

**Elongated Regions Grouping**

  * Grouped **2D and 3D constraints** on body/skin regions
  * Modelling humans == **cylindrical parts within the skeleton geometry**
  * Imaging model to identify region outlines

[Insert image of 2D/3D constraints]


**Classify Regions into Human Limbs**

  * **Geometric grouping algorithms** - matching a view to a collection of images of an object
  * Make a hypothesis object is present, and an estimate of appearance
  * **Future vector from compressed image**
  * Minimum distance classifer to match feature vectors


---

## Classifier - VGG16 model &#x1F499;

* VGG16 is a CNN for large-scale image recognition 
* **Model achieves 92.7% top-5 test accuracy on ImageNet**
* Implemented with Keras and Tensorflow backend in this project

[Insert Image of VGG16 architecture]

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


## Errors and Overfitting &#x1F499;

**Data Augmentation**

  * **Label-peserving transfomations**
    * Transformed images do not need to be stored on the GPU disk
    * **Image translation and horizontal reflections**
    
  * **RGB channel intensities**
    * Add a transformation **(covariance matrix) to each RGB image pixel**
    * Object idenity is invariant to changes in intensity/colour of images
  
**Dropout Rates**

  * ReLu neutrons 
  * Dropout is used for first two fully-connected layers
  

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

* Karen Simonyan, Andrew Zisserman. Very Deep Convolutional Networks for Large-Scale Image Recognition . https://arxiv.org/abs/1409.1556
* Alex Krizhevsky. 2012. ImageNet Classification with Deep Convolutional Networks
* Yahoo Engineering's Caffe DL library and CaffeOnSpark model. https://yahooeng.tumblr.com/post/151148689421/open-sourcing-a-deep-learning-solution-for
* Gabriel Goh. Image Synthesis from Yahoo's open_nsfw. https://open_nsfw.gitlab.io/
* Client-Side NSFW Classification. https://nsfwjs.com/
* Margaret M. FleckDavid A. Forsyth Chris Bregler. Finding Naked People. 1996. http://luthuli.cs.uiuc.edu/~daf/papers/naked.pdf




---


**┬┴┬┴┤ʕ•ᴥ├┬┴┬┴**





