# Salty Wet Man

Inspired by the #Metoo and #Timesup movements in the fight for gender equity in the workplace to create safe economic opportunities for women. Salty Wet Man is a Not Suitable for Work image classification trained on the ImageNet dataset. Defining NSFW material is subjective and the task of identifying these images is non-trivial. Salty-Wet-Man identifies images into two categories: [SFW] positively trained for neutral images that are safe for work [NSFW] negatively trained for pornographic images involving sexually explicit images.

CNN for large-scale image recognition model implementation achieves object recognition functionalities like Content-Based Retrival via Localization, Image Detection, and Segmentation, and Image Region Proposal Cropping.

*Warning. Repo contains abstract nudity and may be unsuitable for the workplace.*

<div>
  
  [![Status](https://img.shields.io/badge/status-work--in--progress-success.svg)]()
  [![GitHub Issues](https://img.shields.io/github/issues/lucylow/salty-wet-man.svg)](https://github.com/lucylow/salty-wet-man/issues)
  [![GitHub Pull Requests](https://img.shields.io/github/issues-pr/lucylow/salty-wet-man.svg)](https://github.com/lucylow/salty-wet-man/pulls)
  [![License](https://img.shields.io/bower/l/bootstrap)]()

</div>

![](https://github.com/lucylow/salty-wet-man/blob/master/readme-images/creepyasfuck.png)  
*The creepy ass emails I get from coworkers with no context.*


-------------

## Table_of_Contents 

* [Half the Sky Motivation](#Half_the_sky_Motivation)
* [Technical Solution](#Technical_solution)
* [Convolutional Neural Networks](#Convolutional_Neural_Networks)
* [Object Recognition](#Object_Recognition_) 
* [NSFW Object Recognition: Content-Based Retrival via Localization](#NSFW_Object_Recognition_Content-Based_Retrival_via_Localization_) 
* [NSFW Object Recognition: Image Cropping](#NSFW_Object_Recognition_Image_Cropping_) 
* [Neural Network: Classifier VGG16 Model](#Neural_Network_Classifier_VGG16_Model_) 
* [Neural Network: Errors and Overfitting](#Neural_Network_Errors_and_Overfitting_) 
* [Technical Installations](#Technical_Installations_) 
* [Technical Visualizations](#Technical_Visualizations_) 
* [References](#References_) 

---
## Half the Sky Motivation 

* Ending gender-based violence
  * Unwanted sexual behaviour while in public, unwanted sexual behaviour online, unwanted sexual behaviour in the workplace, sexual assault, and physical assault:
  * One in three (32%) women experienced unwanted sexual behaviour in public
  * One in five (18%) women experienced online harassment
  * Three in ten (29%) women were targeted by inappropriate sexual behaviour in a work-related environment
  * Four in ten (39%) of women have been physically or sexually assaulted since the age of 15

* #Metoo
  * In 2006, Tarana Burke founded the Me Too movement and began using the phrase "Me Too" 
  * Social movement against sexual abuse and sexual harassment where people publicize allegations of sex crimes committed by powerful prominent men.
  * Helps to show survivors of sexual abuse that they are not alone. 
  * Improve awareness about sexual violence, showing widespread sexual harassment and assault 
  
* #Timesup
  * Time's Up is a social movement against sexual harassment founded on January 1, 2018 by Hollywood celebrities in response to the Weinstein effect and #MeToo
  * Focused on workplace equity and creating equal economic opportunities for women and people of color
  * **“Time’s Up was founded on the premise that everyone, every human being, deserves a right to earn a living, to take care of themselves, to take care of their families, free of the impediments of harassment and sexual assault and discrimination.”**
  * As of December 2018, it has raised more than $22 million for its legal defense fund, and gathered nearly 800 volunteer lawyers.  

---
## Technical Solution

* Defining NSFW material is subjective and the task of identifying these images is **non-trivial**
* **Salty-Wet-Man identifies images solving a binary classification problem:**

  * [SFW] positively trained for **neutral images** that are safe for work
  
  * [NSFW] negatively trained for **pornographic images** involving sexually explicit images
  

---


## Convolutional_Neural_Networks 
  
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

## Object_Recognition 

**Deep Learning's Impact on Computer Vision**

   ![deep learning impact](https://github.com/lucylow/salty-wet-man/blob/master/readme-images/object%20detection%20-%20impact%20of%20deep%20learnning.png)

**Labeled Image-Training Datasets**
  
  * Small image datasets (order of tens of thousands of images) -  MNIST digit-recognition with best error rate  
  * Large image datasets (order of hundreds of thousands of images) - ImageNet
 

**[ImageNet used for Large Scale Object Recognition](www.image-net.org)**

  * **Dataset over 15 million labeled images**
  * Variable-resolution images (256x256)
  * Training, validation, and testing images
  * **Benchmark - ImageNet Large-Scale Visual Recognition Challenge (ILSVRC)**



---  

## NSFW_Object_Recognition:_Content-Based_Retrival_via_Localization 


**Image Location with Large Areas of Skin-colored Regions**

  * Skin region properties - image, color, and texture  
  
  * **Input RGB values (skin spatial pixels)** with log-opponent representation
    * L(x) = 105*logbaseten(x+1+n)
    * I = L(G)
    * Rg = L(R) - L(G)
    * By = L(B) - (L(G) + L(R))/2
    
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
  
  
    ![RNN1](https://github.com/lucylow/salty-wet-man/blob/master/readme-images/RCNN1.png)


---

## NSFW_Object_Recognition:_Detection,_and_Segmentation 

* Object Image Segmentation
  * **Group together skin pixels**
  * Normalized cut 
  
* Input image each pixel with a category label
  * For every pixel - Check if the pixel [skin or not-skin]

* If atleast 30% of the image area skin, the image will be identified as passing the skin filter 

* Training data for this super expensive - need to find images with every pixel labeled


    ![RNN2](https://github.com/lucylow/salty-wet-man/blob/master/readme-images/RCNN2.png)

---

## NSFW_Object_Recognition_Image_Cropping 
* **How would salty-wet-man choose the image crops?**

* Brute force image cropping - sliding window approach (Bad)

* **Region proposals**
  * Looks for edges, and draw boxes around them 
  
* **Region detection without proposals**
  * **YOLO** - [You Only Look Once Algorithm. Unified, Real-Time Object Detection](https://pjreddie.com/media/files/papers/yolo.pdf)
  * **SSD** - [Single Shot Detector](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/single-shot-detectors.html)


---

## Neural_Network_Classifier_VGG16_Model

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
* **Total 16 Layers**

**VGG16 Disadvantages**

* Super slow - takes weeks to train
* **Large disk/bandwidth network achitecture with +533MB**
* Consider varient VGG19 classifer


**VGG16 Keras Implementation**

> keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

  [Full Keras VGG Code](https://keras.io/applications/#extract-features-with-vgg16)


---


## Neural_Network_Errors_and_Overfitting 

**Data Augmentation**

  * **Label peserving transfomations**
    * Transformed images do not need to be stored on GPU disk to save space
    * **Image translation and horizontal reflections**
    * [Image captioning using PyTorch](https://github.com/ruotianluo/ImageCaptioning.pytorch)
    
  * **RGB channel intensities**
    * Add transformation **(covariance matrix) to each RGB image pixel**
    * Object idenity invariant to changes in intensity/colour of images
  
**Dropout Rates**

  * ReLu neutrons 
  * Dropout is used for **first two fully-connected (FC) layers** (4096 and 4096)
  

---

## Technical_Installations 
Requires heavy computation

1. Install Python dependencies and packages (**Keras, TensorFlow, and TensorFlow.js**) - best to run from [virtualenv](https://virtualenv.pypa.io/en/latest/)
   
2. Download and convert the VGG16 model to TensorFlow.js format

3. Launch Node.js script to **load converted model** and compute **maximally-activating input images** for  convnet's filters using gradient ascent in the input space. Save image files under `dist/filters` directory 
   
4. Launch Node.js script to **calculate internal convolutional layers' activations** and gradient-based **Class Activation Map (CAM)**. Save image files under `dist/activation` directory
   
5. Compile. Launch web view at **https://lucylow.github.io/salty-wet-man/**


---


## Technical_Visualizations 


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

## References 

* Lean In. Women in the Workplace 2020 https://wiw-report.s3.amazonaws.com/Women_in_the_Workplace_2020.pdf
* Women and the workplace – How employers can advance equality and diversity – Report from the Symposium on Women and the Workplace
 https://www.canada.ca/en/employment-social-development/corporate/reports/women-symposium.html
* Karen Simonyan, Andrew Zisserman. Very Deep Convolutional Networks for Large-Scale Image Recognition . https://arxiv.org/abs/1409.1556
* Alex Krizhevsky. 2012. ImageNet Classification with Deep Convolutional Networks
* Yahoo Engineering's Caffe DL library and CaffeOnSpark model. https://yahooeng.tumblr.com/post/151148689421/open-sourcing-a-deep-learning-solution-for
* CS231n Computer Vision at Stanford University School of Engineering. Fei Fei Lee. https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv
* Gabriel Goh. Image Synthesis from Yahoo's open_nsfw. https://open_nsfw.gitlab.io/
* Client-Side NSFW Classification. https://nsfwjs.com/
* Ring-Filter image processing algorithm for Order Statistics. http://mfleck.cs.illinois.edu/order-statistics.html
* Mask R-CNN framework for object instance segmentation. https://arxiv.org/abs/1703.06870
* Margaret M. FleckDavid A. Forsyth Chris Bregler. Finding Naked People. 1996. http://luthuli.cs.uiuc.edu/~daf/papers/naked.pdf
* PyTorch tutorial review. https://github.com/ritchieng/the-incredible-pytorch and https://pytorch.org/tutorials/
* ImageNet training in PyTorch. https://github.com/pytorch/examples/tree/master/imagenet
* PyTorch Image Models. https://github.com/rwightman/pytorch-image-models
* Image Captioning PyTorch. https://github.com/ruotianluo/ImageCaptioning.pytorch
* PyTorch Visualizatoins. Implementation of convolutional neural network.
https://github.com/utkuozbulak/pytorch-cnn-visualizations
* Facebook AI Research. "Detectron2". Object detection and segmentation using PyTorch.
https://github.com/facebookresearch/detectron2
* Facebook AI Research.  "Faster R-CNN and Mask R-CNN in PyTorch 1.0". Creating detection and segmentation models using PyTorch .https://github.com/facebookresearch/maskrcnn-benchmark/
* https://en.wikipedia.org/wiki/Gamergate_controversy
* https://en.wikipedia.org/wiki/Cyberstalking





