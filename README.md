# salty-wet-man &#x1F499;

Not Suitable for Work (NSFW) image classification using Keras and Tensorflow.js.

![alt text](https://github.com/lucylow/salty-wet-man/blob/master/64330371_573206533208216_2036770996110753792_n.png)

---


### Classifier &#x1F499;

* Defining NSFW material is subjective and the task of identifying these images is **non-trivial**.
* Identification into two categories:

  * [SFW] positively trained for **neutral images** that are safe for work.
  
  * [NSFW] negatively trained for **pornographic images** involving sexually explicit images. 


---


## Technical - requires heavy computation** &#x1F499;

1. Install Python dependencies and packages (**Keras, TensorFlow, and TensorFlow.js**). Best to run from [virtualenv](https://virtualenv.pypa.io/en/latest/)
   
2. Download and convert the VGG16 model to TensorFlow.js format

3. Launch Node.js script to **load converted model** and compute **maximally-activating input images** for  convnet's filters using gradient ascent in the input space. Save image files under `dist/filters` directory 
   
4. Launch Node.js script to **calculate internal convolutional layers' activations** and gradient-based **Class Activation Map (CAM)**. Save image files under `dist/activation` directory. 
   
5. Compile. Launch web view... **https://lucylow.github.io/salty-wet-man/**


---


## Technical Installations &#x1F499;

Run command:

```sh
yarn visualize
```


Increase the **number of filters to visualize** per convolutional layer from default 8 to a larger value (ex. 18):


```sh
yarn visualize --gpu --filters 18
```


**Default image used for internal-activation and CAM visualization is "nsfw.jpg"**. Switch to another image by using the **"--image waifu-pic"** 👀👀.


```sh
yarn visualize --image waifu-pic.jpg
```


---





**┬┴┬┴┤ʕ•ᴥ├┬┴┬┴**





