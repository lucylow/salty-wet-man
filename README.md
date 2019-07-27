# salty-wet-man

![alt text](https://github.com/lucylow/salty-wet-man/blob/master/64330371_573206533208216_2036770996110753792_n.png)

### General
Not Suitable for Work (NSFW) image classification using Keras and Tensorflow machine learning tools. [**Requires heavy computation**]

### Classifier 
* Defining NSFW material is subjective and the task of identifying these images is non-trivial.
* Identification into two categories
  * [SFW] positively trained for neutral images that are safe for work.
  * [NSFW] negatively trained for pornographic images involving sexually explicit images, acts, or drawings.

## Technical

1. Install Python dependencies including required
   Python package (keras, tensorflow and tensorflowjs). To prevent 
   modifying your global Python environment, run this demo from
   a [virtualenv](https://virtualenv.pypa.io/en/latest/) or
   [pipenv](https://pipenv.readthedocs.io/en/latest/).
2. Download and convert the VGG16 model to TensorFlow.js format
3. Launch a Node.js script to load the converted model and compute
   the maximally-activating input images for the convnet's filters
   using gradient ascent in the input space and save them as image
   files under the `dist/filters` directory [**Requires heavy computation**]
4. Launch a Node.js script to calculate the internal convolutional
   layers' activations and th gradient-based class activation
   map (CAM) and save them as image files under the
   `dist/activation` directory. [**Requires heavy computation**]
5. Compile and launch the web view

## How to install + Prepare node environment

Run the command:
```sh
yarn visualize
```

```sh
yarn visualize --gpu
```

You may also increase the number of filters to visualize per convolutional
layer from the default 8 to a larger value, e.g., 32:

```sh
yarn visualize --gpu --filters 32
```
 [requires heavy computation]

The default image used for the internal-activation and CAM visualization is
"nsfw.jpg". You can switch to another image by using the "--image" waifu-pic, e.g.,

```sh
yarn visualize --image NSFW.jpg
```


### Tools
* TensorflowJS Image Model
* Keras Image Model






