# TF SETUP

This repository is a base to build tensorflow applications

## SETUP

Note : this setup is for Windows only. Linux should come later.

First, you need to download OpenCV v3.4.10 and add it to the ``WORKSPACE`` file (``"windows_opencv"``).

Second, you need to clone the Tensorflow git repository to your local machine, and add it to the ``WORKSPACE`` file (``"org_tensorflow"``).

## EXAMPLE

**NOTE : this is still a work in progress, I was able to setup the GPU Delegate on my windows machine, will add the setup in the future.**

This repository comes with a example app to run classification models.

The default provided model is the [Tensorflow's default classification model](https://www.tensorflow.org/lite/models/image_classification/overview)

The requirements for the network are the following :
- Input must be an RGB image
- Output must be a vector containing K unit, K being the number of classes

There are 3 possibilities :
- C++ classification (CPU only)
- C++ classification (with GPU delegate)
- Python classification (CPU only)

To build the C++ classification (CPU only), run:
```
.\build_cpu.ps1
```

To build the C++ classification (CPU only), run:
```
.\build_gpu.ps1
```

In all cases, the programs can take the following CLI flags :
- ``--image_file`` : the path to the image to run the inference on 
- ``--model_file`` : the path to the tflite model
- ``--label_file`` : the labels for classification
- ``--number_of_inferences`` : the number of inferences to perform (for execution time considerations)

A small benchmark is given, that run the inference on any image :
```
.\benchmark.ps1 <path/to/image>
```

It runs 100 inferences per method (python / c++ cpu / c++ gpu) to display average computation time.

For reference, this was the output on my computer (Intel i3, Intel HD Graphics 520):
```
===================================================
TFLite C++ CPU inference ==========================
===================================================
Inference details : 
model_file : ./example_classification/mobilenet.tflite
label_file : ./example_classification/labels.txt
image_file : .\images\goldfish.png
number_of_inference : 100

Ran 100 inferences in 12.8764
Average time per inference : 128.764 ms
Average inference per seconds : 7.76616

Classification results : 
Label : web site - Score : 0.218719
Label : honeycomb - Score : 0.148013
Label : chainlink fence - Score : 0.100848
Label : envelope - Score : 0.0689244
Label : Windsor tie - Score : 0.0438098
===================================================
TFLite C++ GPU inference ==========================
===================================================
Inference details : 
model_file : ./example_classification/mobilenet.tflite
label_file : ./example_classification/labels.txt
image_file : .\images\goldfish.png
number_of_inference : 100

Ran 100 inferences in 1.02396
Average time per inference : 10.2396 ms
Average inference per seconds : 97.6602

Classification results : 
Label : web site - Score : 0.218718
Label : honeycomb - Score : 0.148012
Label : chainlink fence - Score : 0.100849
Label : envelope - Score : 0.0689248
Label : Windsor tie - Score : 0.0438099
===================================================
TFLite Python CPU inference =======================
===================================================
Inference details : 
model_file : ./example_classification/mobilenet.tflite
label_file : ./example_classification/labels.txt
image_file : .\images\goldfish.png
number_of_inference : 100

Ran 100 inferences in 8.457959651947021
Average time per inference : 84.57959651947021 ms
Average number of inference per seconds : 11.82318243584669

Classification results : 
Label : web site - Score : 0.21871912479400635
Label : honeycomb - Score : 0.14801281690597534
Label : chainlink fence - Score : 0.10084792226552963
Label : envelope - Score : 0.0689244419336319
Label : Windsor tie - Score : 0.04380982741713524
```

As expected, GPU inference is about 10x faster.
