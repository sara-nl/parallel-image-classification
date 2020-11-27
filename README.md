# parallel-image-classification

This is a more-or-less general purpose image classification framework, accelerated through the data parallel paradigm. This means that it can scale on multiple processors (CPUs or GPUs) and/or multiple compute nodes. It supports multiple CNN backbones (a list of supported backbones can be found [here](https://www.tensorflow.org/api_docs/python/tf/keras/applications)). It reads the images directly from a given path, and infers the class labels according to the directory names.

## Usage

### Prerequsites
The following packages are needed:
- Tensorflow
- Horovod
- OpenMPI or MPICH

### Directory structure 
Images should be arranged according to the following directory structure:
```main_directory/
...class_a/
......a_image_1.jpg
......a_image_2.jpg
...class_b/
......b_image_1.jpg
......b_image_2.jpg

```

### Execute the training
```
mpirun -np 4 classification.py --arch EfficientNetB7 --epochs 50
```
Please replace the `-np 4` argument with the actual number of processors (i.e., GPUs) available on the system. For example, if each node has 4 GPUs, and two nodes are used, then `-np 8` should be used. The code will automatically bind each MPI process to a corresponding GPU.  The `--arch` argument should be chosen from the list of Keras applications. The `--epochs` parameters can be chosen accordingly.  After the training, the model is saved in `model.h5` file.

`classification.py` will automatically balance the dataset (if it is imbalanced) by calculating the `class_weight`.

### Evalulation
After the training, some very simple verification (e.g., calculating a confusion matrix) can be done by executing the `plot_matrix.py` script.

