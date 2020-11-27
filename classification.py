import tensorflow as tf
import argparse
import numpy as np
# from time_history import TimeHistory
import time
import glob
import os
from sklearn.utils import class_weight
from models import *
import horovod.tensorflow.keras as hvd


class TimeHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', dest='epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--arch', dest='dnn_arch', type=str, default='ResNet50', help='CNN architecture (e.g., resnet50, vgg16)')
parser.add_argument('-d', '--images-dir', dest='images_dir', type=str, required=True, help='The path of the image directory')
parser.add_argument('--batch-size', dest='batch_size', type=int, default=16, help='batch size')
args = parser.parse_args()


target_size = (512,512)

hvd.init()

# Configure GPUs (if any)
gpus = tf.config.experimental.list_physical_devices('GPU')

if hvd.local_rank() < len(gpus):
    gpu = gpus[hvd.local_rank()]

    tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_visible_devices(gpu, 'GPU')


# load training data
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        # zoom_range=0.1,
        horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
        args.images_dir,
        # target_size=target_size,
        batch_size=args.batch_size,
        class_mode='sparse')

# determine the number of classes

print(dir(train_generator))
num_classes = train_generator.num_classes

# Construct the DNN
# if x_train.shape[-1] == 1:
    # single channel immages. Repeating the BW channel needed
    # model.add(tf.keras.layers.Lambda(lambda x: tf.keras.backend.repeat_elements(x, 3, axis=-1), input_shape=(x_train.shape[1], x_train.shape[2], 1)))
dnn_arch = args.dnn_arch.strip() # convert model name to lower case, strip any white space

model = simple_keras_application_with_imagenet_weight(dnn_arch, (768,768,3), num_classes)


if hvd.rank() == 0:
    print(model.summary())


# Calculate class_weights because the dataset is imbalanced
data_classes = glob.glob(os.path.join(args.images_dir, 'level_*'))
data_n = dict()
for data_class in data_classes:
    data_n[data_class] = len(glob.glob(os.path.join(data_class, '*.png')))

print(data_n)
class_weight = class_weight.compute_class_weight('balanced',
                                                 classes=np.unique(train_generator.classes),
                                                 y=train_generator.classes)
class_weights = dict()
for i, c in enumerate(class_weight):
    class_weights[i] = c
print(class_weights)



# Compile the model
opt = tf.keras.optimizers.Adadelta(1.0 * hvd.size())
# opt = tf.keras.optimizers.Adam(0.5 * hvd.size())
opt = hvd.DistributedOptimizer(opt)
model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
              optimizer=opt,
              metrics=['sparse_categorical_accuracy'])

# Fit the model
cb_time_history = TimeHistory()
callbacks = [cb_time_history, hvd.callbacks.BroadcastGlobalVariablesCallback(0), hvd.callbacks.MetricAverageCallback()]
history = model.fit(train_generator,
                    steps_per_epoch=len(train_generator) // hvd.size(),
                    epochs=args.epochs,
                    callbacks=callbacks,
                    class_weight=class_weights,
                    verbose=1 if hvd.rank()==0 else 0)

if hvd.rank() == 0:
    print(history.history)
    print('Seconds per epoch: ', cb_time_history.times)
    model.save('model.h5')
    print('TensorFlow benchmark done.')
