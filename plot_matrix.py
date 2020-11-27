import tensorflow as tf
import argparse
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import itertools
from sklearn.utils import class_weight

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', dest='epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--arch', dest='dnn_arch', type=str, default='ResNet50', help='CNN architecture (e.g., resnet50, vgg16)')
parser.add_argument('-d', '--images-dir', dest='images_dir', type=str, required=True, help='The path of the image directory')
parser.add_argument('--batch-size', dest='batch_size', type=int, default=16, help='batch size')
args = parser.parse_args()

model = tf.keras.models.load_model('model.h5')
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        # zoom_range=.1,
        horizontal_flip=True)
val_generator = val_datagen.flow_from_directory(
        args.images_dir,
        # target_size=target_size,
        batch_size=args.batch_size,
        class_mode='sparse')

num_of_test_samples = 600
batch_size = 32

class_weight = class_weight.compute_class_weight('balanced',
                                                 classes=np.unique(val_generator.classes),
                                                 y=val_generator.classes)
class_weights = dict()
for i, c in enumerate(class_weight):
    class_weights[i] = c
print(class_weights)


# Y_pred = model.predict_generator(val_generator, num_of_test_samples // batch_size+1)
Y_pred = model.predict_generator(val_generator, verbose=1, class_weight=class_weight)
y_pred = np.argmax(Y_pred, axis=1)


# nr_batches = 10
# img_iter_val_0, img_iter_val_1 = itertools.tee(val_generator, 2)
# y_true = np.vstack(next(img_iter_val_0)[1] for _ in range(nr_batches)).astype('int')
# print(y_true.shape, y_pred.shape)
print(confusion_matrix(val_generator.classes, y_pred))
print(classification_report(val_generator.classes, y_pred, target_names=target_names))

