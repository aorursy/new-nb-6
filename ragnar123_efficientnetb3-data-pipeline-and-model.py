# import numpy as np

# import pandas as pd

# import cv2

# import tensorflow as tf

# import pathlib

# from tqdm import tqdm



# def _bytes_feature(value):

#     """Returns a bytes_list from a string / byte."""

#     if isinstance(value, type(tf.constant(0))):

#         value = value.numpy() # BytesList won't unpack a string from an EagerTensor.

#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



# def _float_feature(value):

#     """Returns a float_list from a float / double."""

#     return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))



# def _int64_feature(value):

#     """Returns an int64_list from a bool / enum / int / uint."""

#     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



# def serialize_example(feature0, feature1, feature2):

#     feature = {

#         'id': _bytes_feature(feature0),

#         'image': _bytes_feature(feature1),

#         'target': _int64_feature(feature2)

#     }

#     example_proto = tf.train.Example(features = tf.train.Features(feature = feature))

#     return example_proto.SerializeToString()

# TRAIN_IMAGE_DIR = '../input/landmark-recognition-2020/train'

# TRAIN = '../input/landmark-image-train/train_encoded.csv'



# # Read image and resize it

# def read_image(image_path, size = (384, 384)):

#     img = cv2.imread(image_path)

#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     img = cv2.resize(img, size)

#     return img





# def get_tf_records(record = 0, size = (384, 384)):

#     df = pd.read_csv(TRAIN)

#     # Get image paths

#     image_paths = [x for x in pathlib.Path(TRAIN_IMAGE_DIR).rglob('*.jpg')]

#     # Get only one group, this is a slow process so we need to make 50 different sessions

#     df = df[df['group'] == record]

#     # Reset index 

#     df.reset_index(drop = True, inplace = True)

#     # Get a list of ids

#     ids_list = list(df['id'].unique())

#     # Write tf records

#     with tf.io.TFRecordWriter('train_{}.tfrec'.format(record)) as writer:

#         for image_path in tqdm(image_paths):

#             image_id = image_path.name.split('.')[0]

#             if image_id in ids_list:

#                 # Get target

#                 target = df[df['id'] == image_id]['landmark_id_encode']

#                 img = read_image(str(image_path), size)

#                 img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 100))[1].tostring()

#                 example = serialize_example(

#                     str.encode(image_id), img, target.values[0]

#                 )

#                 writer.write(example)

                

# get_tf_records(record = 0, size = (384, 384))
# %tensorflow_version 2.x

# from google.colab import drive

# drive.mount('/content/drive')



# !pip install -q efficientnet

# import os

# import re

# import numpy as np

# import pandas as pd

# import random

# import math

# from sklearn import metrics

# from sklearn.model_selection import train_test_split

# import tensorflow as tf

# import efficientnet.tfkeras as efn

# from tensorflow.keras import backend as K

# import tensorflow_addons as tfa

# !pip install gcsfs

# from tqdm.notebook import tqdm as tqdm



# !pip install tensorflow~=2.2.0 tensorflow_gcs_config~=2.2.0

# import requests

# resp = requests.post("http://{}:8475/requestversion/{}".format(os.environ["COLAB_TPU_ADDR"].split(":")[0], tf.__version__))

# if resp.status_code != 200:

#   print("Failed to switch the TPU to TF {}".format(version))



# try:

#     # TPU detection. No parameters necessary if TPU_NAME environment variable is

#     # set: this is always the case on Kaggle.

#     tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

#     print('Running on TPU ', tpu.master())

# except ValueError:

#     tpu = None



# if tpu:

#     tf.config.experimental_connect_to_cluster(tpu)

#     tf.tpu.experimental.initialize_tpu_system(tpu)

#     strategy = tf.distribute.experimental.TPUStrategy(tpu)

# else:

#     # Default distribution strategy in Tensorflow. Works on CPU and single GPU.

#     strategy = tf.distribute.get_strategy()



# print("REPLICAS: ", strategy.num_replicas_in_sync)



# # For tf.dataset

# AUTO = tf.data.experimental.AUTOTUNE



# # Data access

# GCS_PATH = 'gs://kds-8e6633c4a6d544ae006948f95c01d818cf70ee95ed8ea3731ddbd5dc'

# GCS_PATH_2 = 'gs://kds-6c5f45cfe497efd7115b4ccc111abe0d435e12a356d98167abf66c21'

# DICT_PATH = 'gs://kds-80f8b28815daf39c39d710eca9c78b31e9f396674d64cad8af10e75e/train_encoded.csv'



# # Configuration

# EPOCHS = 20

# BATCH_SIZE = 32 * strategy.num_replicas_in_sync

# IMAGE_SIZE = [384, 384]

# # Seed

# SEED = 100

# # Learning rate

# LR = 0.0001

# # Number of classes

# NUMBER_OF_CLASSES = 81313



# # Training filenames directory

# FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train*.tfrec') + tf.io.gfile.glob(GCS_PATH_2 + '/train*.tfrec')

# # Read csv file

# df = pd.read_csv(DICT_PATH)

# # Using 20% of the data to validate

# TRAINING_FILENAMES, VALIDATION_FILENAMES = train_test_split(FILENAMES, test_size = 0.20, random_state = SEED)

# training_groups = [int(re.compile(r"_([0-9]*)\.").search(filename).group(1)) for filename in TRAINING_FILENAMES]

# validation_groups = [int(re.compile(r"_([0-9]*)\.").search(filename).group(1)) for filename in VALIDATION_FILENAMES]

# n_trn_classes = df[df['group'].isin(training_groups)]['landmark_id_encode'].nunique()

# n_val_classes = df[df['group'].isin(validation_groups)]['landmark_id_encode'].nunique()

# print(f'The number of unique training classes is {n_trn_classes} of {NUMBER_OF_CLASSES} total classes')

# print(f'The number of unique validation classes is {n_val_classes} of {NUMBER_OF_CLASSES} total classes')



# # Seed everything

# def seed_everything(seed):

#     random.seed(seed)

#     np.random.seed(seed)

#     os.environ['PYTHONHASHSEED'] = str(seed)

#     tf.random.set_seed(seed)



# # Function to decode our images (normalize and reshape)

# def decode_image(image_data):

#     image = tf.image.decode_jpeg(image_data, channels = 3)

#     # Convert image to floats in [0, 1] range

#     image = tf.cast(image, tf.float32) / 255.0

#     # Explicit size needed for TPU

#     image = tf.reshape(image, [*IMAGE_SIZE, 3])

#     return image



# # This function parse our images and also get the target variable

# def read_tfrecord(example):

#     TFREC_FORMAT = {

#         # tf.string means bytestring

#         "image": tf.io.FixedLenFeature([], tf.string), 

#         # shape [] means single element

#         "target": tf.io.FixedLenFeature([], tf.int64)

#         }

#     example = tf.io.parse_single_example(example, TFREC_FORMAT)

#     image = decode_image(example['image'])

#     target = tf.cast(example['target'], tf.int32)

#     return image, target



# # This function load our tf records and parse our data with the previous function

# def load_dataset(filenames, ordered = False):

#     # Read from TFRecords. For optimal performance, reading from multiple files at once and

#     # Diregarding data order. Order does not matter since we will be shuffling the data anyway

    

#     ignore_order = tf.data.Options()

#     if not ordered:

#         # Disable order, increase speed

#         ignore_order.experimental_deterministic = False 

        

#     # Automatically interleaves reads from multiple files

#     dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads = AUTO)

#     # Use data as soon as it streams in, rather than in its original order

#     dataset = dataset.with_options(ignore_order)

#     # Returns a dataset of (image, label) pairs

#     dataset = dataset.map(read_tfrecord, num_parallel_calls = AUTO) 

#     return dataset



# # This function output the data so that we can use arcface

# def arcface_format(image, target):

#     return {'inp1': image, 'inp2': target}, target



# # Training data pipeline

# def get_training_dataset(filenames, ordered = False):

#     dataset = load_dataset(filenames, ordered = ordered)

#     dataset = dataset.map(arcface_format, num_parallel_calls = AUTO)

#     # The training dataset must repeat for several epochs

#     dataset = dataset.repeat() 

#     dataset = dataset.shuffle(2048)

#     dataset = dataset.batch(BATCH_SIZE)

#     # Prefetch next batch while training (autotune prefetch buffer size)

#     dataset = dataset.prefetch(AUTO)

#     return dataset



# # Validation data pipeline

# def get_validation_dataset(filenames, ordered = True, prediction = False):

#     dataset = load_dataset(filenames, ordered = ordered)

#     dataset = dataset.map(arcface_format, num_parallel_calls = AUTO)

#     # If we are in prediction mode, use bigger batch size for faster prediction

#     if prediction:

#         dataset = dataset.batch(BATCH_SIZE * 4)

#     else:

#         dataset = dataset.batch(BATCH_SIZE)

#     # Prefetch next batch while training (autotune prefetch buffer size)

#     dataset = dataset.prefetch(AUTO) 

#     return dataset



# # Count the number of observations with the tabular csv

# def count_data_items(filenames):

#     records = [int(re.compile(r"_([0-9]*)\.").search(filename).group(1)) for filename in filenames]

#     df = pd.read_csv(DICT_PATH)

#     n = df[df['group'].isin(records)].shape[0]

#     return n



# NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)

# NUM_VALIDATION_IMAGES  = count_data_items(VALIDATION_FILENAMES)

# print(f'Training with {NUM_TRAINING_IMAGES} images')

# print(f'Validating with {NUM_VALIDATION_IMAGES} images')



# # Function for a custom learning rate scheduler with warmup and decay

# def get_lr_callback():

#     lr_start   = 0.000001

#     lr_max     = 0.0000005 * BATCH_SIZE

#     lr_min     = 0.000001

#     lr_ramp_ep = 5

#     lr_sus_ep  = 0

#     lr_decay   = 0.8

   

#     def lrfn(epoch):

#         if epoch < lr_ramp_ep:

#             lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start   

#         elif epoch < lr_ramp_ep + lr_sus_ep:

#             lr = lr_max    

#         else:

#             lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min    

#         return lr



#     lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = False)

#     return lr_callback



# # Function to calculate global average precision score

# def gap_vector(pred, conf, true, return_x = False):

#     '''

#     Compute Global Average Precision (aka micro AP), the metric for the

#     Google Landmark Recognition competition. 

#     This function takes predictions, labels and confidence scores as vectors.

#     In both predictions and ground-truth, use None/np.nan for "no label".



#     Args:

#         pred: vector of integer-coded predictions

#         conf: vector of probability or confidence scores for pred

#         true: vector of integer-coded labels for ground truth

#         return_x: also return the data frame used in the calculation



#     Returns:

#         GAP score

#     '''

#     x = pd.DataFrame({'pred': pred, 'conf': conf, 'true': true})

#     x.sort_values('conf', ascending = False, inplace = True, na_position = 'last')

#     x['correct'] = (x.true == x.pred).astype(int)

#     x['prec_k'] = x.correct.cumsum() / (np.arange(len(x)) + 1)

#     x['term'] = x.prec_k * x.correct

#     gap = x.term.sum() / x.true.count()

#     if return_x:

#         return gap, x

#     else:

#         return gap



# class ArcMarginProduct(tf.keras.layers.Layer):

#     '''

#     Implements large margin arc distance.



#     Reference:

#         https://arxiv.org/pdf/1801.07698.pdf

#         https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/

#             blob/master/src/modeling/metric_learning.py

#     '''

#     def __init__(self, n_classes, s=30, m=0.50, easy_margin=False,

#                  ls_eps=0.0, **kwargs):



#         super(ArcMarginProduct, self).__init__(**kwargs)



#         self.n_classes = n_classes

#         self.s = s

#         self.m = m

#         self.ls_eps = ls_eps

#         self.easy_margin = easy_margin

#         self.cos_m = tf.math.cos(m)

#         self.sin_m = tf.math.sin(m)

#         self.th = tf.math.cos(math.pi - m)

#         self.mm = tf.math.sin(math.pi - m) * m



#     def get_config(self):



#         config = super().get_config().copy()

#         config.update({

#             'n_classes': self.n_classes,

#             's': self.s,

#             'm': self.m,

#             'ls_eps': self.ls_eps,

#             'easy_margin': self.easy_margin,

#         })

#         return config



#     def build(self, input_shape):

#         super(ArcMarginProduct, self).build(input_shape[0])



#         self.W = self.add_weight(

#             name='W',

#             shape=(int(input_shape[0][-1]), self.n_classes),

#             initializer='glorot_uniform',

#             dtype='float32',

#             trainable=True,

#             regularizer=None)



#     def call(self, inputs):

#         X, y = inputs

#         y = tf.cast(y, dtype=tf.int32)

#         cosine = tf.matmul(

#             tf.math.l2_normalize(X, axis=1),

#             tf.math.l2_normalize(self.W, axis=0)

#         )

#         sine = tf.math.sqrt(1.0 - tf.math.pow(cosine, 2))

#         phi = cosine * self.cos_m - sine * self.sin_m

#         if self.easy_margin:

#             phi = tf.where(cosine > 0, phi, cosine)

#         else:

#             phi = tf.where(cosine > self.th, phi, cosine - self.mm)

#         one_hot = tf.cast(

#             tf.one_hot(y, depth=self.n_classes),

#             dtype=cosine.dtype

#         )

#         if self.ls_eps > 0:

#             one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.n_classes



#         output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

#         output *= self.s

#         return output





# # Function to build our model using fine tunning (efficientnet)

# def get_model():



#     with strategy.scope():



#         margin = ArcMarginProduct(

#             n_classes = NUMBER_OF_CLASSES, 

#             s = 64, 

#             m = 0.05, 

#             name='head/arc_margin', 

#             dtype='float32'

#             )



#         inp = tf.keras.layers.Input(shape = (*IMAGE_SIZE, 3), name = 'inp1')

#         label = tf.keras.layers.Input(shape = (), name = 'inp2')

#         x0 = efn.EfficientNetB0(weights = 'imagenet', include_top = False)(inp)

#         x = tf.keras.layers.GlobalAveragePooling2D()(x0)

#         x = tf.keras.layers.Dropout(0.3)(x)

#         x = tf.keras.layers.Dense(512)(x)

#         x = margin([x, label])

        

#         output = tf.keras.layers.Softmax(dtype='float32')(x)



#         model = tf.keras.models.Model(inputs = [inp, label], outputs = [output])



#         opt = tf.keras.optimizers.Adam(learning_rate = LR)



#         model.compile(

#             optimizer = opt,

#             loss = [tf.keras.losses.SparseCategoricalCrossentropy()],

#             metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

#             ) 

        

#         return model



# # Seed everything

# seed_everything(SEED)



# # Build training and validation generators

# train_dataset = get_training_dataset(TRAINING_FILENAMES, ordered = False)

# val_dataset = get_validation_dataset(VALIDATION_FILENAMES, ordered = True, prediction = False)

# STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE



# model = get_model()

# # Using a checkpoint to save best model (want the entire model, not only the weights)

# checkpoint = tf.keras.callbacks.ModelCheckpoint(f'/content/drive/My Drive/Models/baseline_model_effb0_arcface.h5', 

#                                                  monitor = 'val_loss', 

#                                                  save_best_only = True, 

#                                                  save_weights_only = False)

# # Using learning rate scheduler

# cb_lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', 

#                                                        mode = 'min', 

#                                                        factor = 0.5, 

#                                                        patience = 1, 

#                                                        verbose = 1, 

#                                                        min_delta = 0.0001)



# # Train and evaluate our model

# history = model.fit(train_dataset,  

#                     steps_per_epoch = STEPS_PER_EPOCH,

#                     epochs = EPOCHS,

#                     callbacks = [get_lr_callback(), checkpoint],

#                     validation_data = val_dataset,

#                     verbose = 1

#                     )



# # Restart tpu

# tf.tpu.experimental.initialize_tpu_system(tpu)

# # Load best model

# model = tf.keras.models.load_model('/content/drive/My Drive/Models/baseline_model_effb0_arcface.h5')



# # Reset val dataset, now in prediction mode

# val_dataset = get_validation_dataset(VALIDATION_FILENAMES, ordered = True, prediction = True)

# # Get ground truth target for the fold

# val_target = val_dataset.map(lambda image, target: target).unbatch()

# val_targets = list(next(iter(val_target.batch(NUM_VALIDATION_IMAGES))).numpy())



#  # Predictions

# val_image = val_dataset.map(lambda image, target: image['inp1'])

# # Transform validation dataset as a numpy iterator

# val_image = val_image.as_numpy_iterator()

# # Initiate empty list to store predictions and confidences

# target_predictions = []

# target_confidences = []

# # Iterate over validation images and predict in batches of 1024 images

# batches = math.ceil(NUM_VALIDATION_IMAGES / (BATCH_SIZE * 4))

# for image in tqdm(val_image, total = batches):

#     prediction = model.predict(image)

#     target_prediction = np.argmax(prediction, axis = -1)

#     target_confidence = np.max(prediction, axis = -1)

#     target_predictions.extend(list(target_prediction))

#     target_confidences.extend(list(target_confidence))



# # Calculate global average precision for the fold

# gap = gap_vector(target_predictions, target_confidences, val_targets)

# accuracy_score = metrics.accuracy_score(val_targets, target_predictions)

# print(f'Our global average precision score is {gap}')

# print(f'Our accuracy score is {accuracy_score}')
import operator

import gc

import pathlib

import shutil

import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras import backend as K

from scipy import spatial

import cv2



import efficientnet.tfkeras as efn

import math



NUMBER_OF_CLASSES = 81313

IMAGE_SIZE = [384, 384]

LR = 0.0001



class ArcMarginProduct(tf.keras.layers.Layer):

    '''

    Implements large margin arc distance.



    Reference:

        https://arxiv.org/pdf/1801.07698.pdf

        https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/

            blob/master/src/modeling/metric_learning.py

    '''

    def __init__(self, n_classes, s=30, m=0.50, easy_margin=False,

                 ls_eps=0.0, **kwargs):



        super(ArcMarginProduct, self).__init__(**kwargs)



        self.n_classes = n_classes

        self.s = s

        self.m = m

        self.ls_eps = ls_eps

        self.easy_margin = easy_margin

        self.cos_m = tf.math.cos(m)

        self.sin_m = tf.math.sin(m)

        self.th = tf.math.cos(math.pi - m)

        self.mm = tf.math.sin(math.pi - m) * m



    def get_config(self):



        config = super().get_config().copy()

        config.update({

            'n_classes': self.n_classes,

            's': self.s,

            'm': self.m,

            'ls_eps': self.ls_eps,

            'easy_margin': self.easy_margin,

        })

        return config



    def build(self, input_shape):

        super(ArcMarginProduct, self).build(input_shape[0])



        self.W = self.add_weight(

            name='W',

            shape=(int(input_shape[0][-1]), self.n_classes),

            initializer='glorot_uniform',

            dtype='float32',

            trainable=True,

            regularizer=None)



    def call(self, inputs):

        X, y = inputs

        y = tf.cast(y, dtype=tf.int32)

        cosine = tf.matmul(

            tf.math.l2_normalize(X, axis=1),

            tf.math.l2_normalize(self.W, axis=0)

        )

        sine = tf.math.sqrt(1.0 - tf.math.pow(cosine, 2))

        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:

            phi = tf.where(cosine > 0, phi, cosine)

        else:

            phi = tf.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = tf.cast(

            tf.one_hot(y, depth=self.n_classes),

            dtype=cosine.dtype

        )

        if self.ls_eps > 0:

            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.n_classes



        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        output *= self.s

        return output





# Function to build our model using fine tunning (efficientnet)

def get_model(eff = 1):

    



    margin = ArcMarginProduct(

        n_classes = NUMBER_OF_CLASSES, 

        s = 64, 

        m = 0.15, 

        name='head/arc_margin', 

        dtype='float32'

        )



    inp = tf.keras.layers.Input(shape = (*IMAGE_SIZE, 3), name = 'inp1')

    label = tf.keras.layers.Input(shape = (), name = 'inp2')

    if eff == 0:

        x = efn.EfficientNetB0(weights = None, include_top = False)(inp)

    elif eff == 1:

        x = efn.EfficientNetB1(weights = None, include_top = False)(inp)

    elif eff == 2:

        x = efn.EfficientNetB2(weights = None, include_top = False)(inp)

    elif eff == 3:

        x = efn.EfficientNetB3(weights = None, include_top = False)(inp)

    elif eff == 4:

        x = efn.EfficientNetB4(weights = None, include_top = False)(inp)

    elif eff == 5:

        x = efn.EfficientNetB5(weights = None, include_top = False)(inp)

    elif eff == 6:

        x = efn.EfficientNetB6(weights = None, include_top = False)(inp)

    elif eff == 7:

        x = efn.EfficientNetB7(weights = None, include_top = False)(inp)

        

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(512)(x)

    x = margin([x, label])



    output = tf.keras.layers.Softmax(dtype='float32')(x)



    model = tf.keras.models.Model(inputs = [inp, label], outputs = [output])



    opt = tf.keras.optimizers.Adam(learning_rate = LR)



    model.compile(

        optimizer = opt,

        loss = [tf.keras.losses.SparseCategoricalCrossentropy()],

        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

        ) 



    return model





NUM_EMBEDDING_DIMENSIONS = 512

DATASET_DIR = '../input/landmark-image-train/train_encoded.csv'

TEST_IMAGE_DIR = '../input/landmark-recognition-2020/test'

TRAIN_IMAGE_DIR = '../input/landmark-recognition-2020/train'

MODEL1 = get_model(eff = 3)

MODEL1.load_weights('../input/landmark-baseline-model/baseline_model_effb3_arcface_0_15_512.h5')

MODEL1 = tf.keras.models.Model(inputs = MODEL1.input[0], outputs = MODEL1.layers[-4].output)

MODEL2 = get_model(eff = 5)

MODEL2.load_weights('../input/landmark-baseline-model/baseline_model_effb5_arcface_0_15_512.h5')

MODEL2 = tf.keras.models.Model(inputs = MODEL2.input[0], outputs = MODEL2.layers[-4].output)

NUM_TO_RERANK = 1





NUM_PUBLIC_TEST_IMAGES = 10345 # Used to detect if in session or re-run.



# Read image and resize it

def read_image(image_path, size = (384, 384)):

    img = cv2.imread(image_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, size)

    img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 100))[1].tostring()

    img = tf.image.decode_jpeg(img, channels = 3)

    img = tf.image.resize(img, (512, 512))

    img = tf.cast(img, tf.float32) / 255.0

    img = tf.reshape(img, [1, 512, 512, 3])

    return img



# Function to get training and test embeddings

def generate_embeddings(filepaths):

    image_paths = [x for x in pathlib.Path(filepaths).rglob('*.jpg')]

    num_images = len(image_paths)

    ids = num_images * [None]

    # Generate an empty matrix where we can store the embeddings of each image

    embeddings = np.empty((num_images, NUM_EMBEDDING_DIMENSIONS))

    for i, image_path in enumerate(image_paths):

        ids[i] = image_path.name.split('.')[0]

        image_tensor = read_image(str(image_path), (384, 384))

        prediction1 = MODEL1.predict(image_tensor)

        prediction2 = MODEL2.predict(image_tensor)

        prediction = np.average([prediction1, prediction2], axis = 0)

        embeddings[i, :] = prediction

    return ids, embeddings



# This function get the most similar train images for each test image based on cosine similarity

def get_similarities(train_csv, test_directory, train_directory):

    # Get target dictionary

    df = pd.read_csv(train_csv)

    df = df[['id', 'landmark_id']]

    df.set_index('id', inplace = True)

    df = df.to_dict()['landmark_id']

    # Extract the test ids and global feature for the test images

    test_ids, test_embeddings = generate_embeddings(test_directory)

    # Extract the train ids and global features for the train images

    train_ids, train_embeddings = generate_embeddings(train_directory)

    # Initiate a list were we will store the similar training images for each test image (also score)

    train_ids_labels_and_scores = [None] * test_embeddings.shape[0]

    # Using (slow) for-loop, as distance matrix doesn't fit in memory

    for test_index in range(test_embeddings.shape[0]):

        distances = spatial.distance.cdist(

            test_embeddings[np.newaxis, test_index, : ], train_embeddings, 'cosine')[0]

        # Get the indices of the closest images

        top_k = np.argpartition(distances, NUM_TO_RERANK)[:NUM_TO_RERANK]

        # Get the nearest ids and distances using the previous indices

        nearest = sorted([(train_ids[p], distances[p]) for p in top_k], key = lambda x: x[1])

        # Get the labels and score results

        train_ids_labels_and_scores[test_index] = [(df[train_id], 1.0 - cosine_distance) for \

                                                   train_id, cosine_distance in nearest]

        

    del test_embeddings

    del train_embeddings

    gc.collect()

    return test_ids, train_ids_labels_and_scores



# This function aggregate top simlarities and make predictions

def generate_predictions(test_ids, train_ids_labels_and_scores):

    targets = []

    scores = []

    

    # Iterate through each test id

    for test_index, test_id in enumerate(test_ids):

        aggregate_scores = {}

        # Iterate through the similar images with their corresponing score for the given test image

        for target, score in train_ids_labels_and_scores[test_index]:

            if target not in aggregate_scores:

                aggregate_scores[target] = 0

            aggregate_scores[target] += score

        # Get the best score

        target, score = max(aggregate_scores.items(), key = operator.itemgetter(1))

        targets.append(target)

        scores.append(score)

        

    final = pd.DataFrame({'id': test_ids, 'target': targets, 'scores': scores})

    final['landmarks'] = final['target'].astype(str) + ' ' + final['scores'].astype(str)

    final[['id', 'landmarks']].to_csv('submission.csv', index = False)

    return final



def inference_and_save_submission_csv(train_csv, test_directory, train_directory):

    image_paths = [x for x in pathlib.Path(test_directory).rglob('*.jpg')]

    test_len = len(image_paths)

    if test_len == NUM_PUBLIC_TEST_IMAGES:

        # Dummy submission

        shutil.copyfile('../input/landmark-recognition-2020/sample_submission.csv', 'submission.csv')

        return 'Job Done'

    else:

        test_ids, train_ids_labels_and_scores = get_similarities(train_csv, test_directory, train_directory)

        final = generate_predictions(test_ids, train_ids_labels_and_scores)

        return final

    

final = inference_and_save_submission_csv(DATASET_DIR, TEST_IMAGE_DIR, TRAIN_IMAGE_DIR)