# https://github.com/DLTK/DLTK/blob/master/examples/tutorials/01_reading_medical_images_in_tf.ipynb
import os
import tensorflow
import pandas
import numpy
import time
import matplotlib.pyplot as plt
import dltk
import dltk.io
import dltk.io.preprocessing
import dltk.io.augmentation
from dltk.io.augmentation import *
from dltk.io.preprocessing import *
import SimpleITK


class Timer(object):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('{} took {} seconds'.format(self.name, time.time() - self.t))


BATCH_SIZE = 5
ITERATIONS = 100
READER_PARAMS = {'n_examples': 1, 'example_size': [128, 224, 224], 'extract_examples': True}
READER_EXAMPLE_SHAPES = {'features': {'x': READER_PARAMS['example_size'] + [1, ]},
                         'labels': {'y': []}}
READER_EXAMPLE_DTYPES = {'features': {'x': tensorflow.float32},
                         'labels': {'y': tensorflow.int32}}


def load_data_as_dict(file_references, mode, params=None):
    data = {'features': [], 'labels': []}
    for meta_data in file_references:
        subject_id = meta_data[0]
        t1_fn = None

        stik_t1 = SimpleITK.ReadImage(t1_fn)
        t1 = SimpleITK.GetArrayFromImage(stik_t1)
        t1 = dltk.io.preprocessing.whitening(t1)
        t1 = t1[..., numpy.newaxis]
        y = None

        if params['extract_examples']:
            images = dltk.io.augmentation.extract_random_example_array(t1, example_size=params['example_size'], n_examples=params['n_examples'])
            for e in range(params['n_examples']):
                data['features'].append(images[e].astype(numpy.float32))
                data['labels'].append(y.astype(numpy.int32))
        else:
            data['features'].append(images)
            data['labels'].append(y.astype(numpy.int32))

    data['features'] = numpy.array(data['features'])
    data['labels'] = numpy.vstack(data['labels'])
    return data


data = load_data_as_dict()
x = tensorflow.placeholder(READER_EXAMPLE_DTYPES['features']['x'], [None, 128, 224, 224, 1])
y = tensorflow.placeholder(READER_EXAMPLE_DTYPES['labels']['y'], [None, 1])
dataset = tensorflow.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.repeat(None)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(1)
features = data['features']
labels = data['labels']
assert features.shape[0] == labels.shape[0], 'Error Shapes'

iterator = dataset.make_initializable_iterator()
nx = iterator.get_next()
with tensorflow.train.MonitoredTrainingSession() as sess_dict:
    sess_dict.run(iterator.initializer, feed_dict={x: features, y: labels})
    with Timer('Feed Dictionary'):
        for i in range(ITERATIONS):
            dict_batch_features, dict_batch_labels = sess_dict.run(nx)


def load_img_TFR(meta_data, params):
    x = []
    stik_t1 = SimpleITK.ReadImage(None)
    t1 = SimpleITK.GetArrayFromImage(stik_t1)
    t1 = dltk.io.preprocessing.whitening(t1)
    t1 = t1[..., numpy.newaxis]
    y = None
    if params['extract_features']:
        images = dltk.io.augmentation.extract_random_example_array(t1, example_size=params['example_size'], n_examples=params['n_examples'])
        for e in range(params['n_examples']):
            x.append(images[e].astype(numpy.float32))
    else:
        x = images
    return numpy.array(x), y


def _int64_feature(value):
    return tensorflow.train.Feature(int64_list=tensorflow.train.Int64List(value=[value]))


def _float_feature(value):
    return tensorflow.train.Feature(float_list=tensorflow.train.FloatList(value=value))

all_filenames = None
train_filename = 'train.tfrecords'
writer = tensorflow.python_io.TFRecordWriter(train_filename)
for meta_data in all_filenames:
    img, label = load_img_TFR(meta_data, READER_PARAMS)
    feature = {'train/label': _int64_feature(label),
               'train/feature': _float_feature(img.ravel())}
    example = tensorflow.train.Example(features=tensorflow.train.Features(feature=feature))
    writer.write(example.SerializeToString())
writer.close()


def decode(serialized_example):
    features = tensorflow.parse_single_example(serialized_example, features={'train/image': tensorflow.FixedLenFeature([128, 224, 224, 1], tensorflow.float32),
                                                                             'train/label': tensorflow.FixedLenFeature([], tensorflow.int64)})
    return features['train/image'], features['train/label']


dataset = tensorflow.data.TFRecordDataset(train_filename).map(decode)
dataset = dataset.repeat(None)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.repeat(1)

iterator = dataset.make_initializable_iterator()
features, labels = iterator.get_next()
nx = iterator.get_next()
with tensorflow.train.MonitoredTrainingSession() as sess_rec:
    sess_rec.run(iterator.initializer)
    with Timer('TFRecord'):
        for i in range(ITERATIONS):
            try:
                rec_batch_feat, rec_batch_lbl = sess_rec.run([features, labels])
            except tensorflow.errors.OutOfRangeError:
                pass


def read_fn(file_references, mode, params=None):
    for meta_data in file_references:
        sitk_t1 = SimpleITK.ReadImage(None)
        t1 = SimpleITK.GetArrayFromImage(sitk_t1)
        t1 = dltk.io.preprocessing.whitening(t1)
        t1 = t1[..., numpy.newaxis]
        if mode == tensorflow.estimator.ModeKeys.PREDICT:
            yield {'features': {'x': t1}}
        y = None
        if params['extracted_examples']:
            images = dltk.io.augmentation.extract_random_example_array(t1, example_size=params['example_size'], n_examples=params['n_examples'])
            for e in range(params['n_examples']):
                yield {'features': {'x': images[e].astype(numpy.float32)},
                       'labels': {'y': y.astype(numpy.int32)}}
        else:
            yield {'features': {'x': images},
                   'labels': {'y': y.asytpe(numpy.int32)}}
    return


def f():
    fn = read_fn()
    ex = next(fn)
    yield ex


dataset = tensorflow.data.Dataset.from_generator(f, READER_EXAMPLE_DTYPES, READER_EXAMPLE_SHAPES)
dataset = dataset.repeat(None)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(BATCH_SIZE)
iterator = dataset.make_initializable_iterator()
next_dict = iterator.get_next()

with tensorflow.train.MonitoredTrainingSession() as sess_gen:
    sess_gen.run(iterator.initializer)
    with Timer('Generator'):
        for i in range(ITERATIONS):
            gen_batch_img, gen_batch_lbl = sess_gen.run([next_dict['features'], next_dict['labels']])