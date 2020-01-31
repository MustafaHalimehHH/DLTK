# https://github.com/DLTK/DLTK/blob/master/examples/tutorials/02_reading_data_with_dltk_reader.ipynb
import os
import SimpleITK
import dltk.io.preprocessing
import dltk.io.augmentation
import dltk.io.abstract_reader
import tensorflow
import numpy


DATA_PATH = ['D:\Halimeh\Datasets\MSD\Task05_Prostate\imagesTr\prostate_02.nii.gz']


def read_fn(file_references, mode, params=None):
    for meta_data in file_references:
        t1_fn = SimpleITK.ReadImage(meta_data)
        print('t1_fn', type(t1_fn), dir(t1_fn))
        print('t1_f', t1_fn.GetHeight(), t1_fn.GetWidth())
        t1 = SimpleITK.GetImageFromArray(t1_fn)
        t1 = dltk.io.preprocessing.whitening(t1)
        t1 = t1[..., numpy.newaxis]

        if mode == tensorflow.estimator.ModeKeys.PREDICT:
            yield {'features': {'x': t1},
                   'metadata': {'subject_id': 'prostate_02',
                                'stik': t1_fn}}

        sex = None
        y = sex

        if mode == tensorflow.estimator.ModeKeys.TRAIN:
            # for Data Augmentation
            pass
        if params['extract_examples']:
            images = dltk.io.augmentation.extract_random_example_array(t1, example_size=params['example_size'], n_examples=params['n_examples'])
            for e in range(params['n_examples']):
                yield {'features': {'x': images[e].astype(numpy.float32)},
                       'labels': {'y': y.astype(numpy.int32)},
                       'metadata': {'subject_id': None,
                                    'sitk': t1_fn}}
        else:
            yield {'features': {'x': images},
                   'labels': {'y': y.astype(numpy.int32)},
                   'metadata': {'subject_id': None,
                                'sitk': t1_fn}}
    return



reader_params = {'n_examples': 1,
                 'example_size': [128, 224, 224],
                 'extract_examples': True}

it = read_fn(DATA_PATH, None)
ex_dict = next(it)
print('ex_dict', type(ex_dict))


reader_example_shapes = {'features': {'x': reader_params['example_size'] + [1, ]},
                         'labels': {'y': []}}
reader_example_dtypes = {'features': {'x': numpy.float32},
                         'labels': {'y': numpy.int32}}

reader = dltk.io.abstract_reader.Reader(read_fn=read_fn, dtypes=reader_example_dtypes)

input_fn, qinit_hook = reader.get_inputs(file_references=DATA_PATH,
                                         mode=tensorflow.estimator.ModeKeys.TRAIN,
                                         example_shapes=reader_example_shapes,
                                         batch_size=4,
                                         shuffle_cache_size=10,
                                         params=reader_params)
features, labels = input_fn()

s = tensorflow.train.MonitoredTrainingSession(hooks=[qinit_hook])
batch_features, batch_labels = s.run([features, labels])

import matplotlib.pyplot as plt
plt.imshow(batch_features['x'][0, 0, :, :, 0], cmap='gray')
plt.show()

