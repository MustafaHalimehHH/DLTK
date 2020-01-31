# https://github.com/DLTK/DLTK/blob/master/examples/tutorials/03_building_a_model_fn.ipynb
import dltk
import dltk.networks.segmentation.fcn
import tensorflow
import numpy
import matplotlib.pyplot as plt


class NotebookLoggingHook(tensorflow.train.SessionRunHook):
    def __init__(self, fetches):
        self.fetches = fetches
        self.losses = []

    def before_run(self, run_context):
        return tensorflow.train.SessionRunArgs(self.fetches)

    def after_run(self,
                run_context,  # pylint: disable=unused-argument
                run_values):
        fetch_dict = run_values.results
        self.losses.append(fetch_dict['loss'])

        f, axarr = plt.subplots(2, 2, figsize=(16, 8))
        axarr[0, 0].imshow(numpy.squeeze(fetch_dict['x'][0, 0, :, :, 0]), cmap='gray')
        axarr[0, 0].set_title('Input: x')
        axarr[0, 0].axis('off')
        axarr[0, 1].plot(self.losses)
        axarr[0, 1].set_title('Crossentropy Loss')
        axarr[0, 1].set_yscale('log')
        axarr[0, 1].axis('on')
        axarr[1, 0].imshow(numpy.squeeze(fetch_dict['y_'][0, 0, :, :]), cmap='gray', vmin=0, vmax=1)
        axarr[1, 0].set_title('Prediction: y_')
        axarr[1, 0].axis('off')
        axarr[1, 1].imshow(numpy.squeeze(fetch_dict['y'][0, 0, :, :]), cmap='gray', vmin=0, vmax=1)
        axarr[1, 1].set_title('Truth: y')
        axarr[1, 1].axis('off')


nl_hook = NotebookLoggingHook(None)
hooks = [nl_hook]

def model_fn(features, labels, mode, params):
    net_output_ops = dltk.networks.segmentation.fcn.residual_fcn_3d(features['x'],
                                                                    num_classes=2,
                                                                    num_res_units=1,
                                                                    filters=(16, 32, 64),
                                                                    strides=((1, 1, 1),(1, 2, 2), (1, 2, 2)),
                                                                    mode=mode)
    if mode == tensorflow.estimator.ModeKeys.PREDICT:
        return tensorflow.estimator.EstimatorSpec(mode=mode,
                                                  predictions=net_output_ops,
                                                  export_outputs={'out', tensorflow.estimator.export.PredictOutput(net_output_ops)})
    loss = tensorflow.losses.sparse_softmax_cross_entropy(labels['y'], net_output_ops['logits'])
    global_step = tensorflow.train.get_global_step()
    optimizer = tensorflow.train.AdamOptimizer(learning_rate=params['learning_rate'], epsilon=1e-5)
    update_ops = tensorflow.get_collection(tensorflow.GraphKeys.UPDATE_OPS)
    with tensorflow.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step=global_step)

    return tensorflow.estimator.EstimatorSpec(mode=mode, predictions=net_output_ops, loss=loss, train_op=train_op)