from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os


# Implemented 01/08/2020 by Ran Ben Izhak, based on
# https://github.com/Stick-To/ResNeXt-tensorflow

# Corrected and verified vs pretrained pytorch weights  (up to classification head - DID NOT TEST after layer4.2)

# Might have minor difference in edges of feature maps due to the difference in padding the conv ops

# ======================================================================================================= #
# === for end-to-end pytorch to tensorflow migration, both packages (and torchvision) are required ====== #
# ======================================================================================================= #

PYTORCH_TO_TF = [2, 3, 1, 0]   # Filters conversion pytorch.permute parameter


class ResNeXt50:
    def __init__(self, input_images, config, data_format='channels_last', features=True, pretrained=True,
                 weights_path=None):
        '''
        :param input_images: NHWC format
        :param config: architecture configuration file. example in __main__()
        :param data_format: tf data_format ('channels_last', 'channels_first')  DID NOT TEST CHANNEL_FIRST
        :param features: do we need the full model (False) or just up to layer 4.2 (True)
        :param pretrained: loading pretrained weights
        :param weights_path: path to model weights (Existing tf weights or save directory)
        '''
        if not weights_path:
            weights_path = 'weights'
        self.w_path = weights_path

        if pretrained:
            if not os.path.exists(os.path.join(self.w_path, 'checkpoint')):
                try:
                    import torchvision.models as models
                    import torch.nn as nn
                    self.torch_resnext = models.resnext50_32x4d(pretrained=True)
                except:
                    raise ModuleNotFoundError('Failed to load pytorch model'
                                              ' (pretrained==True, weights path does not exist)')
        self.config = config
        self.inputs = dict()
        self.features = features
        self.inputs['images'] = input_images

        self.input_shape = self.inputs['images'].shape
        self.num_classes = config.num_classes
        self.weight_decay = config.weight_decay

        assert data_format in ['channels_last', 'channels_first']
        self.data_format = data_format

        # Architecture params
        assert config.init_conv_filters % config.cardinality == 0
        self.block_list = config.residual_block_list
        self.filters_list = [config.init_conv_filters*(2**i) for i in range(1, len(config.residual_block_list)+1)]
        self.cardinality = config.cardinality   # cardinality = groups param in pytorch conv2d


        self.return_dict = {}
        self.global_step = tf.train.get_or_create_global_step()
        self.is_training = False

        self._define_inputs()
        self._build_graph()
        self._init_session()

        if pretrained:
            if not os.path.exists(os.path.join(self.w_path, 'checkpoint')):
                delattr(self, 'torch_resnext')
                self.save_weight(os.path.join(self.w_path, 'checkpoint'))
            else:
                self.load_weight(os.path.join(self.w_path, 'checkpoint'))

    def _define_inputs(self):
        shape = [None]
        shape.extend(self.input_shape)
        self.images = self.inputs['images']

        self.labels = tf.placeholder(dtype=tf.int32, shape=[None, self.num_classes], name='labels')
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr')

    def _build_graph(self):
        with tf.variable_scope('before_split'):
            if hasattr(self, 'torch_resnext'):
                conv_layer = self.torch_resnext.conv1
                bn_layer = self.torch_resnext.bn1
            else:
                conv_layer = bn_layer = None

            conv1_1 = self._conv2d(self.images,
                                   self.config.init_conv_filters,
                                   self.config.init_conv_kernel_size,
                                   self.config.init_conv_strides,
                                   init_pytorch=conv_layer)
            self.return_dict['conv1'] = conv1_1

            bn = self._bn(conv1_1, bn_layer)
            self.return_dict['bn1'] = bn

            activ_1 = tf.nn.relu(bn)
            self.return_dict['relu1'] = activ_1

            pool1 = self._max_pooling(
                bottom=activ_1,
                pool_size=self.config.init_pooling_pool_size,
                strides=self.config.init_pooling_strides,
                name='pool1'
            )
            self.return_dict['pool1'] = pool1
        with tf.variable_scope('split'):
            residual_block = pool1

            for layer_i in range(len(self.block_list)):
                for block_i in range(self.block_list[layer_i]):
                    torch_layers = self._get_block_layers(layer_i, block_i)
                    strides = 1
                    if block_i == 0:
                        downsample = True
                        if not layer_i == 0:
                            strides = 2
                    else:
                        downsample=False
                    residual_block = self._residual_bottleneck(residual_block,
                                                           self.filters_list[layer_i],
                                                           strides,
                                                           'layer{}.{}'.format(layer_i+1, block_i),
                                                           torch_layers,
                                                           downsample)
                    self.return_dict['layer{}.{}'.format(layer_i+1, block_i)] = residual_block

        axes = [1, 2] if self.data_format == 'channels_last' else [2, 3]
        global_pool = tf.reduce_mean(residual_block, axis=axes, keepdims=False, name='global_pool')
        self.gloabl_pool = global_pool

        if not self.features:
            with tf.variable_scope('final_dense'):
                final_dense = tf.layers.dense(global_pool, self.num_classes, name='final_dense')
            with tf.variable_scope('optimizer'):
                self.logit = tf.nn.softmax(final_dense, name='logit')
                self.classifer_loss = tf.losses.softmax_cross_entropy(self.labels, final_dense, label_smoothing=0.1, reduction=tf.losses.Reduction.MEAN)
                self.l2_loss = self.weight_decay * tf.add_n(
                    [tf.nn.l2_loss(var) for var in tf.trainable_variables()]
                )
                total_loss = self.classifer_loss + self.l2_loss
                lossavg = tf.train.ExponentialMovingAverage(0.9, name='loss_moveavg')
                lossavg_op = lossavg.apply([total_loss])
                with tf.control_dependencies([lossavg_op]):
                    self.total_loss = tf.identity(total_loss)
                var_list = tf.trainable_variables()
                varavg = tf.train.ExponentialMovingAverage(0.9, name='var_moveavg')
                varavg_op = varavg.apply(var_list)
                optimizer = tf.train.MomentumOptimizer(self.lr, momentum=0.9)
                train_op = optimizer.minimize(self.total_loss, global_step=self.global_step)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                self.train_op = tf.group([update_ops, lossavg_op, varavg_op, train_op])
                self.accuracy = tf.reduce_mean(
                    tf.cast(tf.equal(tf.argmax(final_dense, 1), tf.argmax(self.labels, 1)), tf.float32), name='accuracy'
                )

    def _get_block_layers(self, layer_i, block_i):
        '''
        Residual groups in plave layer, group of pytorch
        :param layer:
        :param block:
        :return:
        '''
        return_dict = {}
        keys = []
        keys += ['conv' + str(i) for i in range(1,4)]
        keys += ['bn' + str(i) for i in range(1, 4)]
        keys += ['downsample_conv', 'downsample_bn']
        for k in keys:
            return_dict.setdefault(k, None)
        if hasattr(self, 'torch_resnext'):
            layer = self.torch_resnext.__getattr__('layer' + str(layer_i+1))
            block = layer[block_i]
            return_dict = {'conv1': block.conv1, 'bn1': block.bn1,
                           'conv2': block.conv2, 'bn2': block.bn2,
                           'conv3': block.conv3, 'bn3': block.bn3,
                           }
            if block_i == 0:
                return_dict['downsample_conv'] = block.downsample[0]
                return_dict['downsample_bn'] = block.downsample[1]
        return return_dict

    def _init_session(self):
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()

    def train_one_batch(self, images, labels, lr, sess=None):
        self.is_training = True
        if sess is None:
            sess_ = self.sess
        else:
            sess_ = sess
        _, loss, acc = sess_.run([self.train_op, self.total_loss, self.accuracy],
                                 feed_dict={
                                     self.images: images,
                                     self.labels: labels,
                                     self.lr: lr
                                 })
        return loss, acc

    def validate_one_batch(self, images, labels, sess=None):
        self.is_training = False
        if sess is None:
            sess_ = self.sess
        else:
            sess_ = sess
        logit, acc = sess_.run([self.logit, self.accuracy], feed_dict={
                                     self.images: images,
                                     self.labels: labels,
                                     self.lr: 0.
                                 })
        return logit, acc

    def test_one_batch(self, images, sess=None):
        self.is_training = False
        if sess is None:
            sess_ = self.sess
        else:
            sess_ = sess
        # return_list = [x for x in self.return_dict.values()]
        tst_val = [self.return_dict['conv0'], self.return_dict['bn0'], self.return_dict['relu0']]   # self.logit
        bn_vars = [x for x in tf.trainable_variables() if 'before_split/batch_normalization/gamma' in x.name]
        return_val = sess_.run(tst_val + bn_vars, feed_dict={
                                     self.images: images,
                                     self.lr: 0.
                                 })
        return return_val

    def test_layers(self, input, sess=None):
        self.is_training = False
        if sess is None:
            sess_ = self.sess
        else:
            sess_ = sess
        ops = []
        names = []
        weight_ops = []
        weight_names = []
        for k,v in self.return_dict.items():
            names.append(k)
            ops.append(v)
        len_ops = len(ops)
        for wop in tf.global_variables():
            weight_names.append(wop.name)
            weight_ops.append(wop)

        return_val = sess_.run(ops + weight_ops, feed_dict={
                                     self.images: input,
                                     self.lr: 0.
                                 })
        val_dict = {}
        for k,v in zip(names, return_val[:len_ops]):
            val_dict[k] = v
        weight_dict = {}
        for k,v in zip(weight_names, return_val[len_ops:]):
            weight_dict[k] = v
        return val_dict, weight_dict

    def save_weight(self, path, mode='latest', sess=None):
        assert(mode in ['latest', 'best'])
        if sess is None:
            sess_ = self.sess
        else:
            sess_ = sess
        saver = self.saver if mode == 'latest' else self.best_saver
        saver.save(sess_, path, global_step=self.global_step)
        print('save', mode, 'model in', path, 'successfully')

    def load_weight(self, path,mode='latest', sess=None):
        assert(mode in ['latest', 'best'])
        if sess is None:
            sess_ = self.sess
        else:
            sess_ = sess
        saver = self.saver if mode == 'latest' else self.best_saver
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(path))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess_, ckpt.model_checkpoint_path)
            print('load', mode, 'model in', path, 'successfully')
        else:
            raise FileNotFoundError('Not Found Model File!')

    def _bn(self, bottom, init_pytorch=None):
        if init_pytorch:
            init_args = self._init_bn_pytorch(init_pytorch)
        else:
            init_args = {}
        bn = tf.layers.batch_normalization(
            inputs=bottom,
            axis=3 if self.data_format == 'channels_last' else 1,
            training=self.is_training,
            epsilon=1e-5,
            momentum=0.9,
            **init_args
        )
        return bn

    @staticmethod
    def _init_bn_pytorch(layer):
        params = {}
        for param in ['weight', 'bias', 'running_var', 'running_mean']:
            params[param] = layer.__getattr__(param).data.numpy().copy()

        return {'gamma_initializer': tf.constant_initializer(params['weight']),
                'beta_initializer': tf.constant_initializer(params['bias']),
                'moving_mean_initializer': tf.constant_initializer(params['running_mean']),
                'moving_variance_initializer': tf.constant_initializer(params['running_var'])}

    def _conv2d(self, bottom, filters, kernel_size, strides, groups=1, init_pytorch=None):
        total_conv = []
        filters_per_path = filters // groups
        axis = -1
        for i in range(groups):
            if groups > 1:
                split_bottom = tf.gather(bottom, tf.range(i * filters_per_path, (i + 1) * filters_per_path), axis=axis)
            else:
                split_bottom = bottom
            if init_pytorch:
                init_args = self._init_conv2d_pytorch(init_pytorch, i * filters_per_path, (i + 1) * filters_per_path)
            else:
                init_args = {}

            n_pad = kernel_size // 2
            conv = tf.layers.conv2d(
                inputs=tf.pad(split_bottom, [[0, 0], [n_pad, n_pad], [n_pad, n_pad], [0, 0]]),
                filters=filters_per_path,
                kernel_size=kernel_size,
                strides=strides,
                padding='valid',
                data_format=self.data_format,
                use_bias=False,
                **init_args
            )
            total_conv.append(conv)
        total_conv = tf.concat(total_conv, axis=axis)
        return total_conv

    def _init_conv2d_pytorch(self, layer, i_s=0, i_e=-1):
        conv_w = layer.weight.data.numpy()[i_s:i_e]
        conv_w = conv_w.transpose(PYTORCH_TO_TF)
        conv_init = tf.constant_initializer(conv_w)
        return {'kernel_initializer': conv_init}

    def _residual_bottleneck(self, bottom, filters, strides=1, scope=None, layers=None, downsample=False):  # downsample=False
        with tf.variable_scope(scope):
            with tf.variable_scope('conv_branch'):
                conv = self._conv2d(bottom, filters, 1, 1, init_pytorch=layers['conv1'])
                self.return_dict[scope + '.conv1'] = conv
                bn = self._bn(conv, init_pytorch=layers['bn1'])
                self.return_dict[scope + '.bn1'] = bn


                conv = self._conv2d(tf.nn.relu(bn), filters, 3, strides, groups=self.cardinality, init_pytorch=layers['conv2'])
                self.return_dict[scope + '.conv2'] = conv

                bn = self._bn(conv, init_pytorch=layers['bn2'])
                self.return_dict[scope + '.bn2'] = bn

                conv = self._conv2d(tf.nn.relu(bn), filters*2, 1, 1, init_pytorch=layers['conv3'])
                self.return_dict[scope + '.conv3'] = conv

                bn = self._bn(conv, init_pytorch=layers['bn3'])
                self.return_dict[scope + '.bn3'] = bn


            with tf.variable_scope('identity_branch'):
                if downsample:
                    bottom = self._conv2d(bottom, filters * 2, 1, strides, init_pytorch=layers['downsample_conv'])
                    self.return_dict[scope + '.downsample.0'] = bottom
                    bottom = self._bn(bottom, init_pytorch=layers['downsample_bn'])
                    self.return_dict[scope + '.downsample.1'] = bottom

            relu = tf.nn.relu(bn + bottom)
            self.return_dict[scope + '.relu'] = relu
            return relu

    def _max_pooling(self, bottom, pool_size, strides, name):
        n_pad = pool_size // 2
        return tf.layers.max_pooling2d(
            inputs=tf.pad(bottom, [[0, 0], [n_pad, n_pad], [n_pad, n_pad], [0, 0]]),
            pool_size=pool_size,
            strides=strides,
            padding='valid',
            data_format=self.data_format,
            name=name
        )

    def _avg_pooling(self, bottom, pool_size, strides, name):
        n_pad = pool_size // 2
        return tf.layers.average_pooling2d(
            inputs=tf.pad(bottom, [[0, 0], [n_pad, n_pad], [n_pad, n_pad], [0, 0]]),
            pool_size=pool_size,
            strides=strides,
            padding='valid',
            data_format=self.data_format,
            name=name
        )

    def _dropout(self, bottom, name):
        return tf.layers.dropout(
            inputs=bottom,
            rate=self.prob,
            training=self.is_training,
            name=name
        )


def test_diff_layer(tf_net, torch_net):
    np.random.seed(42)
    inputs = np.random.random((1, 224, 224, 3))
    # tensorflow - all return values at once
    tf_out, tf_weights = tf_net.test_layers(inputs)

    torch_net.eval()
    hooks = {}
    [hooks.update({name: Hook(layer.eval(), name)}) for name, layer in torch_net.named_modules() if name in tf_out.keys()]
    torch_input = torch.Tensor(inputs).permute((0, 3, 1, 2))
    with torch.no_grad():
        torch_net.eval()
        _ = torch_net(torch_input)

    for k,v in hooks.items():
        if k in tf_out.keys():
            print('\nLayer {} test'.format(k))
            _ = test_out_diff_single(v.output, tf_out[k])
            print('debug')
    return


def test_out_diff_single(torch_out, tf_out):
    norm_diff = np.linalg.norm(torch_out.permute((0, 2, 3, 1)).numpy() - tf_out)
    max_diff = np.max(np.abs(torch_out.permute((0, 2, 3, 1)).numpy() - tf_out))
    min_diff = np.min(np.abs(torch_out.permute((0, 2, 3, 1)).numpy() - tf_out))
    if norm_diff < 1e-4:
        print('SAME')
    else:
        print('Norm diff - {},  max diff - {}, min diff - {}'.format(norm_diff, max_diff, min_diff))
    return norm_diff, max_diff, min_diff


class Hook():
    def __init__(self, module, name, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
        self.name = name

    def hook_fn(self, module, input, output):
        self.input = input[0].detach().clone()
        self.output = output.detach().clone()

    def close(self):
        self.hook.remove()


if __name__ == '__main__':

    from utils.config import Config
    import torchvision.models as models
    import torch
    cfg = Config()
    flat_inputs = tf.placeholder(tf.float32, cfg.input_shape)
    net = ResNeXt50(flat_inputs, cfg)

    with torch.no_grad():
        torch_resnext = models.resnext50_32x4d(pretrained=True)

 # ===================== TESTING DIFF ============ #

    test_diff_layer(net, torch_resnext)
    print('Finished testing')
