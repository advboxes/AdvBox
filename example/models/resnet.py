from __future__ import division
from builtins import str
from builtins import range
from builtins import object
from past.utils import old_div
import paddle
import paddle.fluid as fluid
import math
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant

__all__ = ["ResNet", "ResNet50", "ResNet101", "ResNet152"]

train_parameters = {
    "input_size": [3, 224, 224],
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225],
    "learning_strategy": {
        "name": "piecewise_decay",
        "batch_size": 256,
        "epochs": [30, 60, 90],
        "steps": [0.1, 0.01, 0.001, 0.0001]
    }
}


def create_parameter(layers, shape, dtype):
    # use layerhelper to init bias, scale, mean, variance
    helper = LayerHelper("batch_norm", **locals())
    param_name = "batch_norm_" + str(layers)
    scale = helper.create_parameter(
        attr=fluid.ParamAttr(name=param_name + '.w' + '_0'),
        shape=[shape],
        dtype=dtype,
        default_initializer=Constant(1.0))
    scale.stop_gradient = True

    bias = helper.create_parameter(
        attr=fluid.ParamAttr(name=param_name + '.b' + '_0'),
        shape=[shape],
        dtype=dtype,
        is_bias=True)
    bias.stop_gradient = True

    mean = helper.create_parameter(
        attr=ParamAttr(
            name=param_name + '.w' + '_1',
            initializer=Constant(0.0),
            trainable=False),
        shape=[shape],
        dtype=dtype)
    mean.stop_gradient = True

    variance = helper.create_parameter(
        attr=ParamAttr(
            name=param_name + '.w' + '_2',
            initializer=Constant(1.0),
            trainable=False),
        shape=[shape],
        dtype=dtype)
    variance.stop_gradient = True

    return scale, bias, mean, variance

class ResNet(object):
    def __init__(self, layers=50):
        self.params = train_parameters
        self.layers = layers

    def conv_net(self, input, class_dim=1000):
        layers = self.layers
        self.cur_layers = 0
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_filters = [64, 128, 256, 512]

        conv = self.conv_bn_layer(
            input=input, num_filters=64, filter_size=7, stride=2, act='relu')

        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')

        for block in range(len(depth)):
            for i in range(depth[block]):
                conv = self.bottleneck_block(
                    input=conv,
                    num_filters=num_filters[block],
                    stride=2 if i == 0 and block != 0 else 1)

        pool = fluid.layers.pool2d(
            input=conv, pool_size=7, pool_type='avg', global_pooling=True)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        convfc = fluid.layers.fc(input=pool,
                              size=class_dim,
                              act=None,
                              param_attr=fluid.param_attr.ParamAttr(
                                  name='fc' + '_0.w_0',
                                  initializer=fluid.initializer.Uniform(-stdv,
                                                                        stdv)),
                              bias_attr=fluid.ParamAttr(name='fc' + '_0.b_0'))

        return convfc


    def net(self, input, class_dim=1000):
        convfc = self.conv_net(input, class_dim)
        logits = fluid.layers.softmax(input=convfc)

        return logits

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None):
        param_name = 'conv2d_' + str(self.cur_layers)
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=int(old_div((filter_size - 1), 2)),
            groups=groups,
            param_attr=fluid.ParamAttr(name=param_name + '.w' + '_0'),
            act=None,
            bias_attr=False
            )
        #print(conv)
        ###original BN op
        #conv_bn = fluid.layers.batch_norm(input=conv, act=act)

        ###replace BN op with batch_norm formula
        #
        # bn_tmp = (x - p_mean)/ sqrt(p_variance + const)
        # conv_x = bn_tmp * p_scale + p_bias
        #
        p_scale, p_bias, p_mean, p_variance = create_parameter(self.cur_layers, conv.shape[1], conv.dtype)
        bn_tmp1 = paddle.fluid.layers.elementwise_sub(conv, p_mean, axis=1)
        bn_tmp2 = paddle.fluid.layers.sqrt(p_variance)
        bn_tmp3 = paddle.fluid.layers.elementwise_div(bn_tmp1, bn_tmp2, axis=1)
        bn_tmp4 = paddle.fluid.layers.elementwise_mul(bn_tmp3, p_scale, axis=1)
        conv_x = paddle.fluid.layers.elementwise_add(bn_tmp4, p_bias, axis=1)

        if 'relu' == act:
            conv_bn = fluid.layers.relu(conv_x)
        else:
            conv_bn = conv_x

        self.cur_layers += 1

        return conv_bn

    def shortcut(self, input, ch_out, stride):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            return self.conv_bn_layer(input, ch_out, 1, stride)
        else:
            return input

    def bottleneck_block(self, input, num_filters, stride):
        conv0 = self.conv_bn_layer(
            input=input, num_filters=num_filters, filter_size=1, act='relu')
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu')
        conv2 = self.conv_bn_layer(
            input=conv1, num_filters=num_filters * 4, filter_size=1, act=None)

        short = self.shortcut(input, num_filters * 4, stride)

        return fluid.layers.elementwise_add(x=short, y=conv2, act='relu')


def ResNet50():
    model = ResNet(layers=50)
    return model


def ResNet101():
    model = ResNet(layers=101)
    return model


def ResNet152():
    model = ResNet(layers=152)
    return model
