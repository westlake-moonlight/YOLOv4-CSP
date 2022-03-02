"""Build a YOLO-V4-CSP module in TensorFlow 2.8.
基于 2021 年 2 月发布的第二版 Scaled-YOLOv4 论文： https://arxiv.org/abs/2011.08036
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

# 设置如下全局变量，用大写字母。

CLASSES = 80  # 如果使用 COCO 数据集，则需要探测 80 个类别。

# 为了获得更快的速度，使用小的特征图。原始模型的 p5 特征图为 19x19，模型输入图片为 608x608。
# 只需要设置 P5 特征图大小即可，高度和宽度应该一样，建议设置大小在 [10, 19] 之间。
FEATURE_MAP_P5 = np.array((19, 19))  # 19, 19
FEATURE_MAP_P4 = FEATURE_MAP_P5 * 2  # 38, 38
FEATURE_MAP_P3 = FEATURE_MAP_P4 * 2  # 76, 76

# 格式为 height, width。当两者大小不同时尤其要注意。是 FEATURE_MAP_P3 的 8 倍。
MODEL_IMAGE_SIZE = FEATURE_MAP_P3 * 8  # 608, 608

# 如果使用不同大小的 FEATURE_MAP，应该相应调整预设框的大小。
resize_scale = 19 / FEATURE_MAP_P5[0]

# 根据 YOLO V3 论文的 2.3 节，设置 ANCHOR_BOXES 。除以比例后取整数部分。
ANCHOR_BOXES_P5 = [(116 // resize_scale, 90 // resize_scale),
                   (156 // resize_scale, 198 // resize_scale),
                   (373 // resize_scale, 326 // resize_scale)]
ANCHOR_BOXES_P4 = [(30 // resize_scale, 61 // resize_scale),
                   (62 // resize_scale, 45 // resize_scale),
                   (59 // resize_scale, 119 // resize_scale)]
ANCHOR_BOXES_P3 = [(10 // resize_scale, 13 // resize_scale),
                   (16 // resize_scale, 30 // resize_scale),
                   (33 // resize_scale, 23 // resize_scale)]

EPSILON = 1e-10


class MishActivation(keras.layers.Layer):
    """mish 激活函数。为了便于迁移到其它平台 serialization ，使用子类方法 subclassing，
    不使用层 layers.Lambda。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mish = tfa.activations.mish

    def call(self, inputs):
        """Conv, BN, _activation 模块，在 backbone 和 PANet 中将被多次调用。

        Arguments：
            inputs: 一个张量。数据类型为 float32 或 float16 。
        Returns:
            x: 一个张量，经过 mish 激活函数的处理，形状和输入张量相同。
        """

        x = self._mish(inputs)

        return x

    def get_config(self):
        config = super().get_config()  # 继承 self.name
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# 根据需要对模型的权重 weights 使用 Constraint。注意该 class 在 TF 2.4 可以运行，
# 但在 TF 2.8 中，对同一个模型，容易产生 NaN 权重，原因不明。
class LimitWeights(tf.keras.constraints.Constraint):

    def __call__(self, w):
        clipped_weight = tf.clip_by_value(
            w, clip_value_min=-0.001,
            clip_value_max=0.001)

        return clipped_weight


weight_limits = LimitWeights()


def conv_bn_mish(inputs, filters, kernel_size, strides=1, use_bias=False,
                 padding='same', separableconv=False, rate_regularizer=0.001,
                 training=None, convolution_only=False, conv_name=None):
    """Conv, BN, mish 激活函数模块，在 backbone 和 PANet 中将被多次调用。YOLO-v4-CSP 
    中不再使用 leaky_relu 激活函数。

    Arguments：
        inputs: 一个张量。数据类型为 float32。
        filters： 一个整数，是卷积模块的过滤器数量。
        kernel_size： 一个整数，是卷积核的大小。
        strides： 一个整数，是卷积的步进值。
        use_bias： 一个布尔值，只有在为 True 时卷积层才使用 bias 权重。
        padding： 一个字符串，是卷积的 padding 方式。
        separableconv： 一个布尔值，设置是否使用 Separableconv2D。
        rate_regularizer： 一个浮点数，是卷积的 L2 正则化率。
        training: 一个布尔值，表示处于训练模式或是推理 inference 模式。
        convolution_only： 一个布尔值，为 True 时表示该模块只使用卷积，为 False 则表示
            该模块包括卷积，BN，mish 激活函数。
        conv_name： 一个字符串，设置该 conv_bn_mish 模块的名字。
    Returns:
        x: 一个张量，经过卷积，Batch Normalization 和 mish 激活函数的处理，形状和输入张
            量相同。
    """

    regularizer_l2 = keras.regularizers.L2(rate_regularizer)

    if separableconv:
        # 不要遗漏对 bias 的限制。因为在模型的最终输出部分，只有一个卷积层，并且该层会使用
        # bias，如果忘记了设置，将可能出现 bias 的权重极大，达到 inf 的状态，然后导致
        # 出现 NaN。
        x = keras.layers.SeparableConv2D(
            filters=filters, kernel_size=kernel_size,
            strides=strides, padding=padding, use_bias=use_bias,
            depthwise_initializer=keras.initializers.HeNormal(),
            pointwise_initializer=keras.initializers.HeNormal(),
            depthwise_regularizer=regularizer_l2,
            pointwise_regularizer=regularizer_l2,
            bias_regularizer=regularizer_l2,
            # depthwise_constraint=weight_limits,
            # pointwise_constraint=weight_limits,
            # bias_constraint=weight_limits,
            name=conv_name)(inputs)

    else:
        x = keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size,
            strides=strides, padding=padding, use_bias=use_bias,
            kernel_initializer=keras.initializers.HeNormal(),
            kernel_regularizer=regularizer_l2,
            bias_regularizer=regularizer_l2,
            # kernel_constraint=weight_limits,  # conv_weight_limits
            # bias_constraint=weight_limits,  # vector_weight_limits
            name=conv_name)(inputs)

    # convolution_only 为 True，则不需要进行 BatchNormalization 和 MishActivation。
    if not convolution_only:

        # 注意 BatchNormalization 也有 γ 和 β 两个参数 weight，在对参数进行
        # regularization 和 constraint 时，也要加上这两个参数。否则它们可以变得很大，
        # 可能导致出现 NaN。
        x = keras.layers.BatchNormalization(
            beta_regularizer=regularizer_l2,
            gamma_regularizer=regularizer_l2,

            # beta_constraint=weight_limits,  # vector_weight_limits
            # gamma_constraint=weight_limits,  # vector_weight_limits
        )(x, training=training)

        x = MishActivation()(x)

    return x


def darknet53_residual(inputs, filters, training=None):
    """CSPDarknet53 的第一个 block，使用普通的 darknet53_residual，即没有经过 CSP
    分支操作。

    Arguments:
        inputs: 一个 3D 张量。形状为 (height, width, 3)。数据类型
        为 float32 (混合精度模式下为 float16 类型）。使用时 Keras 将会自动
        插入一个第 0 维度，作为批量维度。
        filters: 一个整数，表示当前 _csp_block 输出张量的过滤器数量。
        training: 一个布尔值，表示处于训练模式或是推理 inference 模式。
    Returns:
        x: 一个 3D 张量，形状为 (height / 2, width / 2, filters)。数据类型
            为 float32 (混合精度模式下为 float16 类型）。
    """

    x = conv_bn_mish(inputs=inputs, filters=filters, kernel_size=3,
                     strides=2, training=training)

    residual = x

    x = conv_bn_mish(inputs=x, filters=(filters // 2),
                     kernel_size=1, training=training)

    x = conv_bn_mish(inputs=x, filters=filters,
                     kernel_size=3, training=training)

    x = keras.layers.Add()([x, residual])

    return x


def csp_block(inputs, filters, residual_quantity, training=None):
    """CSPDarknet53 的基本组成单元，用于在保证模型准确度的前提下，降低计算量。

    大体实现方式是，把输入张量的特征图缩小到一半，然后分成两个分支，每个分支只使用一半数量的
    过滤器，并且只在主支进行卷积和 residual block 等计算，最后再把两个分支拼接起来。

    Arguments:
        inputs: 一个 3D 张量。形状为 (height, width, 3)。数据类型
        为 float32 (混合精度模式下为 float16 类型）。使用时 Keras 将会自动
        插入一个第 0 维度，作为批量维度。
        filters: 一个整数，表示当前 _csp_block 输出张量的过滤器数量。
        residual_quantity: 一个整数，表示在主支中 residual block 的数量。
        training: 一个布尔值，表示处于训练模式或是推理 inference 模式。
    Returns:
        x: 一个 3D 张量，形状为 (height / 2, width / 2, filters)。数据类型
            为 float32 (混合精度模式下为 float16 类型）。
    """

    # 每个 csp block 的第一个卷积，使用 strides=2 进行下采样。注意第一个卷积的卷积核大小
    # 应该是 3！ 论文中配图 figure 4 有误，配图 figure 4 左下角是 CSP block，它第一个
    # 卷积核大小写的是 1，其实应该是 3.
    x = conv_bn_mish(inputs=inputs, filters=filters, kernel_size=3,
                     strides=2, training=training)

    # split_branch 作为 CSP 的一个分支，最后将和主支进行 concatenation。
    split_branch = conv_bn_mish(inputs=x, filters=(filters // 2), 
                                kernel_size=1, training=training)

    x = conv_bn_mish(inputs=x, filters=(filters // 2), 
                     kernel_size=1, training=training)
    
    for _ in range(residual_quantity):
        residual = x
        x = conv_bn_mish(inputs=x, filters=(filters // 2), 
                         kernel_size=1, training=training)
        x = conv_bn_mish(inputs=x, filters=(filters // 2), 
                         kernel_size=3, training=training)
        x = keras.layers.Add()([x, residual])

    x = conv_bn_mish(inputs=x, filters=(filters // 2), 
                     kernel_size=1, training=training)

    x = keras.layers.Concatenate()([x, split_branch])

    x = conv_bn_mish(inputs=x, filters=filters, 
                     kernel_size=1, training=training)

    return x


def csp_darknet53(inputs, training=None):
    """CSPDarknet53 的主体架构。

    Arguments:
        inputs: 一个 3D 张量，表示一批图片。形状为 (608, 608, 3)。数据类型
            为 float32 (混合精度模式下为 float16 类型）。使用时 Keras 将会自动插入
            一个第 0 维度，作为批量维度。
        training: 一个布尔值，表示处于训练模式或是推理 inference 模式。
    Returns:
        p5_backbone: 一个 3D 张量，特征图最小。如果输入图片大小为 (608, 608, 3)，
            则该特征图大小为 (19, 19, 1024)。
        p4_backbone: 一个 3D 张量，特征图中等。如果输入图片大小为 (608, 608, 3)，
            则该特征图大小为 (38, 38, 512)。
        p3_backbone: 一个 3D 张量，特征图最大。如果输入图片大小为 (608, 608, 3)，
            则该特征图大小为 (76, 76, 256)。
    """

    p5_backbone, p4_backbone, p3_backbone = None, None, None

    # 图片输入模型之后，遇到的第一个卷积模块，使用普通卷积，不使用 separableconv。
    x = conv_bn_mish(inputs=inputs, filters=32, kernel_size=3,
                     separableconv=False, training=training)

    # 第一个 block 是普通的 darknet53_residual，residual 数量为 1.
    x = darknet53_residual(inputs=x, filters=64, training=training)

    # 后面 4 个 block 才是 csp residual
    parameters_csp_block_2 = 128, 2
    parameters_csp_block_3 = 256, 8
    parameters_csp_block_4 = 512, 8
    parameters_csp_block_5 = 1024, 4

    parameters_csp_blocks = [parameters_csp_block_2, parameters_csp_block_3,
                             parameters_csp_block_4, parameters_csp_block_5]

    for parameters_csp_block in parameters_csp_blocks:

        filters = parameters_csp_block[0]
        residual_number = parameters_csp_block[1]

        x = csp_block(inputs=x, filters=filters,
                      residual_quantity=residual_number, training=training)
        # backbone 应该输出 3 个分支， p5, p4, p3。
        if x.shape[-1] == 1024:
            p5_backbone = x
        elif x.shape[-1] == 512:
            p4_backbone = x
        elif x.shape[-1] == 256:
            p3_backbone = x

    return p5_backbone, p4_backbone, p3_backbone


def spp(inputs):
    """SPP 模块.

    将输入分成 4 个分支，其中 3 个分支分别使用 5, 9, 13 的 pool_size 进行池化，然后
    和输入再拼接起来，进行返回。该模块输入和输出的形状完全相同。
    """

    # 注意必须设置 strides=1，否则默认 strides=pool_size，会进行下采样。
    maxpooling_5 = keras.layers.MaxPooling2D(
        pool_size=5, strides=1, padding='same')(inputs)

    maxpooling_9 = keras.layers.MaxPooling2D(
        pool_size=9, strides=1, padding='same')(inputs)

    maxpooling_13 = keras.layers.MaxPooling2D(
        pool_size=13, strides=1, padding='same')(inputs)

    x = keras.layers.Concatenate(axis=-1)(
        [inputs, maxpooling_5, maxpooling_9, maxpooling_13])

    return x


def reversed_csp(inputs, target_filters, reversed_csp_quantity,
                 reversed_spp_csp=False, training=None):
    """Reversed CSP 模块。在 PANet 的上采样和下采样分支中要被多次用到。

    Arguments:
        inputs: 一个 3D 张量。形状为 (height, width, filters)。数据类型
            为 float32 (混合精度模式下为 float16 类型）。使用时 Keras 将会自动
            插入一个第 0 维度，作为批量维度。
        target_filters: 一个整数，表示该模块最终输出的过滤器数量。
        reversed_csp_quantity: 一个整数，表示在主支中 reversed_csp 的数量。
        reversed_spp_csp: 一个布尔值。该值为 True 时，插入最大池化模块 SPP 。此时
            这个 Reversed CSP 模块就变成了 Reversed-CSP-SPP 模块。
            只有 backbone 输出的最小特征层 p5 会用到这个 Reversed-CSP-SPP 模块。
            (如果是 YOLO-v4 large，最小特征层则可能是 p6, p7)
        training: 一个布尔值，表示处于训练模式或是推理 inference 模式。

    Returns:
        x: 一个 3D 张量，形状为 (height, width, filters / 2)。数据类型
            为 float32 （混合精度模式下为 float16 类型）。
    """

    # 如果是 reversed_spp_csp，第一个卷积块的特征通道数和输入是一样的，等于
    # target_filters 的 2 倍。实际上无须操作，等于直接使用 CSP-BLOCK 的输出。
    if reversed_spp_csp:
        x = inputs

    # 而如果是 reversed_csp，第一个卷积块就会把特征通道数量降一半。
    else:
        x = conv_bn_mish(inputs=inputs, filters=target_filters,
                         kernel_size=1, training=training)

    # split_branch 作为 CSP 的一个分支，最后将和主支进行 concatenation。
    split_branch = conv_bn_mish(inputs=x, filters=target_filters,
                                kernel_size=1, training=training)

    for i in range(reversed_csp_quantity):

        # 主支的第一个卷积，使用 1x1 卷积把特征通道数量降为一半。
        x = conv_bn_mish(inputs=x, filters=target_filters,
                         kernel_size=1, training=training)

        # reversed CSP 的 reversed，应该是指没有 residual 模块，只是单纯的循环。
        x = conv_bn_mish(inputs=x, filters=target_filters,
                         kernel_size=3, training=training)

        # 最小特征层 p5，要使用 SPP 模块，得到 Reversed-CSP-SPP 模块。并且 spp 应该只
        # 在第 0 个循环时使用一次。
        if reversed_spp_csp and (i == 0):

            # 对于 reversed_spp_csp，在 spp 之后增加了一个 1x1 的卷积
            x = conv_bn_mish(inputs=x, filters=target_filters,
                             kernel_size=1, training=training)

            x = spp(x)

    x = keras.layers.Concatenate(axis=-1)([x, split_branch])

    x = conv_bn_mish(inputs=x, filters=target_filters,
                     kernel_size=1, training=training)

    return x


def upsampling_branch(upsampling_input, lateral_input,
                      target_filters, reversed_csp_quantity, training=None):
    """PANet 的上采样分支。对 2个输入进行拼接，然后执行 reversed CSP 操作。

    Arguments:
        upsampling_input: 一个 3D 张量。形状为(height / 2, width / 2, filters)。
            数据类型为 float32 (混合精度模式下为 float16 类型）。使用时 Keras 将
            会自动插入一个第 0 维度，作为批量维度。
        lateral_input: 一个 3D 张量。形状为 (height, width, filters)。
            数据类型为 float32 (混合精度模式下为 float16 类型）。使用时 Keras 将
            会自动插入一个第 0 维度，作为批量维度。
        target_filters: 一个整数，表示在该模块最终输出的过滤器数量。
        reversed_csp_quantity: 一个整数，表示在主支中 residual block 的数量。
        training: 一个布尔值，表示处于训练模式或是推理 inference 模式。

    Returns:
        upsampling_output: 一个 3D 张量。形状为 (height, width, filters / 2)。
            数据类型为 float32 （混合精度模式下为 float16 类型）。
    """

    # 先用 1x1 卷积，将 2 个输入调整特征通道数量，然后再拼接。
    lateral_input = conv_bn_mish(inputs=lateral_input, filters=target_filters,
                                 kernel_size=1, training=training)

    upsampling_input = conv_bn_mish(
        inputs=upsampling_input, filters=target_filters,
        kernel_size=1, training=training)

    upsampling_input = keras.layers.UpSampling2D(size=2)(upsampling_input)

    concatenated = keras.layers.Concatenate(axis=-1)(
        [lateral_input, upsampling_input])

    # 进行 Reversed CSP 操作。
    upsampling_output = reversed_csp(
        inputs=concatenated, target_filters=target_filters,
        reversed_csp_quantity=reversed_csp_quantity, training=training)

    return upsampling_output


def downsampling_branch(downsampling_input, lateral_input,
                        target_filters, reversed_csp_quantity, training=None):
    """PANet 的下下采样分支。对 2 个输入进行拼接，然后执行 reversed CSP 操作。

    Arguments:
        downsampling_input: 一个 3D 张量。形状为(height * 2, width * 2,
            filters / 2)。
            数据类型为 float32 (混合精度模式下为 float16 类型）。使用时 Keras 将
            会自动插入一个第 0 维度，作为批量维度。
        lateral_input: 一个 3D 张量。形状为 (height, width, filters)。
            数据类型为 float32 (混合精度模式下为 float16 类型）。使用时 Keras 将
            会自动插入一个第 0 维度，作为批量维度。
        target_filters: 一个整数，表示在该模块最终输出的过滤器数量。
        reversed_csp_quantity: 一个整数，表示在主支中 residual block 的数量。
        training: 一个布尔值，表示处于训练模式或是推理 inference 模式。

    Returns:
        downsampling_output: 一个 3D 张量。形状为 (height, width, filters)。
            数据类型为 float32 （混合精度模式下为 float16 类型）。
    """

    # 先用 3x3 卷积，将 downsampling_input 调整特征通道数量，然后再拼接。
    downsampling_input = conv_bn_mish(
        inputs=downsampling_input, filters=target_filters,
        kernel_size=3, strides=2, training=training)

    concatenated = keras.layers.Concatenate(axis=-1)(
        [lateral_input, downsampling_input])

    # 拼接之后，特征通道数量增大了一倍，用 reversed_csp 可以将其减半，
    # 达到 target_filters。
    downsampling_output = reversed_csp(
        inputs=concatenated, target_filters=target_filters,
        reversed_csp_quantity=reversed_csp_quantity, training=training)

    return downsampling_output


def panet(inputs, training=None):
    """对 backbone 输入的 3 个张量，先使用上采样分支对其进行处理，然后用下采样分支
    进行处理，最后返回 3 个张量。

    Arguments:
        inputs: 一个元祖，包含来自 backbone 输出的 3个 3D 张量。分别表示为
            p5_backbone, p4_backbone, p3_backbone。使用时 Keras 将
            会自动插入一个第 0 维度，作为批量维度。
        training: 一个布尔值，表示处于训练模式或是推理 inference 模式。
    Returns:
        p5_neck: 一个 3D 张量，特征图最小。形状为 (19, 19, 512)。（3 个输出
            p5_neck，p4_neck， p3_neck，均假定输入图片大小为 (608, 608, 3)）
            使用时 Keras 将会自动插入一个第 0 维度，作为批量维度。
        p4_neck: 一个 3D 张量，特征图中等。形状为 (38, 38, 256)。使用时 Keras 将
            会自动插入一个第 0 维度，作为批量维度。
        p3_neck: 一个 3D 张量，特征图最大。形状为 (76, 76, 128)。使用时 Keras 将
            会自动插入一个第 0 维度，作为批量维度。
    """

    p5_backbone, p4_backbone, p3_backbone = inputs

    # 将 PANet 看做 2 个分支，upsampling_branch 和 downsampling_branch。
    # 先处理上采样分支，必须按 p5,p4,p3 的顺序，p5_upsampling 表示在上采样分支的 p5。
    p5_upsampling = reversed_csp(
        inputs=p5_backbone, target_filters=512,
        reversed_csp_quantity=2, reversed_spp_csp=True, training=training)

    p4_upsampling = upsampling_branch(
        upsampling_input=p5_upsampling, lateral_input=p4_backbone,
        target_filters=256, reversed_csp_quantity=2, training=training)

    p3_upsampling = upsampling_branch(
        upsampling_input=p4_upsampling, lateral_input=p3_backbone,
        target_filters=128, reversed_csp_quantity=2, training=training)

    # 再处理下采样分支，必须按 p3,p4,p5 的顺序，p5_neck 表示下采样分支的 p5，同时也是
    # PANet 的输出，所以叫 p5_neck。
    p3_neck = p3_upsampling
    p4_neck = downsampling_branch(
        downsampling_input=p3_neck, lateral_input=p4_upsampling,
        target_filters=256, reversed_csp_quantity=2, training=training)

    p5_neck = downsampling_branch(
        downsampling_input=p4_neck, lateral_input=p5_upsampling,
        target_filters=512, reversed_csp_quantity=2, training=training)

    return p5_neck, p4_neck, p3_neck


def heads(inputs, training=None):
    """将来自 neck 部分的 3 个输入转换为 3个 heads。

    Arguments:
        inputs: 一个元祖，包含来自 backbone 输出的 3 个 3D 张量。分别表示为
            p5_neck, p4_neck, p3_neck。使用时 Keras 将会自动插入一个第 0 维度，
            作为批量维度。
        training: 一个布尔值，表示处于训练模式或是推理 inference 模式。
    Returns:
        p5_head: 一个 3D 张量，特征图最小。形状为 (19, 19, 255)。（3 个输出
            p5_head, p4_head, p3_head，均假定输入图片大小为 (608, 608, 3)）
            使用时 Keras 将会自动插入一个第 0 维度，作为批量维度。
        p4_head: 一个 3D 张量，特征图中等。形状为 (38, 38, 255)。使用时 Keras 将
            会自动插入一个第 0 维度，作为批量维度。
        p3_head: 一个 3D 张量，特征图最大。形状为 (76, 76, 255)。使用时 Keras 将
            会自动插入一个第 0 维度，作为批量维度。
    """

    p5_neck, p4_neck, p3_neck = inputs

    p5_head = conv_bn_mish(inputs=p5_neck, filters=1024,
                           kernel_size=3, training=training)

    # 在模型的最终输出部分，只有一个卷积层，并且该层会使用 bias。
    p5_head = conv_bn_mish(inputs=p5_head, filters=255, separableconv=False,
                           kernel_size=1, convolution_only=True, use_bias=True,
                           conv_name='p5')

    p4_head = conv_bn_mish(inputs=p4_neck, filters=512,
                           kernel_size=3, training=training)
    p4_head = conv_bn_mish(inputs=p4_head, filters=255, separableconv=False,
                           kernel_size=1, convolution_only=True, use_bias=True,
                           conv_name='p4')

    p3_head = conv_bn_mish(inputs=p3_neck, filters=256,
                           kernel_size=3, training=training)
    p3_head = conv_bn_mish(inputs=p3_head, filters=255, separableconv=False,
                           kernel_size=1, convolution_only=True, use_bias=True,
                           conv_name='p3')

    return p5_head, p4_head, p3_head


def yolo_v4_csp(inputs, training=None):
    """YOLO-v4-CSP module。

    Arguments：
        inputs：一个 4D 图片张量，形状为 (batch_size, 608, 608, 3)，数据类型为
            tf.float32。可以用全局变量 MODEL_IMAGE_SIZE 设置不同大小的图片输入。
        training: 一个布尔值，用于设置模型是处在训练模式或是推理 inference 模式。
            在预测时，如果不使用 predict 方法，而是直接调用模型的个体，则必须设置参
            数 training=False，比如 model(x, training=False)。因为这样才能让模
            型的 dropout 层和 BatchNormalization 层以 inference 模式运行。而如
            果是使用 predict 方法，则不需要设置该 training 参数。
    Returns:
        head_outputs: 一个元祖，包含 3 个 tf.float32 类型的张量，张量形状为
            (batch_size, 19, 19, 255), (batch_size, 38, 38, 255),
            (batch_size, 76, 76, 255)。最后 1 个维度大小为 255，可以转换为 (3, 85)，
            表示有 3 个预测框，每个预测结果是一个长度为 85 的向量。
            在这个长度为 85 的向量中，第 0 位是置信度，第 1 位到第 81 位，代表 80 个
            类别的 one-hot 编码，最后 4 位，则是预测框的位置和坐标，格式为
            (x, y, height, width)，其中 x，y 是预测框的中心点坐标，height, width
            是预测框的高度和宽度。对于一个训练好的模型，这 4 个数值的范围都应该在
            [0, 608] 之间。
    """

    backbone_outputs = csp_darknet53(inputs=inputs, training=training)
    neck_outputs = panet(inputs=backbone_outputs, training=training)
    head_outputs = heads(inputs=neck_outputs, training=training)

    return head_outputs


def create_model(input_shape=None):
    """创建一个新的 yolo_v4_csp 模型。"""

    if input_shape is None:
        input_shape = *MODEL_IMAGE_SIZE, 3

    keras.backend.clear_session()

    # inputs 是一个 Keras tensor，也叫符号张量 symbolic tensor，这种张量没有实际的值，
    # 只是在创建模型的第一步--构建计算图时会用到。模型创建好之后就不再使用符号张量。
    inputs = keras.Input(shape=input_shape)
    outputs = yolo_v4_csp(inputs=inputs)

    model = keras.Model(
        inputs=inputs, outputs=outputs, name='yolo_v4_csp_model')

    return model


def _transform_predictions(prediction, anchor_boxes):
    """将模型的预测结果转换为模型输入的图片大小，可以在全局变量 MODEL_IMAGE_SIZE 中
    设置模型输入图片大小。

    将每个长度为 85 的预测结果进行转换，第 0 位为置信度 confidence，第 1 位到第 81
    位为分类的 one-hot 编码。置信度和分类结果都需要用 sigmoid 转换为 [0, 1]之间的数，
    相当于转换为概率值。最后 4 位是探测框的预测结果，需要根据 YOLO V3 论文进行转换。
    倒数第 4 位到倒数第 3 位为预测框的中心点，需要对中心点用 sigmoid 函数进行转换。然
    后乘以一个比例，就得到中心点在模型输入图片中的实际坐标值。
    倒数第 2 位和最后一位是探测框的宽度和高度，需要先用指数函数转换为非负数，再乘以探测
    框的高度和宽度。

    Arguments:
        prediction: 一个 3D 张量，形状为 (N, *batch_size_feature_map, 255)，是
            模型的 3 个预测结果张量之一。height, width 是特征图大小。使用时 Keras
            将会自动插入一个第 0 维度，作为批量维度。
        anchor_boxes: 一个元祖，其中包含 3 个元祖，代表了当前 prediction 所对应
            的 3 个预设框。

    Returns:
        transformed_prediction: 一个 4D 张量，形状为 (*FEATURE_MAP_Px, 3, 85)。
            FEATURE_MAP_Px 是 p5, p4, p3 特征图大小。3 是特征图每个位置上，预设框
            的数量。85 是单个预测结果的长度。
            长度为 85 的预测向量，第 0 为表示是否有物体的概率， 第 1 位到第 80 位，
            是表示物体类别的 one-hot 编码，而最后 4 位，则分别是物体框 bbox 的参数
            (center_x, center_y, height, width)。
    """

    # 下面的注释以 p5 为例。在 p5 的 19x19 个位置上，将每个长度为 255 的向量，转换为
    #  3x85 的形状。
    #  85 位分别表示 [confidence, classification..., tx, ty, th, tw]，其中
    #  classification 部分，共包含 80 位，最后 4 位是探测框的中心点坐标和大小。

    # prediction 的形状为 (N, *batch_size_feature_map, 255), N 为 batch_size。
    # 进行 reshape时，必须带上 batch_size，所以用 batch_size_feature_map
    batch_size_feature_map = prediction.shape[: 3]
    prediction = tf.reshape(prediction,
                            shape=(*batch_size_feature_map, 3, 85))

    # get_probability 形状为 (N, 19, 19, 3, 81)，包括置信度和分类结果两部分。
    get_probability = tf.math.sigmoid(prediction[..., : 81])

    # confidence 形状为 (N, 19, 19, 3, 1) 需要配合使用 from_logits=False
    confidence = get_probability[..., : 1]

    # classification 形状为 (N, 19, 19, 3, 80)，需要配合使用 from_logits=False
    classification = get_probability[..., 1: 81]

    # prediction 的形状为 (N, 19, 19, 3, 85), N 为 batch_size。
    feature_map = prediction.shape[1: 3]

    # 根据 YOLO V3 论文中的 figure 2，需要对 bbox 坐标和尺寸进行转换。tx_ty 等标
    # 注记号和论文的记号一一对应。
    # tx_ty 形状为 (N, 19, 19, 3, 2)，分别代表 tx, ty。
    tx_ty = prediction[..., -4: -2]
    # 根据 YOLO V3论文，需要先取得 cx_cy。cx_cy 实际是一个比例值，在计算 IOU 和损失
    # 值之前，应该转换为 608x608 大小图片中的实际值。
    # 注意，根据论文 2.1 节第一段以及配图 figure 2，cx_cy 其实是每一个 cell
    # 的左上角点，这样预测框的中心点 bx_by 才能达到该 cell 中的每一个位置。
    grid = tf.ones(shape=feature_map)  # 构造一个 19x19 的网格
    cx_cy = tf.where(grid)  # where 函数可以获取张量的索引值，也就是 cx, cy
    cx_cy = tf.cast(x=cx_cy, dtype=tf.float32)  # cx_cy 原本是 int64 类型

    # cx_cy 的形状为 (361, 2)， 361 = 19 x 19，下面将其形状变为 (1, 19, 19, 1, 2)
    cx_cy = tf.reshape(cx_cy, shape=(1, *feature_map, 2))
    cx_cy = cx_cy[..., tf.newaxis, :]  # 展示一下 tf.newaxis 的用法

    #  cx_cy 的形状为 (1, 19, 19, 1, 2), tx_ty 的形状为 (N, 19, 19, 3, 2)
    bx_by = tf.math.sigmoid(tx_ty) + cx_cy

    # 下面根据 th, tw, 计算 bh, bw。th_tw 形状为 (N, 19, 19, 3, 2)
    th_tw = prediction[..., -2:]
    ph_pw = tf.convert_to_tensor(anchor_boxes, dtype=tf.float32)
    # ph_pw 的形状为 (3, 2)，和上面的 cx_cy 同理，需要将 ph_pw 的形状变为
    # (1, 1, 3, 2)
    ph_pw = tf.reshape(ph_pw, shape=(1, 1, 1, 3, 2))
    # 此时 ph_pw 和 th_tw 的张量阶数 rank 相同，会自动扩展 broadcast，进行算术运算。
    bh_bw = ph_pw * tf.math.exp(th_tw)

    # 在计算 CIOU 损失时，如果高度宽度过大，计算预测框面积会产生 NaN 值，导致模型无法
    # 训练。所以把预测框的高度宽度限制到不超过图片大小即可。
    bh_bw = tf.clip_by_value(
        bh_bw, clip_value_min=0, clip_value_max=MODEL_IMAGE_SIZE[0])

    # bx_by，bh_bw 为比例值，需要转换为在 608x608 大小图片中的实际值。
    image_scale_height = MODEL_IMAGE_SIZE[0] / feature_map[0]
    image_scale_width = MODEL_IMAGE_SIZE[1] / feature_map[1]
    image_scale = image_scale_height, image_scale_width

    # bx_by 是一个比例值，乘以比例 image_scale 之后，bx_by 将代表图片中实际
    # 的长度数值。比如此时 bx, by 的数值可能是 520， 600 等，数值范围 [0, 608]
    # 而 bh_bw 已经是一个长度值，不需要再乘以比例。
    bx_by *= image_scale

    bx_by = tf.clip_by_value(
        bx_by, clip_value_min=0, clip_value_max=MODEL_IMAGE_SIZE[0])

    transformed_prediction = tf.concat(
        values=[confidence, classification, bx_by, bh_bw], axis=-1)

    return transformed_prediction


def predictor(inputs):
    """对模型输出的 1 个 head 进行转换。

    转换方式为：
    先将 head 的形状从 (batch_size, height, width, 255) 变为 (batch_size, height,
    width, 3, 85)。将每个长度为 85 的预测结果进行转换，第 0 位为置信度 confidence，
    第 1 位到第 81位为分类的 one-hot 编码，均需要用 sigmoid 转换为 [0, 1] 之间的数。
    最后 4 位是探测框的预测结果，需要根据 YOLO V3 论文进行转换。
    倒数第 4 位到倒数第 3 位为预测框的中心点，需要对中心点用 sigmoid 函数进行转换。然
    后乘以一个比例，就得到中心点在模型输入图片中的实际坐标值。
    倒数第 2 位和最后一位是探测框的宽度和高度，需要先用指数函数转换为非负数，再乘以探测
    框的高度和宽度。

    Arguments:
        inputs: 一个元祖，包含来自 Heads 输出的 3个 3D 张量。分别表示为
            p5_head, p4_head, p3_head。使用时 Keras 将会自动插入一个第 0 维度，
            作为批量维度。
    Returns:
        p5_prediction: 一个 5D 张量，形状为 (batch_size, height, width, 3, 85)。
            height, width 是特征图大小。3 是特征图的每个位置上，预设框的数量。
            85 是单个预测结果的长度。下面 p4_prediction， p3_prediction 也是一样。
            p5_prediction 的 height, width 为 19, 19.
        p4_prediction: 一个 5D 张量，形状为 (batch_size, 38, 38, 3, 85)。
        p3_prediction: 一个 5D 张量，形状为 (batch_size, 76, 76, 3, 85)。
    """

    # px_head 代表 p5_head, p4_head, p3_head
    px_head = inputs
    feature_map_size = px_head.shape[1: 3]

    anchor_boxes_px = None
    if feature_map_size == (*FEATURE_MAP_P5,):
        anchor_boxes_px = ANCHOR_BOXES_P5
    elif feature_map_size == (*FEATURE_MAP_P4,):
        anchor_boxes_px = ANCHOR_BOXES_P4
    elif feature_map_size == (*FEATURE_MAP_P3,):
        anchor_boxes_px = ANCHOR_BOXES_P3

    px_prediction = _transform_predictions(px_head, anchor_boxes_px)

    return px_prediction
