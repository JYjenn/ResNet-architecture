import tensorflow as tf
from .resnet1d import ResNet, BottleneckBlock


def ecg_feature_extractor(arch=None, stages=None, initial_blocks=(64, 7, 2)):
    if arch is None or arch == 'resnet18':
        resnet = ResNet(num_outputs=None,
                        blocks=(2, 2, 2, 2)[:stages],
                        kernel_size=(7, 5, 5, 3),
                        include_top=False,
                        initial_blocks=initial_blocks)
    elif arch == 'resnet34':
        resnet = ResNet(num_outputs=None,
                        blocks=(3, 4, 6, 3)[:stages],
                        kernel_size=(7, 5, 5, 3),
                        include_top=False)
    elif arch == 'resnet50':
        resnet = ResNet(num_outputs=None,
                        blocks=(3, 4, 6, 3)[:stages],
                        kernel_size=(7, 5, 5, 3),
                        block_fn=BottleneckBlock,
                        include_top=False)
    elif arch == 'resnet18_3':
        resnet = ResNet(num_outputs=None,
                        blocks=(2, 2, 2, 2)[:stages],
                        kernel_size=(3, 3, 3, 3),
                        include_top=False)
    else:
        raise ValueError('unknown architecture: {}'.format(arch))
    feature_extractor = tf.keras.Sequential([
        resnet,
        tf.keras.layers.GlobalAveragePooling1D()
    ])
    return feature_extractor


def hr_feature_extractor(arch=None, stages=None,
                         initial_blocks=(64, 7, 2),
                         blocks=(2, 2, 2, 2),
                         filters=(8, 16, 32, 64),
                         num_outputs=2,
                         kernel_size=(7, 5, 5, 3)):
    if arch is None or arch == 'resnet18':
        resnet = ResNet(num_outputs=num_outputs,
                        initial_blocks=initial_blocks,
                        blocks=blocks[:stages],
                        kernel_size=kernel_size,
                        filters=filters,
                        include_top=True)
    elif arch == 'resnet34':
        resnet = ResNet(num_outputs=num_outputs,
                        initial_blocks=initial_blocks,
                        blocks=(3, 4, 6, 3)[:stages],
                        kernel_size=kernel_size,
                        filters=filters,
                        include_top=True)
    elif arch == 'resnet50':
        resnet = ResNet(num_outputs=num_outputs,
                        initial_blocks=initial_blocks,
                        blocks=(3, 4, 6, 3)[:stages],
                        kernel_size=kernel_size,
                        block_fn=BottleneckBlock,
                        filters=filters,
                        include_top=True)
    else:
        raise ValueError('unknown architecture: {}'.format(arch))

    return resnet
