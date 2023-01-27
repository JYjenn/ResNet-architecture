import tensorflow as tf
from .modules.ecg_feature_net import ecg_feature_extractor


def get_noise_model_with_stage_info(input_shape,
                                    resnet_architecture, pretrain_weight_path=None,
                                    output_len=2,
                                    output_len_noise=2,
                                    initial_blocks=(64, 7, 2)
                                    ):
    # error handling
    if not resnet_architecture in ["resnet18", "resnet34", "resnet50"]:
        raise ValueError('resnet_architecture --> ["resnet18", "resnet34", "resnet50"]')
    # noise model
    input_noise_model = tf.keras.Input(input_shape)
    model_resnet17_noise_pre_new = get_model_heart_rate_resnet(arch=resnet_architecture, initial_blocks=initial_blocks)
    model_resnet17_noise_features = model_resnet17_noise_pre_new.layers[0](input_noise_model)
    out_glob = tf.keras.layers.GlobalAveragePooling1D()(model_resnet17_noise_features)
    out = tf.keras.layers.Dense(output_len, activation='softmax', name='sas_label')(out_glob)
    out_noise = tf.keras.layers.Dense(output_len_noise, activation='softmax', name='noise_label')(out_glob)
    out_stage = tf.keras.layers.Dense(output_len_noise, activation='softmax', name='stage_label')(out_glob)

    model = tf.keras.Model(input_noise_model, [out, out_noise, out_stage])

    if pretrain_weight_path is not None:
        model.load_weights(pretrain_weight_path)
        model.trainable = False
    else:
        pass
        model.compile(optimizer='ADAM',
                      loss=[tf.keras.losses.CategoricalCrossentropy(),
                            tf.keras.losses.CategoricalCrossentropy(),
                            tf.keras.losses.CategoricalCrossentropy()],
                      metrics=['accuracy'])
    return model


def get_model_heart_rate_resnet(arch='resnet18', initial_blocks=(64, 7, 2)):

    model = ecg_feature_extractor(arch=arch, initial_blocks=initial_blocks)

    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    model.compile(optimizer='ADAM',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
