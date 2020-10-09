from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, normalization, Lambda, BatchNormalization, ReLU
from keras.optimizers import SGD, Adam
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import tensorflow as tf


def cal_similarity(features):
    sim_mat = features @ K.transpose(features)   # (N,N)
    abs_mat = K.sqrt(K.sum(K.square(features), axis=1, keepdims=True))
    abs_mat = abs_mat @ K.transpose(abs_mat)
    sim_mat = sim_mat / abs_mat
    return sim_mat


def circle_loss(y_true, y_pred):

    def circle_loss_(features, labels, scale=64, margin=0.25):
        # labels: (N,cls) one-hot label
        # features: (N,k) feature embedding
        sim_mat = cal_similarity(features)
        label_mat = K.cast(labels @ K.transpose(labels), tf.bool)
        sin_mat_p = tf.where(label_mat, sim_mat, tf.zeros_like(sim_mat))
        sin_mat_n = tf.where(label_mat, tf.zeros_like(sim_mat), sim_mat)

        alpha_p = K.relu(1 + margin - sin_mat_p)
        alpha_n = K.relu(sin_mat_n + margin)

        delta_p = 1 - margin
        delta_n = margin

        circle_loss_n = K.log(1 + K.sum(K.exp(scale*alpha_n*(sin_mat_n-delta_n))))
        circle_loss_p = K.log(1 + K.sum(K.exp(scale*alpha_p*(sin_mat_p-delta_p))))

        loss = circle_loss_n - circle_loss_p
        # loss = tf.Print(loss, [circle_loss_n, circle_loss_p])

        return loss

    return circle_loss_(y_pred, y_true)


def circle_model(input_shape=(28,28,1)):

    inpt = Input(input_shape)

    x = Conv2D(16, 3, strides=1, padding='same')(inpt)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(16, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)

    x = Conv2D(32, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(32, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)

    x = Conv2D(64, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(64, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    # inpt2 = Input((3,))
    # x = Lambda(circle_loss, arguments={'labels': inpt2})(x)

    model = Model(inpt, x)
    # model.compile(Adam(1e-2,5e-4), loss=lambda y_true, y_pred: y_pred, metrics=[])
    model.compile(Adam(1e-2,5e-4), loss=circle_loss, metrics=[])

    return model


if __name__ == '__main__':

    import numpy as np
    import cv2
    import glob

    data_dir = "/Users/amber/dataset/mnist"

    # train
    x_train = np.zeros((1500, 28, 28, 1))
    y_train = np.zeros((1500, 3))

    for idx, folder in enumerate(['d0', 'd1', 'd2']):
        filelst = glob.glob(data_dir + '/' + folder + '/*png')
        cnt = 0
        while cnt < 500:
            img = cv2.imread(filelst[cnt], 0)
            x_train[500*idx+cnt, :, :, 0] = img / 255.
            y_train[500*idx+cnt][idx] = 1
            cnt += 1

    print(x_train.shape, np.max(x_train))
    print(y_train.shape, np.max(y_train))

    model = circle_model(input_shape=(28,28,1))
    model.load_weights("circle_cls3_ep_20_loss_5.994.h5")
    filepath = "circle_cls3_ep_{epoch:02d}_loss_{loss:.3f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', mode='min',verbose=1, save_weights_only=True)
    model.fit(x=x_train, y=y_train, shuffle=True,
              batch_size=64, epochs=20,
              verbose=1, validation_split=0.1,
              callbacks=[checkpoint])




