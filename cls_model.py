from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, normalization, Lambda, BatchNormalization, ReLU
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
import keras.backend as K
import tensorflow as tf
from keras.callbacks import ModelCheckpoint


def cls_model(input_shape=(28,28,1)):

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
    # x = Dense(128, activation='relu')(x)
    x = Dense(3, activation='softmax')(x)

    model = Model(inpt, x)
    model.compile(SGD(5e-3,5e-4), loss=categorical_crossentropy, metrics=['acc'])

    return model


if __name__ == '__main__':

    import numpy as np
    import cv2
    import glob

    data_dir = "/Users/amber/dataset/mnist"

    # train
    # x_train = np.zeros((1500, 28, 28, 1))
    # y_train = np.zeros((1500, 3))

    # for idx, folder in enumerate(['d0', 'd1', 'd2']):
    #     filelst = glob.glob(data_dir + '/' + folder + '/*png')
    #     cnt = 0
    #     while cnt < 500:
    #         img = cv2.imread(filelst[cnt], 0)
    #         x_train[500*idx+cnt, :, :, 0] = img / 255.
    #         y_train[500*idx+cnt][idx] = 1
    #         cnt += 1

    # print(x_train.shape, np.max(x_train))
    # print(y_train.shape, np.max(y_train))

    # model = cls_model(input_shape=(28,28,1))
    # # model.load_weights("ce_cls3_ep_14_loss_0.281.h5")
    # filepath = "ce_cls3_ep_{epoch:02d}_loss_{loss:.3f}.h5"
    # checkpoint = ModelCheckpoint(filepath, monitor='loss', mode='min',verbose=1)
    # model.fit(x=x_train, y=y_train, shuffle=True,
    #           batch_size=32, epochs=20,
    #           verbose=1, validation_split=0.1,
    #           callbacks=[checkpoint])

    # test
    x_test = np.zeros((300, 28, 28, 1))
    y_test = np.zeros((300, 1))

    for idx, folder in enumerate(['d0', 'd1', 'd2']):
        filelst = glob.glob(data_dir + '/' + folder + '/*png')
        cnt = 0
        while cnt < 100:
            img = cv2.imread(filelst[cnt+1000], 0)
            x_test[100*idx+cnt, :, :, 0] = img / 255.
            y_test[100*idx+cnt] = idx
            cnt += 1

    model = cls_model(input_shape=(28,28,1))
    model.load_weights("ce_cls3_ep_20_loss_0.074.h5")

    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)

    print('acc:', np.sum(y_test==y_pred) / 300.)







