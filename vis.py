import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from cls_model import cls_model
from circle_loss import circle_model
from keras.models import Model
import cv2
import glob


if __name__ == '__main__':

    data_dir = "/Users/amber/dataset/mnist"

    # test
    x_test = np.zeros((300, 28, 28, 1))
    y_test = np.zeros((300, 3))

    for idx, folder in enumerate(['d0', 'd1', 'd2']):
        filelst = glob.glob(data_dir + '/' + folder + '/*png')
        cnt = 0
        while cnt < 100:
            img = cv2.imread(filelst[cnt+1000], 0)
            x_test[100*idx+cnt, :, :, 0] = img / 255.
            y_test[100*idx+cnt][idx] = 1
            cnt += 1

    model = cls_model(input_shape=(28,28,1))
    model.load_weights("weights/ce_cls3_ep_20_loss_0.074.h5")
    newmodel = Model(model.inputs, model.get_layer(index=-1).output)

    # model = circle_model(input_shape=(28,28,1))
    # model.load_weights("weights/circle_cls3_ep_20_loss_4.050.h5")
    # newmodel = Model(model.inputs, model.get_layer(index=-1).output)

    y_pred = newmodel.predict([x_test])
    print(y_pred.shape)

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(y_pred)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    plt.figure(figsize=(6, 6))
    color = ['r', 'g', 'b']
    for i in range(3):
        plt.scatter(X_norm[i*100:i*100+100,1], X_norm[i*100:i*100+100,0], color=color[i])
        for j in range(100):
            plt.text(X_norm[i*100+j,1], X_norm[i*100+j,0], str(i))
    plt.show()




