import sklearn.datasets
import matplotlib.pyplot as plt
import numpy as np

def draw_separator(X, Y, sess, X_placeholder, class_):
    x_min, x_max = X[:, 0].min() - 0.3 , X[:, 0].max() + 0.3
    y_min, y_max = X[:, 1].min() - 0.3 , X[:, 1].max() + 0.3
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

    # here "model" is your model's prediction (classification) function
    Z = sess.run(class_, feed_dict={X_placeholder:np.c_[xx.ravel(), yy.ravel()] }) 

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.axis('on')

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], s=40, c=Y, cmap=plt.cm.Spectral)


def xor_dataset():
    dataset_X = np.zeros((4,2))

    dataset_X[1][1] = 1
    dataset_X[2][0] = 1
    dataset_X[3][0] = 1
    dataset_X[3][1] = 1

    dataset_Y = np.zeros((4))
    dataset_Y[1] = 1
    dataset_Y[2] = 1

    return dataset_X, dataset_Y
  