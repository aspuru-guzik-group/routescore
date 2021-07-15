import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('scientific')


def plot_score(func, fname):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # func[func>15] = np.nan

    surf = ax.plot_surface(X, Y, func, cmap=cm.viridis)
    ax.set_xlabel('Human time (h)')
    ax.set_ylabel('Robot time (h)')
    ax.set_zlabel('TTC (h)')
    # ax.set_title(title)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 60)
    # ax.set_zlim(0,10)

    # fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


tH = np.linspace(0, 6)
tM = np.linspace(0, 60)
X, Y = np.meshgrid(tH, tM)


def time_FIN(x, y):
    # global cH, cM
    cH = 52.97
    cM = 17.13
    return np.sqrt(((cH / cM) * x) ** 2 + ((cM / cH * y) ** 2))


plot_score(time_FIN(X, Y), 'Figure_2')
