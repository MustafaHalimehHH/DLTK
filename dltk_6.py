# http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/20_Expand_With_Interpolators.html
import SimpleITK
import numpy
import math
import matplotlib.pyplot as plt


def my_show(img, title=None, margin=0.05):
    if img.GetDimension() == 3:
        img = SimpleITK.Tile((img[img.GetSize()[0] // 2, :, :],
                              img[:, img.GetSize()[1] // 2, :],
                              img[:, :, img.GetSize()[2] // 2]),
                             [2, 2])
    aimg = SimpleITK.GetArrayViewFromImage(img)
    print('aimg', aimg.shape)
    x_size, y_size = aimg.shape
    dpi = 80
    figsize = (1 + margin) * y_size / dpi, (1 + margin) * x_size / dpi
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])
    t = ax.imshow(aimg)
    if len(aimg.shape) == 2:
        t.set_cmap('gray')
    if title:
        plt.title(title)
    plt.show()


def marscher_lobb(size=40, alpha=0.25, f_M=6.0):
    img = SimpleITK.PhysicalPointSource(SimpleITK.sitkVectorFloat32, [size] * 3, [-1] * 3, [2.0 / size] * 3)
    imgx = SimpleITK.VectorIndexSelectionCast(img, 0)
    imgy = SimpleITK.VectorIndexSelectionCast(img, 1)
    imgz = SimpleITK.VectorIndexSelectionCast(img, 2)
    del img
    r = SimpleITK.Sqrt(imgx ** 2 + imgy ** 2)
    print('r', r.GetDimension())
    del imgx, imgy
    pr = SimpleITK.Cos((2.0 * math.pi * f_M) * SimpleITK.Cos((math.pi / 2.0) * r))
    print('pr', pr.GetDimension())
    return (1.0 - SimpleITK.Sin((math.pi / 2.0) * imgz) + alpha * (1.0 + pr)) / (2.0 * (1.0 + alpha))


my_show(marscher_lobb())
my_show(marscher_lobb(100))

ml = marscher_lobb()
ml = ml[:, :, ml.GetSize()[-1] // 2]

my_show(SimpleITK.Expand(ml, [15] * 3, SimpleITK.sitkNearestNeighbor), title='nearest neighbour')
my_show(SimpleITK.Expand(ml, [15] * 3, SimpleITK.sitkLinear), title='linear')

