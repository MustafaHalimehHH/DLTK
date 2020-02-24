# http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/300_Segmentation_Overview.html
import os
from ipywidgets import widgets, interact, interactive
import SimpleITK
import numpy
from NNKeras import downloaddata
import matplotlib.pyplot as plt



def my_show(img, title=None, margin=0.05, dpi=80):
    nda = SimpleITK.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    slicer = False

    if nda.ndim == 3:
        c = nda.shape[-1]
        if not c in (3, 4):
            slicer = True
    elif nda.ndim == 4:
        c = nda.shape[-1]
        if not c in (3, 4):
            raise RuntimeError('Unable to show 3D-vector image')
        slicer = True

    if (slicer):
        y_size = nda.shape[1]
        x_size = nda.shape[2]
    else:
        y_size = nda.shape[0]
        x_size = nda.shape[1]

    fig_size = (1 + margin) * y_size / dpi, (1 + margin) * x_size / dpi

    def callback(z=None):
        extent = (0, x_size * spacing[1], y_size * spacing[0], 0)
        fig = plt.figure(figsize=fig_size, dpi=dpi)
        ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])
        plt.set_cmap('gray')
        if z is None:
            ax.imshow(nda, extent=extent, interpolation=None)
        else:
            ax.imshow(nda[z, ...], extent=extent, interpolation=None)
        if title:
            plt.title(title)

        plt.show()

    if slicer:
        interact(callback, z=(0, nda.shape[0]-1))
    else:
        callback()


def my_show_3d(img, x_slices=[], y_slices=[], z_slices=[], title=None, margin=0.05, dpi=80):
    size = img.GetSize()
    img_x_slices = [img[x, :, :] for x in x_slices]
    img_y_slices = [img[:, y, :] for y in y_slices]
    img_z_slices = [img[:, :, z] for z in z_slices]

    max_len = max(len(x_slices), len(y_slices), len(z_slices))
    img_null = SimpleITK.Image([0, 0], img.GetPixelID(), img.GetNumberOfComponentsPerPixel())
    img_slices = []
    d = 0

    if len(img_x_slices):
        img_slices += img_x_slices + [img_null] * (max_len - len(img_x_slices))
        d += 1
    if len(img_y_slices):
        img_slices += img_y_slices + [img_null] * (max_len - len(img_y_slices))
        d += 1
    if len(img_z_slices):
        img_slices += img_z_slices + [img_null] * (max_len - len(img_z_slices))
        d += 1

    if max_len != 0:
        if img.GetNumberOfComponentsPerPixel() == 1:
            img = SimpleITK.Tile(img_slices, [max_len, d])
        else:
            img_comps = []
            for i in range(0, img.GetNumberOfComponentsPerPixel()):
                img_slices_c = [SimpleITK.VectorIndexSelectionCast(s, i) for s in img_slices]
                img_comps.append(SimpleITK.Tile(img_slices_c, [max_len, d]))
            img = SimpleITK.Compose(img_comps)
    my_show(img, title, margin, dpi)


img_T1 = SimpleITK.ReadImage(downloaddata.fetch_data('nac-hncma-atlas2013-Slicer4Version/Data/A1_grayT1.nrrd'))
img_T2 = SimpleITK.ReadImage(downloaddata.fetch_data('nac-hncma-atlas2013-Slicer4Version/Data/A1_grayT2.nrrd'))
img_T1_255 = SimpleITK.Cast(SimpleITK.RescaleIntensity(img_T1), SimpleITK.sitkUInt8)
img_T2_255 = SimpleITK.Cast(SimpleITK.RescaleIntensity(img_T2), SimpleITK.sitkUInt8)
my_show_3d(img_T1)
plt.show()