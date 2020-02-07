# http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/
import SimpleITK
import matplotlib.pyplot as plt


image = SimpleITK.Image(256, 128, 64, SimpleITK.sitkInt16)
image_2D = SimpleITK.Image(64, 64, SimpleITK.sitkFloat32)
image_2D = SimpleITK.Image([32, 32], SimpleITK.sitkUInt32)
image_RGB = SimpleITK.Image([128, 128], SimpleITK.sitkVectorUInt8, 3)

print(image.GetSize())
print(image.GetOrigin())
print(image.GetSpacing())
print(image.GetDirection())
print(image.GetDimension())
print(image.GetNumberOfComponentsPerPixel())
print(image.GetWidth())
print(image.GetHeight())
print(image.GetDepth())
print(image.GetPixelIDValue())
print(image.GetPixelIDTypeAsString())

for key in image.GetMetaDataKeys():
    print('{0}: {1}'.format(key, image.GetMetaData(key)))

nda = SimpleITK.GetArrayFromImage(image_RGB)
img = SimpleITK.GetImageFromArray(nda, isVector=True)
print(img)
img = SimpleITK.GaussianSource(size=[64]*2)
img = SimpleITK.GaborSource(size=[60]*2, frequency=0.03)
img = img ** 2
'''
import numpy
import glob
import random
batch_size = 1
data_path = 'D:\Halimeh\Datasets\MSD\Task05_Prostate\imagesTr'
files = glob.glob(data_path + '\p*')
random_index = numpy.random.choice(len(files), size=batch_size, replace=False)
c = random.choices(files, k=2)
print('c', type(c), c)

print('ffff', files[0, 1, 2])
l = random_index.tolist()
print('l', type(l), l)
print('random_index', type(random_index), random_index, len(files), type(files))
batch_files = files[l]
print('batch_files', batch_files)
'''
'''
print('tttt', 0 % 4, 1 % 4, 2 % 4, 4 % 4)
def f():
    for i in range(5):
        yield i * 10
        print('i', i)

a = f()
b = next(a)
print('b', b)
c = next(a)
print('c', c)
'''
import glob
import nibabel
import nibabel.arraywriters
import random
import skimage
import numpy

batch_size = 1
def prepare_data_batch():
    print('FUNCTION START')
    data_path = 'D:\Halimeh\Datasets\MSD\Task05_Prostate\imagesTr'
    files = glob.glob(data_path + '\p*')
    batch_A_images = []
    batch_B_images = []
    while True:
        print('WHILE START')
        file_path = random.choice(files)
        print('file_path', file_path)
        # for file_path in batch_files:
        nifti = nibabel.load(file_path)

        slices = nifti.get_fdata()
        print('slices', slices.shape)
        scaling, intercept, mn, mx = nibabel.volumeutils.calculate_scale(slices, numpy.uint8, allow_intercept=True)
        print('scaling', scaling, 'intercept', intercept)
        aw = nibabel.arraywriters.SlopeArrayWriter(slices, calc_scale=False)
        print(aw.calc_scale())
        s, i = nibabel.arraywriters.get_slope_inter(aw)
        print('s', s, 'i', i)
        slices = (slices - intercept) / scaling
        slices[slices > 255] = 255
        slices[slices < 0] = 0
        slices = (slices - 127.5) / 127.5
        slices = skimage.transform.resize(slices, output_shape=(
            256, 256, slices.shape[2], slices.shape[3]))
        for v in range(slices.shape[2]):
            print('FOR START', v)
            # print('v', v)
            mi = slices[..., v, 0]
            mi = mi[:, :, None]
            mj = slices[..., v, 1]
            mj = mj[:, :, None]
            batch_A_images.append(mi)
            batch_B_images.append(mj)
            if (v + 1) % batch_size == 0:
                numpy.random.shuffle(batch_A_images)
                numpy.random.shuffle(batch_B_images)
                # print('batch_A_images', type(batch_A_images), len(batch_A_images))
                # print('batch_B_images', type(batch_B_images), len(batch_B_images))
                yield batch_A_images, batch_B_images
                batch_A_images = []
                batch_B_images = []
                # print(len(batch_A_images))
                # print(len(batch_B_images))

g = prepare_data_batch()
for i in range(35):
    a, b = next(g)
    print('a', type(a))
    print('b', type(b))

# plt.imshow(SimpleITK.GetArrayViewFromImage(img), cmap='gray')



# plt.show()

