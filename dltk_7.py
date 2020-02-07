#
import os
import numpy
import SimpleITK
import nibabel
import glob


def unify_spacing(image, interpolator=SimpleITK.sitkLinear):
    target_spacing = (0.60, 0.60, 4.0)
    original_spacing = image.GetSpacing()
    # print('original_spacing', original_spacing)
    # if all(spc == original_spacing[0] for spc in original_spacing):
    # return SimpleITK.Image(image)
    if all(numpy.isclose(origin_spc, target_spc) for origin_spc, target_spc in zip(original_spacing, target_spacing)):
        print('[*] Equal')
        return SimpleITK.Image(image)

    else:
        print('[!] Not Equal')
        arr = SimpleITK.GetArrayFromImage(image)
        # print('arr', arr.shape, arr[0, ...].shape)
        image_3d = SimpleITK.GetImageFromArray(arr[0, ...])
        # print('image_3d', image_3d.GetSize())
        # image_3d = SimpleITK.Compose(
            # [SimpleITK.VectorIndexSelectionCast(image, channel) for channel in range(3)])
        original_size = image_3d.GetSize()
        # min_spacing = min(original_spacing)
        min_spacing = min(target_spacing)
        new_spacing = [min_spacing] * image_3d.GetDimension()
        new_size = [int(round(osz * ospc / min_spacing)) for osz, ospc in zip(original_size, original_spacing)]
        # print('new_size', new_size)
        return SimpleITK.Resample(image_3d, new_size, SimpleITK.Transform(), interpolator, image_3d.GetOrigin(), new_spacing,
                                  image_3d.GetDirection(), 0, image_3d.GetPixelID())



DATA_PATH = 'D:\Halimeh\Datasets\MSD\Task05_Prostate\imagesTr\prostate_02.nii.gz'
sitk = SimpleITK.ReadImage(DATA_PATH)
print('img', sitk)
# ni = nibabel.load(DATA_PATH)
# print('ni', ni)
print('SimpleITK', sitk.GetOrigin())
print('SimPleITK', sitk.GetSize())
print('SimpleITK', sitk.GetSpacing())

DATA_PATH = 'D:\Halimeh\Datasets\MSD\Task05_Prostate\imagesTr'
files = glob.glob(DATA_PATH + '/p*')
i = 0
for f in files:
    print('---' * 20)
    if i > 3:
        break
    i += 1
    sitk = SimpleITK.ReadImage(f)
    # print(type(sitk.GetOrigin()))
    print('Origin {:4.2f}, {:4.2f}, {:4.2f}, {:4.2f}'.format(*sitk.GetOrigin()),
          'Spacing {:4.2f}, {:4.2f}, {:4.2f}, {:4.2f}'.format(*sitk.GetSpacing()),
          'Size', sitk.GetSize())
    sitk = unify_spacing(sitk)
    print('Origin {:4.2f}, {:4.2f}, {:4.2f}'.format(*sitk.GetOrigin()),
          'Spacing {:4.2f}, {:4.2f}, {:4.2f}'.format(*sitk.GetSpacing()),
          'Size', sitk.GetSize())

