# https://openneuro.org/datasets/ds000002/versions/00002
import os
import SimpleITK
import numpy
import matplotlib.pyplot as plt


DATA_PATH = 'C:\\Users\\halimeh\\Downloads\\sub-01_func_sub-01_task-deterministicclassification_run-01_bold.nii.gz'

img = SimpleITK.ReadImage(DATA_PATH)
print('img', img)
arr = SimpleITK.GetArrayFromImage(img)
print('arr', arr.shape)
print('arr', arr.shape[::-1])
arr2 = arr.transpose(2, 3, 1, 0)
print('arr2', arr2.shape)

plt.imshow(arr2[..., 14, 4], cmap='gray')
plt.show()

# Transverse-Axial X-Z
axial = numpy.transpose(arr2, [0, 2, 1, 3])
print('axial', axial.shape)
plt.imshow(axial[..., 10, 10], cmap='gray')
plt.title('Axial')
plt.show()

# Coronal-Frontal X-Y
coronal = numpy.transpose(arr2, [0, 1, 2, 3])
print('coronal', coronal.shape)
plt.imshow(coronal[..., 10, 10], cmap='gray')
plt.title('Coronal')
plt.show()

# Sagittal Y-Z
sagittal = numpy.transpose(arr2, [1, 2, 0, 3])
print('sagittal', sagittal.shape)
plt.imshow(sagittal[..., 10, 10], cmap='gray')
plt.title('Sagittal')
plt.show()