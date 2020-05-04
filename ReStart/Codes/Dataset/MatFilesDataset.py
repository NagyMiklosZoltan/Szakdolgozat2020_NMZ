import h5py
import numpy as np

fMRI_mat = r'C:\Users\NagyMiklosZoltan\PycharmProjects\Szakdolgozat2020\algonautsChallenge2019/Training_Data/92_Image_Set/target_fmri.mat'

fMRI = h5py.File(fMRI_mat, 'r')
print(fMRI.keys())

data = fMRI.get('EVC_RDMs')
data = np.array(data)

print(np.shape(data))
average = np.mean(data, axis=2)
print(np.shape(average))



