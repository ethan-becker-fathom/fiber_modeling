import scipy.io

file_location = 'Lumerical_Results/TAF_8.3-20-30_dispersion-850nm_bend-0.0mm_full-dataset.mat'
mat = scipy.io.loadmat(file_location)

# print(mat)
print(mat['1']['E'])


# import h5py
# with h5py.File(file_location, 'r') as f:
#     print(f.keys())