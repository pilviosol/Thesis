
import pathlib
import os
import shutil
from functions import resample


SR_16kHz = 16000

vn_train_path = "/nas/home/spol/Thesis/URPM_vn_fl/vn_train"
vn_train_resampled_path = "/nas/home/spol/Thesis/URPM_vn_fl/vn_train_resampled"
vn_test_path = "/nas/home/spol/Thesis/URPM_vn_fl/vn_test"
vn_test_resampled_path = "/nas/home/spol/Thesis/URPM_vn_fl/vn_test_resampled"
fl_train_path = "/nas/home/spol/Thesis/URPM_vn_fl/fl_train"
fl_train_resampled_path = "/nas/home/spol/Thesis/URPM_vn_fl/fl_train_resampled"
fl_test_path = "/nas/home/spol/Thesis/URPM_vn_fl/fl_test"
fl_test_resampled_path = "/nas/home/spol/Thesis/URPM_vn_fl/fl_test_resampled"


try:
    shutil.rmtree(vn_train_resampled_path, ignore_errors=True)
    shutil.rmtree(vn_test_resampled_path, ignore_errors=True)
    shutil.rmtree(fl_train_resampled_path, ignore_errors=True)
    shutil.rmtree(fl_test_resampled_path, ignore_errors=True)
except OSError:
    print("Removal of the directory %s failed" % vn_train_resampled_path)
    print("Removal of the directory %s failed" % vn_test_resampled_path)
    print("Removal of the directory %s failed" % fl_train_resampled_path)
    print("Removal of the directory %s failed" % fl_test_resampled_path)
else:
    print("Successfully removed the directory %s" % vn_train_resampled_path)
    print("Successfully removed the directory %s" % vn_test_resampled_path)
    print("Successfully removed the directory %s" % fl_train_resampled_path)
    print("Successfully removed the directory %s" % fl_test_resampled_path)


try:
    os.mkdir(vn_train_resampled_path)
    os.mkdir(vn_test_resampled_path)
    os.mkdir(fl_train_resampled_path)
    os.mkdir(fl_test_resampled_path)
except OSError:
    print("Creation of the directory  failed")


resample(vn_train_path, vn_train_resampled_path, SR_16kHz)
resample(vn_test_path, vn_test_resampled_path, SR_16kHz)
resample(fl_train_path, fl_train_resampled_path, SR_16kHz)
resample(fl_test_path, fl_test_resampled_path, SR_16kHz)

