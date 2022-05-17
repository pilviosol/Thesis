from functions import remove_files_if_pitch_not_matching


path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN_SUBSET/reducted_organ/'
elimination_list = ['009', '010', '020', '030', '040', '050', '060']

remove_files_if_pitch_not_matching(path, elimination_list)
