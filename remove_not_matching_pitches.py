from functions import remove_files_if_pitch_not_matching


path = '/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/matching_organ_TEST/'
"""elimination_list = ['070',
                    '071',
                    '072',
                    '073',
                    '074',
                    '075',
                    '076',
                    '077',
                    '078',
                    '079']"""
elimination_list = ['069', '070', '071', '072', '074', '076', '079', '080', '082', '088', '089', '090', '091', '092',
                    '094', '095', '097', '098', '099']

remove_files_if_pitch_not_matching(path, elimination_list)
