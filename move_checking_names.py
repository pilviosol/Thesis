import pathlib
import shutil
"""
# features path
reducted_flutes = pathlib.Path("/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/FW_normalised_flute_0305_VALID/").iterdir()
reducted_strings = pathlib.Path("/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/FW_normalised_string_0305_VALID/").iterdir()
reducted_keyboards = pathlib.Path("/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/FW_normalised_keyboard_0305_VALID/").iterdir()
reducted_guitars = pathlib.Path("/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/FW_normalised_guitar_1805_VALID/").iterdir()
reducted_organs = pathlib.Path("/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/FW_normalised_organ_1805_VALID/").iterdir()

# WAV path
flutes_VALID = pathlib.Path("/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/flute_acoustic/").iterdir()
all_VALID = pathlib.Path("/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN/nsynth-train/audio/").iterdir()

strings_VALID = pathlib.Path("/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/string_acoustic/").iterdir()
keyboards_VALID = pathlib.Path("/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/keyboard_acoustic/").iterdir()
guitars_VALID = pathlib.Path("/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/guitar_acoustic/").iterdir()
organs_VALID = pathlib.Path("/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/organ_electronic/").iterdir()

# destination WAV path
new_flutes_VALID = pathlib.Path("/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/07062022/WAV_flute/")
new_strings_VALID = pathlib.Path("/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/07062022/WAV_string/")
new_keyboards_VALID = pathlib.Path("/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/07062022/WAV_keyboard/")
new_guitars_VALID = pathlib.Path("/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/07062022/WAV_guitar/")
new_organs_VALID = pathlib.Path("/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/07062022/WAV_organ/")
new_all_VALID = pathlib.Path("/nas/home/spol/Thesis/NSYNTH/NSYNTH_VALID_SUBSET/07062022/WAV_all/")
"""
reducted_flutes = pathlib.Path("/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/FW_normalised_flute_0605_TEST/").iterdir()
flutes_TEST = pathlib.Path("/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/flute_acoustic/").iterdir()
new_flutes_TEST = pathlib.Path("/nas/home/spol/Thesis/NSYNTH/NSYNTH_TEST_SUBSET/07062022/WAV_flute/")
all_VALID = pathlib.Path("/nas/home/spol/Thesis/NSYNTH/NSYNTH_TRAIN/nsynth-train/audio/").iterdir()


reducted_names = []

# instruments = [reducted_flutes, reducted_strings, reducted_keyboards, reducted_guitars, reducted_organs]

instruments = [reducted_flutes]
for instrument in instruments:
    for element in instrument:
        name = element.name[18:-16]
        print(name)
        reducted_names.append(name)


count = 0
for wav in all_VALID:
    wav_name = wav.name[0:-4]
    if wav_name in reducted_names:
        count += 1
        print(count)
        print(wav_name)
        shutil.copy(wav, new_flutes_TEST)


print('debuggo')





