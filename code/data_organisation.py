import os
import shutil 
import argparse

parser = argparse.ArgumentParser(description="Preprocess RAVDESS data for IFN")
parser.add_argument("--base_path", help="Path of the original RAVDESS data", type=str, required=True, default="../data/Audio_Speech_Actors_01-24/")
parser.add_argument("--processed_path", help="Path to put the processed files in", type=str, required=True, default="../data/processed/")
args = parser.parse_args()

BASE_PATH = args.base_path
all_files = os.listdir(BASE_PATH)

PROCESSED_PATH = args.processed_path


if os.path.exists(PROCESSED_PATH) == False:
    os.mkdir(PROCESSED_PATH)
    print('Created directory ', PROCESSED_PATH)


# Iterating through each speaker to create the reference audio list

if os.path.exists(PROCESSED_PATH+'reference/') == False:
    os.mkdir(PROCESSED_PATH+'reference/')

if os.path.exists(PROCESSED_PATH+'train2/') == False:
    os.mkdir(PROCESSED_PATH+'train2/')

reference_audio_list = []
for speaker in all_files:
    speaker_files = os.listdir(BASE_PATH+speaker)
    for speaker_file in speaker_files:
        temp = speaker_file.split('.')
        split_dest = ""
        temp2 = temp[0].split('-')
        speaker_no = int(temp2[-1])
        split_name = temp[0][:-3]
        if split_name == '03-01-01-01-01-01':
            split_dest = 'reference/'
            reference_audio_list.append(speaker_file)
            shutil.copyfile(BASE_PATH+speaker+"/"+speaker_file, PROCESSED_PATH+split_dest+speaker_file)
        else:
            split_dest = 'train2/'
            shutil.copyfile(BASE_PATH+speaker+"/"+speaker_file, PROCESSED_PATH+split_dest+speaker_file)            

