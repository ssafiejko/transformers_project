import os
import random
import shutil
from scipy.io import wavfile
import numpy as np
from glob import glob
import warnings
warnings.filterwarnings("ignore", category=UserWarning, append=True)

def split_wav_to_seconds(input_path, output_dir, file_number, filename_list):
    os.makedirs(output_dir, exist_ok=True)
    sample_rate, data = wavfile.read(input_path)
    total_secs = data.shape[0] // sample_rate
    
    print(f'[INFO] Splitting into {total_secs} chunks of {sample_rate} samples each.')
    for i in range(total_secs):
        chunk = data[i * sample_rate: (i + 1) * sample_rate]
        g_id = f"g{random.getrandbits(28):07x}"
        while g_id in filename_list:
            g_id = f"g{random.getrandbits(28):07x}" # no repetitions
        filename_list.append(g_id)
        out_filename = f"{g_id}_nohash_{file_number}.wav"
        out_path = os.path.join(output_dir, out_filename)
        wavfile.write(out_path, sample_rate, chunk)


def prepare_lists(output_dir, p_valid, p_test, val_file='validation_list.txt', test_file='testing_list.txt', category_name='silence'):
    """Distribute files into validation and testing lists proportionally per original file."""
    all_files = sorted(glob(os.path.join(output_dir, '*.wav')))
    
    print(f'[INFO] Deleting all {category_name}/* entries from "{val_file}" and "{test_file}"...')

    # Clean up old entries
    for txt in [val_file, test_file]:
        if os.path.exists(txt):
            with open(txt, 'r') as f:
                lines = [line for line in f if not line.startswith(f'{category_name}/')]
            with open(txt, 'w') as f:
                f.writelines(lines)
    
    # Group by original file number (order is random)
    grouped = {}
    for f in all_files:
        basename = os.path.basename(f)
        parts = basename.split('_nohash_')
        if len(parts) != 2:
            continue
        group_key = parts[1].split('_')[0]  # file_number
        grouped.setdefault(group_key, []).append(f)

    # Sampling
    val_lines, test_lines = [], []
    for group in grouped.values():
        random.shuffle(group)
        n = len(group)
        n_val = int(p_valid * n)
        n_test = int(p_test * n)

        val = group[:n_val]
        test = group[n_val:n_val + n_test]

        val_lines.extend([f"{category_name}/{os.path.basename(f)}\n" for f in val])
        test_lines.extend([f"{category_name}/{os.path.basename(f)}\n" for f in test])
    
    print(f'[INFO] Found {len(val_lines)} validation and {len(test_lines)} testing files.')
    print(f'[INFO] Writing to "{val_file}" and "{test_file}"...')
    # Append to files
    with open(val_file, 'a') as f:
        f.writelines(val_lines)
    with open(test_file, 'a') as f:
        f.writelines(test_lines)
    print(f'[INFO] Success!')
    print('---------------------------------------------------------')
    print(f'[INFO] Train files: {len(all_files) - len(val_lines) - len(test_lines)}')
    print(f'[INFO] Validation files: {len(val_lines)}') 
    print(f'[INFO] Testing files: {len(test_lines)}')
    print('---------------------------------------------------------')
    
def split_noise(category_name,
                p_valid,
                p_test,
                input_folder="./tensorflow-speech-recognition-challenge/train/audio/_background_noise_",
                val_file="./tensorflow-speech-recognition-challenge/train/validation_list.txt",
                test_file="./tensorflow-speech-recognition-challenge/train/testing_list.txt"):
    """Main function to process all WAV files."""
    output_folder = os.path.join("./tensorflow-speech-recognition-challenge/train/audio", category_name)
    if os.path.exists(output_folder):
        print(f'Directory with "{category_name}" already exists. Deleting...')
        shutil.rmtree(output_folder)
    
    input_files = sorted(glob(os.path.join(input_folder, '*.wav')))
    print(f'[INFO] Found {len(input_files)} files in input folder.')
    filename_list = []  # To track unique file names
    for idx, filepath in enumerate(input_files):
        print(f'[INFO] {os.path.basename(filepath)}...')
        split_wav_to_seconds(filepath, output_folder, idx, filename_list)
    print('--------------------------------------------------------')
    prepare_lists(output_folder, p_valid, p_test, val_file, test_file, category_name)