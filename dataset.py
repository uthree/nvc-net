import torch
import torchaudio
import glob
from tqdm import tqdm
import os
import random

from janome.tokenizer import Tokenizer
KATAKANA_LETTER_SMALL_A = 12449


class PhoneticTokenizer:
    def __init__(self):
        self.janome_tokenizer = Tokenizer()

    def tokenize(self, s):
        phonetic = ""
        tokens = self.janome_tokenizer.tokenize(s)
        for token in tokens:
            phonetic = phonetic + token.phonetic
        phonetic = list(phonetic)
        return [max(ord(p) - KATAKANA_LETTER_SMALL_A, 0) + 2 for p in phonetic]


class WaveFileDirectory(torch.utils.data.Dataset):
    def __init__(self, source_dir_paths=[], length=65536, max_files=-1, sampling_rate=44100, shuffle_paths=True):
        super().__init__()
        print("Loading Data")
        self.path_list = []
        self.data = []
        formats = ["mp3", "wav", "ogg"]
        print("Getting paths")
        for dir_path in source_dir_paths:
            for fmt in formats:
                self.path_list += glob.glob(os.path.join(dir_path, f"**/*.{fmt}"), recursive=True)
        if shuffle_paths:
            random.shuffle(self.path_list)
        if max_files != -1:
            self.path_list = self.path_list[:max_files]
        print("Chunking")
        for path in tqdm(self.path_list):
            tqdm.write(path)
            wf, sr = torchaudio.load(path) # wf.max() = 1 wf.min() = -1
            # Resample
            wf = torchaudio.functional.resample(wf, sr, sampling_rate)
            # Chunk
            waves = torch.split(wf, length, dim=1)
            tqdm.write(f"    Loading {len(waves)} data...")
            for w in waves:
                if w.shape[1] == length:
                    self.data.append(w[0])
        self.length = length
        print(f"Loaded total {len(self.data)} data.")

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)



class ASRDataset(torch.utils.data.Dataset):
    def __init__(self, source_dir_path, max_tokens=30, max_len=262144, pad=0, eos=1, sampling_rate=44100):
        super().__init__()
        self.audio_path_list = []
        self.text_path_list = []
        self.waves = []
        self.texts = []
        tokenizer = PhoneticTokenizer()

        formats = ["mp3", "wav", "ogg"]
        print("Getting paths")
        for fmt in formats:
            self.audio_path_list += glob.glob(os.path.join(source_dir_path, f"**/*.{fmt}"), recursive=True)

        for ap in self.audio_path_list:
            noext = os.path.splitext(os.path.basename(ap))[0]
            dirname = os.path.dirname(ap)
            self.text_path_list.append(os.path.join(dirname, f"{noext}.txt"))

        print("Loading")
        for ap, tp in tqdm(zip(self.audio_path_list, self.text_path_list)):
            wf, sr = torchaudio.load(ap)
            # Resample
            wf = torchaudio.functional.resample(wf, sr, sampling_rate)
            # Crop
            if wf.shape[1] > max_len:
                wf = wf[:, :max_len]
            else:
                wf = torch.cat([wf, torch.zeros(1, max_len - wf.shape[1])], dim=1)

            # Load text
            with open(tp) as f:
                p = tokenizer.tokenize(f.read())
            p.append(eos)
            while len(p) < max_tokens:
                p.append(pad)

            if len(p) > max_tokens:
                p = p[:max_tokens]
            p = torch.LongTensor(p)

            self.waves.append(wf[0])
            self.texts.append(p)

    def __getitem__(self, index):
        return self.waves[index], self.texts[index]

    def __len__(self):
        return len(self.waves)
