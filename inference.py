import argparse
import os

import torch
import torchaudio
from model import VoiceConvertor
from torchaudio.functional import resample as resample

import matplotlib.pyplot as plt
import torch.nn.functional as F


parser = argparse.ArgumentParser(description="Inference")

parser.add_argument('-d', '--device', default='cpu', choices=['cpu', 'cuda', 'mps'],
                    help="Device setting. Set this option to cuda if you need to use ROCm.")
parser.add_argument('-i', '--input', default='./inputs',
                    help="Input directory")
parser.add_argument('-o', '--output', default='./outputs',
                    help="Output directory")
parser.add_argument('-t', '--target', default='./target.wav',
                    help="Target voice")

args = parser.parse_args()

device = torch.device(args.device)

C = VoiceConvertor().to(device)
C.load_state_dict(torch.load("./convertor.pt", map_location=device))
Es = C.speaker_encoder
Ec = C.content_encoder
G = C.generator

print("Encoding target speaker...")

wf, sr = torchaudio.load(args.target)
wf = wf.to(device)
wf = resample(wf, sr, 22050)

mean, logvar = Es(wf)
spk = mean #+ torch.randn(*logvar.shape, device=logvar.device) * torch.exp(logvar)

if not os.path.exists(args.output):
    os.mkdir(args.output)

for i, fname in enumerate(os.listdir(args.input)):
    print(f"Converting {fname}")
    with torch.no_grad():
        wf, sr = torchaudio.load(os.path.join(args.input, fname))
        wf = resample(wf, sr, 22050)
        
        z = Ec(wf)
        plt.imshow(F.interpolate(z.unsqueeze(1), size=(64, z.shape[2])).squeeze(1).squeeze(0).cpu())
        plt.savefig(os.path.join(args.output , f"{i}.png"))
        wf = G(z, spk)

        wf = resample(wf, 22050, sr)
        out_path = os.path.join(args.output, f"output_{fname}_{i}.wav")
        torchaudio.save(out_path, src=wf, sample_rate=sr)
