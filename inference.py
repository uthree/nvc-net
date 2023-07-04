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
parser.add_argument('-t', '--target', default='target.wav',
                    help="Target voice")

args = parser.parse_args()

device = torch.device(args.device)

C = VoiceConvertor().to(device)
C.load_state_dict(torch.load("./convertor.pt", map_location=device))
Es = C.speaker_encoder
Ec = C.content_encoder
G = C.generator

if args.target == 'random':
    print("Sampling from gaussian deviation...")
    spk = torch.randn(1, 128, 1).to(device)
else:
    print("Encoding target speaker...")

    wf, sr = torchaudio.load(args.target)
    wf = wf.to(device)
    wf = resample(wf, sr, 22050)

    mean, logvar = Es(wf)
    spk = mean + torch.exp(logvar) * torch.randn(*logvar.shape, device=logvar.device)

if not os.path.exists(args.output):
    os.mkdir(args.output)

plt.imshow(spk.reshape(16, 8).detach().cpu().numpy())
plt.savefig(os.path.join(args.output , f"speaker.png"))


l2_dists = []
for i, fname in enumerate(os.listdir(args.input)):
    print(f"Converting {fname}")
    with torch.no_grad():
        wf, sr = torchaudio.load(os.path.join(args.input, fname), normalize=True)
        wf = resample(wf, sr, 22050)
        
        z = Ec(wf)
        plt.imshow(F.interpolate(z.unsqueeze(1), size=(64, z.shape[2])).squeeze(1).squeeze(0).cpu())
        plt.savefig(os.path.join(args.output , f"{i}.png"))
        wf = G(z, spk)
        out_spk, _ = Es(wf)
        spk_l2_dist = ((out_spk - spk) ** 2).mean()
        print(f"Speaker Distance: {spk_l2_dist.item()}")
        l2_dists.append(spk_l2_dist)

        wf = resample(wf, 22050, sr)
        out_path = os.path.join(args.output, f"output_{fname}_{i}.wav")
        torchaudio.save(out_path, src=wf, sample_rate=sr)

l2_dists_mean = 0
for d in l2_dists:
    l2_dists_mean += d / len(l2_dists)

print(f"Mean of distance: {l2_dists_mean}")
