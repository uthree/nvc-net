import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import VoiceConvertor, ASRModule
from dataset import ASRDataset


def load_or_init_models(device=torch.device('cpu')):
    C = VoiceConvertor().to(device)
    A = ASRModule().to(device)
    if os.path.exists("convertor.pt"):
        C.load_state_dict(torch.load("convertor.pt", map_location=device))
    for param in C.generator.parameters():
        param.requires_grad=False
    for param in C.speaker_encoder.parameters():
        param.requires_grad=False
    if os.path.exists("asr.pt"):
        A.load_state_dict(torch.load("asr.pt", map_location=device))
    return C, A


def save_models(C, A):
    torch.save(C.state_dict(), "convertor.pt")
    torch.save(A.state_dict(), "asr.pt")
    print("saved models")


parser = argparse.ArgumentParser(description="Train ASR")

parser.add_argument('dataset_path')
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-e', '--epoch', default=6000, type=int)
parser.add_argument('-b', '--batch', default=2, type=int)
parser.add_argument('-fp16', default=False, type=bool)
parser.add_argument('-m', '--maxdata', default=-1, type=int, help="max dataset size")
parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float)

args = parser.parse_args()
device = torch.device(args.device)

C, A = load_or_init_models(device)
Ec = C.content_encoder
Easr = A.encoder
Dasr = A.decoder

ds = ASRDataset(args.dataset_path)
dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True)

CSE = nn.CrossEntropyLoss(ignore_index=0)

scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
optimizer = optim.Adam(list(A.parameters()) + list(Ec.parameters()), lr=args.learning_rate)


for epoch in range(args.epoch):
    print(f"Epoch #{epoch}")
    bar = tqdm(total=len(ds))
    for batch, (wf, phonemes) in enumerate(dl):
        N = wf.shape[0]
        
        amp = torch.rand(N, 1).to(device) * 0.75 + 0.25
        wf = wf.to(device)
        wf = wf * amp
        phonemes = phonemes.to(device)

        sz = phonemes.shape[1]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(sz, device=device)

        dec_in = torch.cat([torch.zeros(N, 1, device=device, dtype=torch.long), phonemes[:, :-1]], dim=1)
        dec_tgt = phonemes

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            mem = Easr(Ec(wf))
            dec_out = Dasr(dec_in, mem, tgt_mask=tgt_mask, tgt_key_padding_mask=(dec_in != 0))
            loss_asr = CSE(
                    torch.flatten(dec_out, start_dim=0, end_dim=1),
                    torch.flatten(dec_tgt, start_dim=0, end_dim=1))
            loss = loss_asr
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bar.set_description(desc=f"ASR Loss: {loss_asr.item():.6f}")
        bar.update(N)
        if batch % 100 == 0:
            save_models(C, A)
