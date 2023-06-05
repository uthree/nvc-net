import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from tqdm import tqdm

from model import VoiceConvertor, Discriminator, MelSpectrogramLoss
from dataset import WaveFileDirectory


def load_or_init_models(device=torch.device('cpu')):
    C = VoiceConvertor().to(device)
    D = Discriminator().to(device)
    if os.path.exists("convertor.pt"):
        C.load_state_dict(torch.load("convertor.pt", map_location=device))
    if os.path.exists("discriminator.pt"):
        D.load_state_dict(torch.load("discriminator.pt", map_location=device))
    return C, D


def save_models(C, D):
    torch.save(C.state_dict(), "convertor.pt")
    torch.save(D.state_dict(), "discriminator.pt")
    print("saved models")

parser = argparse.ArgumentParser(description="Train NVC-Net")

parser.add_argument('dataset_path')
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-e', '--epoch', default=60, type=int)
parser.add_argument('-b', '--batch', default=4, type=int)
parser.add_argument('-fp16', default=False, type=bool)
parser.add_argument('-m', '--maxdata', default=-1, type=int, help="max dataset size")
parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float)

args = parser.parse_args()
device = torch.device(args.device)

C, D = load_or_init_models(device)

Ec = C.content_encoder
Es = C.speaker_encoder
G = C.generator

weight_kl = 0.02
weight_con = 0# 10.0
weight_rec = 10.0
weight_mel = 1.0

ds = WaveFileDirectory(
        [args.dataset_path],
        length=32768,
        max_files=args.maxdata)

dl = torch.utils.data.DataLoader(ds, batch_size=args.batch*2, shuffle=True)

OptC = optim.Adam(C.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
OptD = optim.Adam(D.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))

mel_loss = MelSpectrogramLoss().to(device)
L1 = nn.L1Loss()
BCE = nn.BCEWithLogitsLoss()
scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)


for epoch in range(args.epoch):
    tqdm.write(f"Epoch #{epoch}")
    bar = tqdm(total=len(ds))
    for batch, wave in enumerate(dl):
        N = wave.shape[0]
        if N % 2 != 0:
            continue
        wave = wave.to(device)
        wave_src, wave_tgt = wave.chunk(2, dim=0)
        
        OptC.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            src_mean, src_logvar = Es(wave_src)
            z_src = src_mean + torch.exp(src_logvar) * torch.randn(*src_logvar.shape, device=src_logvar.device)
            tgt_mean, tgt_logvar = Es(wave_tgt)
            z_tgt = tgt_mean + torch.exp(tgt_logvar) * torch.randn(*tgt_logvar.shape, device=tgt_logvar.device)

            c = Ec(wave_src)
            wave_rec = G(c, z_src)

            loss_fm = D.feat_loss(wave_rec, wave_src)
            loss_mel = mel_loss(wave_rec, wave_src)
            loss_rec = loss_fm + weight_mel * loss_mel
            wave_fake = G(c, z_tgt)
            loss_adv = 0
            logits = D.logits(wave_fake)
            for logit in logits:
                loss_adv += BCE(logit, torch.zeros_like(logit)) / len(logits)
            loss_con = ((Ec(wave_fake) - c) ** 2).mean()
            loss_kl = (-1 - src_logvar + torch.exp(src_logvar) + src_mean ** 2).mean() +\
                    (-1 - tgt_logvar + torch.exp(tgt_logvar) + src_mean ** 2).mean()
            loss_C = loss_adv + loss_rec * weight_rec + weight_con * loss_con + weight_kl * loss_kl
        scaler.scale(loss_C).backward()
        torch.nn.utils.clip_grad_norm_(C.parameters(), 1.0, 2.0)
        scaler.step(OptC)
        
        OptD.zero_grad()
        wave_fake = wave_fake.detach()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            loss_D = 0
            logits = D.logits(wave_fake)
            for logit in logits:
                loss_D += BCE(logit, torch.ones_like(logit))  / len(logits)
            logits = D.logits(wave_src)
            for logit in logits:
                loss_D += BCE(logit, torch.zeros_like(logit))  / len(logits)
        scaler.scale(loss_D).backward()
        torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0, 2.0)
        scaler.step(OptD)

        scaler.update()

        if batch % 100 == 0:
            save_models(C, D)
        tqdm.write(f"Adv.: {loss_adv.item():.4f}, F.M.: {loss_fm.item():.4f}, Mel.: {loss_mel.item():.4f}, K.L.: {loss_kl.item():.4f}, Con. {loss_con.item():.4f}")
        bar.set_description(f"C: {loss_C.item():.4f}, D: {loss_D.item():.4f}")
        bar.update(N)


