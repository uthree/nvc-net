import argparse
import os
import random

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


def random_flip(x):
    scale = torch.randint(low=0, high=1, size=(x.shape[0], 1), device=x.device) * 2 - 1
    return x * scale

parser = argparse.ArgumentParser(description="Train NVC-Net")

parser.add_argument('dataset_path')
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-e', '--epoch', default=6000, type=int)
parser.add_argument('-b', '--batch', default=8, type=int)
parser.add_argument('-fp16', default=False, type=bool)
parser.add_argument('-m', '--maxdata', default=-1, type=int, help="max dataset size")
parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float)
parser.add_argument('--freeze-encoder', default=False, type=bool)
parser.add_argument('--save-frequency', default=100, type=int)
parser.add_argument('-gacc', '--gradient-accumulation', type=int, default=1)

args = parser.parse_args()
device = torch.device(args.device)

C, D = load_or_init_models(device)

Ec = C.content_encoder
Es = C.speaker_encoder
G = C.generator

weight_kl = 0.02
weight_con = 10.0
weight_rec = 10.0
weight_mel = 4.5

grad_acc = args.gradient_accumulation

ds = WaveFileDirectory(
        [args.dataset_path],
        length=32768,
        max_files=args.maxdata)

dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True)

OptC = optim.Adam(C.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
OptD = optim.Adam(D.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))

mel_loss = MelSpectrogramLoss().to(device)
BCE = nn.BCEWithLogitsLoss()
scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

C.train()
D.train()
if args.freeze_encoder:
    for param in Ec.parameters():
        param.requires_grad=False

for epoch in range(args.epoch):
    tqdm.write(f"Epoch #{epoch}")
    bar = tqdm(total=len(ds))
    for batch, wave in enumerate(dl):
        N = wave.shape[0]
        wave = wave.to(device)
        # Data Augmentation
        wave = wave * (torch.rand(N, 1,device=device) * 0.75 + 0.25)
        wave = random_flip(wave)
        wave_src = wave
        wave_tgt = torch.roll(wave, dims=0, shifts=1)

        with torch.cuda.amp.autocast(enabled=args.fp16):
            src_mean, src_logvar = Es(wave_src)
            z_src = src_mean + torch.exp(src_logvar) * torch.rand_like(src_logvar)
            z_tgt = torch.roll(z_src, dims=0, shifts=1)

            c = Ec(wave_src)
            wave_rec = G(c, z_src)

            loss_fm = D.feat_loss(wave_rec, random_flip(wave_src))
            loss_mel = mel_loss(wave_rec, wave_src)
            loss_rec = loss_fm + weight_mel * loss_mel
            wave_fake = G(c, z_tgt)
            loss_adv = 0
            logits = D.logits(wave_fake)
            for logit in logits:
                loss_adv += BCE(logit, torch.zeros_like(logit)) / len(logits)
            
            loss_con = ((Ec(wave_fake) - c) ** 2).mean()

            loss_kl = (-1 - src_logvar + torch.exp(src_logvar) + src_mean ** 2).mean()
            
            loss_C = loss_adv + loss_rec * weight_rec + weight_con * loss_con + weight_kl * loss_kl 
        scaler.scale(loss_C).backward()
        torch.nn.utils.clip_grad_norm_(C.parameters(), 1.0, 2.0)
        if torch.any(torch.isnan(loss_C)):
            exit()
        
        if batch % grad_acc == 0:
            scaler.step(OptC)
            OptC.zero_grad()
        
        OptD.zero_grad()
        wave_fake = random_flip(wave_fake.detach())
        with torch.cuda.amp.autocast(enabled=args.fp16):
            loss_D = 0
            logits = D.logits(wave_fake)
            for logit in logits:
                loss_D += BCE(logit, torch.ones_like(logit)) / len(logits)
            logits = D.logits(wave_src)
            for logit in logits:
                loss_D += BCE(logit, torch.zeros_like(logit)) / len(logits)
            loss_D = loss_D / 2
        scaler.scale(loss_D).backward()
        torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0, 2.0)
        scaler.step(OptD)

        scaler.update()

        if batch % args.save_frequency == 0:
            save_models(C, D)
        tqdm.write(f"Adv.: {loss_adv.item():.4f}, F.M.: {loss_fm.item():.4f}, Mel.: {loss_mel.item():.4f}, K.L.: {loss_kl.item():.4f}, Con.: {loss_con.item():.4f}")
        bar.set_description(f"C: {loss_C.item():.4f}, D: {loss_D.item():.4f}")
        bar.update(N)


