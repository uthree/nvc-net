import argparse
import os

import torch
import torchaudio
from model import VoiceConvertor
from torchaudio.functional import resample as resample

import numpy as np
import pyaudio

parser = argparse.ArgumentParser(description="Inference")

parser.add_argument('-d', '--device', default='cpu', choices=['cpu', 'cuda', 'mps'],
                    help="Device setting. Set this option to cuda if you need to use ROCm.")
parser.add_argument('-t', '--target', default='./target.wav',
                    help="Target voice")
parser.add_argument('-c', '--chunk', default=2048, type=int)
parser.add_argument('-b', '--buffer', default=8, type=int)
parser.add_argument('-fp16', default=False, type=bool)
parser.add_argument('-ic', '--inputchannels', default=1, type=int)
parser.add_argument('-oc', '--outputchannels', default=1, type=int)
parser.add_argument('-lc', '--loopbackchannels', default=1, type=int)
parser.add_argument('-i', '--input', default=0, type=int)
parser.add_argument('-o', '--output', default=0, type=int)
parser.add_argument('-l', '--loopback', default=-1, type=int)

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
spk = mean# + torch.randn(*logvar.shape, device=logvar.device) * torch.exp(logvar)

audio = pyaudio.PyAudio()
stream_input = audio.open(
        format=pyaudio.paInt16,
        rate=44100,
        channels=args.inputchannels,
        input_device_index=args.input,
        input=True)
stream_output = audio.open(
        format=pyaudio.paInt16,
        rate=44100, 
        channels=args.outputchannels,
        output_device_index=args.output,
        output=True)
stream_loopback = audio.open(
        format=pyaudio.paInt16,
        rate=44100, 
        channels=args.loopbackchannels,
        output_device_index=args.loopback,
        output=True) if args.loopback != -1 else None

print("Converting Voice...")


buffer = []
chunk = args.chunk
buffer_size = args.buffer
while True:
    data = stream_input.read(chunk, exception_on_overflow=False)
    data = np.frombuffer(data, dtype=np.int16)
    buffer.append(data)
    if len(buffer) > buffer_size:
        del buffer[0]
    else:
        continue
    data = np.concatenate(buffer, 0)
    data = data.astype(np.float32) / 32768
    data = torch.from_numpy(data).to(device)
    data = torch.unsqueeze(data, 0)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=args.fp16):
            data = resample(data, 44100, 22050)
            data = G(Ec(data), spk)
            data = resample(data, 22050, 44100)
    data = data[0]
    data = data.cpu().numpy()
    data = data * 32768
    data = data.astype(np.int16)
    s = (chunk * buffer_size) // 2 - (chunk // 2)
    e = (chunk * buffer_size) - s
    data = data[s:e]
    data = data.tobytes()
    stream_output.write(data)
    if stream_loopback is not None:
        stream_loopback.write(data)
