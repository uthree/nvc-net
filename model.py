import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import torchaudio


def padding_wave(x, period=256):
    if x.shape[1] % period != 0:
        pad_len = period - (x.shape[1] % period)
        x = torch.cat([x, torch.zeros(x.shape[0], pad_len, device=x.device)], dim=1)
    return x


class SpeakerEncoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.to_mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=22050,
                n_mels=80,
                n_fft=1024)
        self.layers = nn.Sequential(
                nn.Conv1d(80, 32, 3, 1, 1),
                nn.LeakyReLU(0.2),
                nn.Conv1d(32, 64, 3, 1, 1),
                nn.LeakyReLU(0.2),
                nn.AvgPool1d(2),
                nn.Conv1d(64, 128, 3, 1, 1),
                nn.LeakyReLU(0.2),
                nn.AvgPool1d(2),
                nn.Conv1d(128, 256, 3, 1, 1),
                nn.LeakyReLU(0.2),
                nn.AvgPool1d(2),
                nn.Conv1d(256, 512, 3, 1, 1),
                nn.LeakyReLU(0.2),
                nn.AvgPool1d(2),
                nn.Conv1d(512, 512, 3, 1, 1),
                nn.LeakyReLU(0.2),
                nn.AvgPool1d(2))
        self.output_layer = nn.Conv1d(512, 256, 1, 1, 0)

    def forward(self, x):
        x = self.to_mel(x)
        x = self.layers(x)
        x = x.mean(dim=2, keepdim=True)
        x = self.output_layer(x)
        mean, logvar = x.chunk(2, dim=1)
        return mean, logvar



class ContentEncoderResBlock(nn.Module):
    def __init__(self, channels, dilation=3):
        super().__init__()
        self.res_conv = weight_norm(nn.Conv1d(channels, channels, 1, 1, 0))
        self.input_conv = weight_norm(nn.Conv1d(channels, channels * 2, 3, 1, padding='same', dilation=dilation))
        self.channels = channels 
        self.output_conv = weight_norm(nn.Conv1d(channels, channels, 1, 1, 0))
    
    def forward(self, x):
        res = self.res_conv(x)
        x = self.input_conv(x)
        x = torch.sigmoid(x[:, self.channels:]) * torch.tanh(x[:, :self.channels])
        x = self.output_conv(x)
        return x + res


class ContentEncoderResStack(nn.Module):
    def __init__(self, channels, num_blocks=4, dilation=3):
        super().__init__()
        self.layers = nn.ModuleList(
                [ContentEncoderResBlock(channels, dilation) for _ in range(num_blocks)])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ContentEncoder(nn.Module):
    def __init__(self, channels=[32, 64, 128, 256], downsample_rates=[2, 2, 8, 8]):
        super().__init__()
        self.res_blocks = nn.ModuleList([])
        self.downsamples = nn.ModuleList([])
        for i, c in enumerate(channels):
            self.res_blocks.append(ContentEncoderResStack(c, 3 ** i))
        for c1, c2, rate in zip(channels, channels[1:] + [512], downsample_rates):
            self.downsamples.append(
                    weight_norm(
                        nn.Conv1d(c1, c2, rate*2, rate, rate//2)))
        self.input_layer = weight_norm(nn.Conv1d(1, 32, 7, 1, 3))
        self.output_layers = nn.Sequential(
                weight_norm(nn.Conv1d(512, 4, 7, 1, 3)),
                nn.GELU(),
                weight_norm(nn.Conv1d(4, 4, 7, 1, 3)))

    def forward(self, x):
        x = padding_wave(x)
        x = x.unsqueeze(1)
        x = self.input_layer(x)
        x = F.gelu(x)
        for r, d in zip(self.res_blocks, self.downsamples):
            x = r(x)
            x = F.gelu(x)
            x = d(x)
        x = self.output_layers(x)
        return x


class GeneratorResBlock(nn.Module):
    def __init__(self, channels, d_spk=128, dilation=3):
        super().__init__()
        self.res_conv = weight_norm(nn.Conv1d(channels, channels, 1, 1, 0))
        self.input_conv = weight_norm(nn.Conv1d(channels, channels * 2, 3, 1, padding='same', dilation=dilation))
        self.spk_conv = weight_norm(nn.Conv1d(d_spk, channels * 2, 1, 1, 0))
        self.channels = channels 
        self.output_conv = weight_norm(nn.Conv1d(channels, channels, 1, 1, 0))
    
    def forward(self, x, spk):
        res = self.res_conv(x)
        x = self.input_conv(x) + self.spk_conv(spk)
        x = torch.sigmoid(x[:, self.channels:]) * torch.tanh(x[:, :self.channels])
        x = self.output_conv(x)
        return x + res


class GeneratorResStack(nn.Module):
    def __init__(self, channels, d_spk=128, num_blocks=4, dilation=3):
        super().__init__()
        self.layers = nn.ModuleList(
                [GeneratorResBlock(channels, d_spk, dilation) for _ in range(num_blocks)])

    def forward(self, x, spk):
        for layer in self.layers:
            x = layer(x, spk)
        return x


class Generator(nn.Module):
    def __init__(self, channels=[256, 128, 64, 32], upsample_rates=[2, 2, 8, 8]):
        super().__init__()
        self.res_blocks = nn.ModuleList([])
        self.upsamples = nn.ModuleList([])
        for i, c in enumerate(channels):
            self.res_blocks.append(GeneratorResStack(c, 128, 3 ** i))
        for c1, c2, rate in zip(channels, [512] + channels[:-1], upsample_rates):
            self.upsamples.append(
                    weight_norm(
                        nn.ConvTranspose1d(c2, c1, rate*2, rate, rate//2)))
        self.input_layers = nn.Sequential(
                weight_norm(
                    nn.Conv1d(4, 512, 7, 1, 3)),
                nn.GELU(),
                weight_norm(
                    nn.Conv1d(512, 512, 7, 1, 3)),
                nn.GELU())
        self.output_layer = weight_norm(
                nn.Conv1d(32, 1, 7, 1, 3))

    def forward(self, x, spk):
        x = self.input_layers(x)
        for r, u in zip(self.res_blocks, self.upsamples):
            x = F.gelu(x)
            x = u(x)
            x = r(x, spk)
        x = self.output_layer(x)
        x = torch.tanh(x)
        x = x.squeeze(1)
        return x


class VoiceConvertor(nn.Module):
    def __init__(self):
        super().__init__()
        self.content_encoder = ContentEncoder()
        self.speaker_encoder = SpeakerEncoder()
        self.generator = Generator()

class ScaleDiscriminator(nn.Module):
    def __init__(
            self,
            segment_size=16,
            channels=[64, 64, 64],
            norm_type='spectral',
            kernel_size=11,
            strides=[1, 1, 1],
            dropout_rate=0.1,
            groups=[],
            pool = 1
            ):
        super().__init__()
        self.pool = torch.nn.AvgPool1d(pool)
        self.segment_size = segment_size
        if norm_type == 'weight':
            norm_f = nn.utils.weight_norm
        elif norm_type == 'spectral':
            norm_f = nn.utils.spectral_norm
        else:
            raise f"Normalizing type {norm_type} is not supported."
        self.layers = nn.ModuleList([])
        self.outputs = nn.ModuleList([])

        self.input_layer = norm_f(nn.Conv1d(segment_size, channels[0], 1, 1, 0))
        for i in range(len(channels)-1):
            if i == 0:
                k = 15
            else:
                k = kernel_size
            self.layers.append(
                    norm_f(
                        nn.Conv1d(channels[i], channels[i+1], k, strides[i], 0, groups=groups[i])))
            self.outputs.append(
                    norm_f(
                        nn.Conv1d(channels[i+1], 1, 1, 1, 0)))

    def forward(self, x):
        # Padding
        if x.shape[1] % self.segment_size != 0:
            pad_len = self.segment_size - (x.shape[1] % self.segment_size)
            x = torch.cat([x, torch.zeros(x.shape[0], pad_len, device=x.device)], dim=1)
        x = x.view(x.shape[0], self.segment_size, -1)
        x = self.pool(x)
        x = self.input_layer(x)
        logits = []
        for layer, output in zip(self.layers, self.outputs):
            x = layer(x)
            x = F.leaky_relu(x, 0.2)
            logits.append(output(x))
        return logits

    def feat(self, x):
        # Padding
        if x.shape[1] % self.segment_size != 0:
            pad_len = self.segment_size - (x.shape[1] % self.segment_size)
            x = torch.cat([x, torch.zeros(x.shape[0], pad_len, device=x.device)], dim=1)
        x = x.view(x.shape[0], self.segment_size, -1)
        x = self.pool(x)
        x = self.input_layer(x)
        feats = [x]
        for layer in self.layers:
            x = layer(x)
            x = F.leaky_relu(x, 0.2)
            feats.append(x)
        return feats


class MultiScaleDiscriminator(nn.Module):
    def __init__(
            self,
            segments=[1, 1, 1],
            channels=[32, 64, 128, 256, 512, 1024],
            kernel_sizes=[15, 41, 41, 41, 41, 41],
            strides=[1, 2, 4, 4, 4, 4],
            groups=[1, 1, 2, 2, 4, 4],
            pools=[1, 2, 4]
            ):
        super().__init__()
        self.sub_discriminators = nn.ModuleList([])
        for i, (k, sg, p) in enumerate(zip(kernel_sizes, segments, pools)):
            if i == 0:
                n = 'spectral'
            else:
                n = 'weight'
            self.sub_discriminators.append(
                    ScaleDiscriminator(sg, channels, n, k, strides, groups=groups, pool=p))

    def forward(self, x):
        logits = []
        for sd in self.sub_discriminators:
            logits += sd(x)
        return logits

    def feat(self, x):
        feats = []
        for sd in self.sub_discriminators:
            feats = feats + sd.feat(x)
        return feats


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.MSD = MultiScaleDiscriminator()

    def logits(self, x):
        return self.MSD(x)

    def feat_loss(self, fake, real):
        with torch.no_grad():
            real_feat = self.MSD.feat(real)
        fake_feat = self.MSD.feat(fake)
        loss = 0
        for r, f in zip(real_feat, fake_feat):
            loss = loss + F.l1_loss(f, r) / len(real_feat)
        return loss


class MelSpectrogramLoss(nn.Module):
    def __init__(self, sample_rate=22050, n_ffts=[512, 1024, 2048], n_mels=80, normalized=False):
        super().__init__()
        self.to_mels = nn.ModuleList([])
        for n_fft in n_ffts:
            self.to_mels.append(torchaudio.transforms.MelSpectrogram(sample_rate,
                                                                n_mels=n_mels,
                                                                n_fft=n_fft,
                                                                normalized=normalized,
                                                                hop_length=256))

    def forward(self, fake, real):
        loss = 0
        for to_mel in self.to_mels:
            to_mel = to_mel.to(real.device)
            with torch.no_grad():
                real_mel = to_mel(real)
            loss += F.l1_loss(to_mel(fake), real_mel).mean() / len(self.to_mels)
        return loss
