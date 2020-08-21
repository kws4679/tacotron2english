import IPython.display as ipd

import sys
sys.path.append('waveglow/')
import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
# from denoiser import Denoiser

if __name__ == '__main__':
    hparams = create_hparams()
    hparams.sampling_rate = 22050
    checkpoint_path = "tacotron2_statedict.pt"
    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    # _ = model.cuda().eval().half()
    _ = model.eval().half()

    waveglow = torch.load(waveglow_path)['model']
    # waveglow.cuda().eval().half()
    waveglow.eval().half()

    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)

    text = "Waveglow is really awesome!"
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    # sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).long()

    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
        ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)

        # audio_denoised = denoiser(audio, strength=0.01)[:, 0]
        # ipd.Audio(audio_denoised.cpu().numpy(), rate=hparams.sampling_rate)