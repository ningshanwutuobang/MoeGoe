
from scipy.io.wavfile import write
from mel_processing import spectrogram_torch
from text import text_to_sequence, _clean_text
from models import SynthesizerTrn
import utils
import commons
import sys
import re
from torch import no_grad, LongTensor
import logging
import numpy as np

logging.getLogger('numba').setLevel(logging.WARNING)


def ex_print(text, escape=False):
    if escape:
        print(text.encode('unicode_escape').decode())
    else:
        print(text)


def get_text(text, hps, cleaned=False):
    if cleaned:
        text_norm = text_to_sequence(text, hps.symbols, [])
    else:
        text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm


def ask_if_continue():
    while True:
        answer = input('Continue? (y/n): ')
        if answer == 'y':
            break
        elif answer == 'n':
            sys.exit(0)


def print_speakers(speakers, escape=False):
    if len(speakers) > 100:
        return
    print('ID\tSpeaker')
    for id, name in enumerate(speakers):
        ex_print(str(id) + '\t' + name, escape)


def get_speaker_id(message):
    speaker_id = input(message)
    try:
        speaker_id = int(speaker_id)
    except:
        print(str(speaker_id) + ' is not a valid ID!')
        sys.exit(1)
    return speaker_id


def get_label_value(text, label, default, warning_name='value'):
    value = re.search(rf'\[{label}=(.+?)\]', text)
    if value:
        try:
            text = re.sub(rf'\[{label}=(.+?)\]', '', text, 1)
            value = float(value.group(1))
        except:
            print(f'Invalid {warning_name}!')
            sys.exit(1)
    else:
        value = default
    return value, text


def get_label(text, label):
    if f'[{label}]' in text:
        return True, text.replace(f'[{label}]', '')
    else:
        return False, text


if __name__ == '__main__':
    if '--escape' in sys.argv:
        escape = True
    else:
        escape = False

    model = "../MoeGoe/models/402_epochs.pth" # input('Path of a VITS model: ')
    config = "../MoeGoe/models/config.json" # input('Path of a config file: ')

    hps_ms = utils.get_hparams_from_file(config)
    n_speakers = hps_ms.data.n_speakers if 'n_speakers' in hps_ms.data.keys() else 0
    n_symbols = len(hps_ms.symbols) if 'symbols' in hps_ms.keys() else 0
    speakers = hps_ms.speakers if 'speakers' in hps_ms.keys() else ['0']
    use_f0 = hps_ms.data.use_f0 if 'use_f0' in hps_ms.data.keys() else False
    emotion_embedding = hps_ms.data.emotion_embedding if 'emotion_embedding' in hps_ms.data.keys() else False

    net_g_ms = SynthesizerTrn(
        n_symbols,
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=n_speakers,
        emotion_embedding=emotion_embedding,
        **hps_ms.model)
    _ = net_g_ms.eval()
    utils.load_checkpoint(model, net_g_ms)

    text = "[ZH]你好[ZH]"
    length_scale, text = get_label_value(
                        text, 'LENGTH', 1, 'length scale')
    noise_scale, text = get_label_value(
        text, 'NOISE', 0.667, 'noise scale')
    noise_scale_w, text = get_label_value(
        text, 'NOISEW', 0.8, 'deviation of noise')
    cleaned, text = get_label(text, 'CLEANED')

    print(text)

    stn_tst = get_text(text, hps_ms, cleaned=cleaned)

    # print_speakers(speakers, escape)
    speaker_id = 3 # get_speaker_id('Speaker ID: ')
    # out_path = input('Path to save: ')

    audio_path = "a.mp3" # input('Path of an audio file to convert:\n')
    # print_speakers(speakers)


    with no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = LongTensor([stn_tst.size(0)])
        sid = LongTensor([speaker_id])
        audio = utils.load_audio_to_torch(
        audio_path, hps_ms.data.sampling_rate)
        y = audio.unsqueeze(0)

        spec = spectrogram_torch(y, hps_ms.data.filter_length,
                                    hps_ms.data.sampling_rate, hps_ms.data.hop_length, hps_ms.data.win_length,
                                    center=False)

        spec_lengths = LongTensor([spec.size(-1)])
        attn = net_g_ms.text_aligned(x_tst, x_tst_lengths, spec, spec_lengths, sid=sid).data.cpu().float().numpy()[0,0]

    x = stn_tst.data.cpu().float().numpy()
    y = np.sum(attn, axis=0)

    print(_clean_text(text, hps_ms.data.text_cleaners))
    print(np.stack([x, y], axis=0))
