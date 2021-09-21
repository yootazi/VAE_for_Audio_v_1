import librosa
import os
import pickle
import numpy as np
import soundfile as sf


class SoundGenerator:
    """SoundGenerator is responsible for generating audios from
    spectrograms.
    """

    def __init__(self, vae, hop_length):
        self.vae = vae
        self.hop_length = hop_length
        self._min_max_normaliser = MinMaxNormaliser(0, 1)        # private attribute

    def generate(self, spectrograms, min_max_values):
        generated_spectrograms, latent_representations = \
            self.vae.reconstruct(spectrograms)
        signals = self.convert_spectrograms_to_audio(generated_spectrograms, min_max_values)
        return signals, latent_representations

    def convert_spectrograms_to_audio(self, spectrograms, min_max_values):
        signals = []
        for spectrogram, min_max_value in zip(spectrograms, min_max_values):
            # reshape the log spectrogram
            log_spectrogram = spectrogram[:, :, 0]
            # apply denormalisation
            denorm_log_spec = self._min_max_normaliser.denormalise(
                log_spectrogram, min_max_value["min"], min_max_value["max"])
            # log spectrogram -> spectrogram
            spec = librosa.db_to_amplitude(denorm_log_spec)
            # apply Griffin-Lim
            signal = librosa.istft(spec, hop_length=self.hop_length)        # applying Griffin-Lim (inversed short term furier transform -> STFT) to spectrogram to get audio
            # append signal to "signals"
            signals.append(signal)
        return signals
        
        

HOP_LENGTH = 256
SAVE_DIR_ORIGINAL = "/content/gdrive/MyDrive/ai_music_projects/VAE-for-Audio-v.1/samples/original/"          # sampled spectrograms - we should have 5 original audio files
SAVE_DIR_GENERATED = "/content/gdrive/MyDrive/ai_music_projects/VAE-for-Audio-v.1/samples/generated/"        # we should have 5 audio files created from 5 spectrograms that has been sampled, saved in this folder at the end
MIN_MAX_VALUES_PATH = "/content/gdrive/MyDrive/musicdata/vae_for_audio/min_max_values.pkl"
SPECTROGRAMS_PATH = "/content/gdrive/MyDrive/musicdata/vae_for_audio/spectrograms"

def load_fsdd(spectrograms_path):
    x_train = []
    file_paths = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path) # (n_bins, n_frames, 1)
            x_train.append(spectrogram)
            file_paths.append(file_path)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis] # -> (3000, 256, 64, 1)
    return x_train, file_paths


def select_spectrograms(spectrograms,
                        file_paths,
                        min_max_values,
                        num_spectrograms=2):
    sampled_indexes = np.random.choice(range(len(spectrograms)), num_spectrograms)
    sampled_spectrogrmas = spectrograms[sampled_indexes]
    file_paths = [file_paths[index] for index in sampled_indexes]
    sampled_min_max_values = [min_max_values[file_path] for file_path in
                           file_paths]
    print(file_paths)
    print(sampled_min_max_values)
    return sampled_spectrogrmas, sampled_min_max_values


def save_signals(signals, save_dir, sample_rate=22050):
    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, str(i) + ".wav")
        sf.write(save_path, signal, sample_rate)


if __name__ == "__main__":
    # initialise sound generator
    vae = VAE.load("model")
    sound_generator = SoundGenerator(vae, HOP_LENGTH)

    # load spectrograms + min max values
    with open(MIN_MAX_VALUES_PATH, "rb") as f:     # open in reading binary mode
        min_max_values = pickle.load(f)

    specs, file_paths = load_fsdd(SPECTROGRAMS_PATH)

    # sample spectrograms + min max values
    sampled_specs, sampled_min_max_values = select_spectrograms(specs,
                                                                file_paths,
                                                                min_max_values,
                                                                5)

    # generate audio for sampled spectrograms
    signals, _ = sound_generator.generate(sampled_specs,
                                          sampled_min_max_values)

    # convert spectrogram samples to audio
    original_signals = sound_generator.convert_spectrograms_to_audio(
        sampled_specs, sampled_min_max_values)

    # save audio signals
    save_signals(signals, SAVE_DIR_GENERATED)
    save_signals(original_signals, SAVE_DIR_ORIGINAL)

