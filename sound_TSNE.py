def kk():
    os.system('kill -9 %d' % os.getpid())


import librosa
import numpy as np
import glob
import os
from matplotlib import pyplot as plt
from sklearn import manifold


# Program setting

AUDIO_DIR = '../data'
SAVE_DIR = os.path.join(AUDIO_DIR, '_processed')

CUTOFF_SIZE = 4  # Cut the file into 4s segment
FRAMES = 40 * CUTOFF_SIZE + 1
BANDS = 60

LABEL_TO_C = {
    'air_conditioner': 0,
    'car_horn': 1,
    'children_playing': 2,
    'dog_bark': 3,
    'nasa_sound': 4
}

PROCESS_AND_SAVE = True


# Helper functions

def get_label(filename):
    return filename.replace('../data/', '')


def windows(data, window_size):
    # assert window_size % 2 == 0
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += window_size


def extract_feature_from_file(filename, bands=BANDS, frames=FRAMES):
    window_size = 512 * (frames - 1)
    log_specgrams = []
    sound_series, s = librosa.load(filename)
    label = get_label(filename)

    for start, end in windows(sound_series, window_size):
        if len(sound_series[start: end]) == window_size:
            signal = sound_series[start: end]
            melspec = librosa.feature.melspectrogram(signal, n_mels=bands)
            logspec = librosa.logamplitude(melspec)
            logspec = logspec.T.flatten()[:, np.newaxis].T
            log_specgrams.append(logspec)

    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams), bands, frames, 1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)

    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])

    return np.array(features), np.array([label] * features.shape[0])


def extract_feature_from_dir(dirname, file_extensions=('*.wav', '*.mp3'), bands=BANDS, frames=FRAMES, save_data=False):
    features = None
    for file_extension in file_extensions:
        for filename in glob.glob(os.path.join(dirname, '*', file_extension)):
            print filename
            _features, _labels = extract_feature_from_file(filename=filename, bands=bands, frames=frames)
            if not len(_features):
                continue
            if features is None:
                features, labels = _features, _labels
            else:
                features = np.append(features, _features, axis=0)
                labels = np.append(labels, _labels)
    if save_data:
        np.save(os.path.join(SAVE_DIR, 'features.npy'), features)
        np.save(os.path.join(SAVE_DIR, 'labels.npy'), labels)

    return features, labels

if __name__ == '__main__':
    if PROCESS_AND_SAVE:
        features, labels = extract_feature_from_dir(AUDIO_DIR, file_extensions=['*.mp3', '*.wav'], save_data=True)
    else:
        features = np.load(os.path.join(SAVE_DIR, 'features.npy'))
        labels = np.load(os.path.join(SAVE_DIR, 'labels.npy'))

    tsne = manifold.TSNE(n_components=2, perplexity=10, init='pca', random_state=0)

    x = features.reshape(features.shape[0], features.shape[1] * features.shape[2] * features.shape[3])
    tsne_data = tsne.fit_transform(x)

    plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=[LABEL_TO_C[x.split('/')[-2]] for x in labels], cmap=plt.cm.rainbow)

    with open(os.path.join(SAVE_DIR, 'result.csv'), 'w') as f:
        f.write('name,sound,x,y\n')
        for i in range(tsne_data.shape[0]):
            val = tsne_data[i, :].tolist()
            f.write('{name},{label},{x},{y}\n'.format(name=labels[i].split('/')[-1], label=labels[i].split('/')[-2], x=val[0], y=val[1]))

    plt.show()
