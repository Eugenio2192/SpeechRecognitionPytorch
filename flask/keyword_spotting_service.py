from train import SpeechRecognitionNet
import torch
import numpy as np
import librosa
SAVED_MODEL_PATH = "MODEL/model.ckpt"
NUM_SAMPLES_TO_CONSIDER = 22050

class _KeywordSpottingService:
    model = None
    _mappings = [
        "on",
        "down",
        "go",
        "stop",
        "no",
        "right",
        "off",
        "left"
    ]
    _instance = None

    def predict(self, file_path):

        # extract MFCCs
        MFCCs = self.preprocess(file_path)
        # convert 2d MFCCs array into 4d array
        MFCCs = torch.tensor(MFCCs[np.newaxis, np.newaxis, ...]).float()
        outputs = self.model(MFCCs)
        _, predicted = torch.max(outputs.data, 1)
        keyword = self._mappings[predicted.item()]
        return keyword

    def preprocess(self, file_path, n_mfcc=13, n_fft=2048, hop_length=512):
        signal, sr = librosa.load(file_path)

        if len(signal) > NUM_SAMPLES_TO_CONSIDER:
            signal = signal[:NUM_SAMPLES_TO_CONSIDER]

        MFCCs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

        return MFCCs

def keywordSpottingService():
    if _KeywordSpottingService._instance is None:
        model = SpeechRecognitionNet()
        model.load_state_dict(torch.load(SAVED_MODEL_PATH))
        model.eval()
        _KeywordSpottingService._instance = _KeywordSpottingService()
        _KeywordSpottingService.model = model

    return _KeywordSpottingService._instance

if __name__ == "__main__":

    kss = keywordSpottingService()
    print((kss.predict("test/Down.wav")))
    print((kss.predict("test/Left.wav")))