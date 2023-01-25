# written by zilin
from features.flow.backward_iat_max import BackwardIatMax
from features.flow.backward_iat_mean import BackwardIatMean
from features.flow.backward_iat_min import BackwardIatMin
from features.flow.backward_iat_std import BackwardIatStd
from features.flow.backward_iat_total import BackwardIatTotal

class FeatureExtractor:
    def __init__(self,windows):
        self.features = []
        self.windows = windows

    def add_windows(self,windows):
        for wnd in windows:
            self.windows.append(wnd)

    def add_features(self):
        b_iat_max = BackwardIatMax("b_iat_max")
        b_iat_mean = BackwardIatMean("b_iat_mean")
        b_iat_min = BackwardIatMin("b_iat_min")
        b_iat_std = BackwardIatStd("b_iat_std")
        b_iat_total = BackwardIatTotal("b_iat_total")

        self.features.append(b_iat_max)
        self.features.append(b_iat_mean)
        self.features.append(b_iat_min)
        self.features.append(b_iat_std)
        self.features.append(b_iat_total)

    def process_windows(self):
        for wnd in self.windows:
            self.extract_feature(wnd)

    # extract feature of one window
    def extract_feature(self,window):
        for f in self.features:
            f.extract_feature(window)

