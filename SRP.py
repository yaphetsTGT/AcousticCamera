import numpy as np
import Geometry
import Constants

UsePHATWeighting = False

def EvaluateGeneralizedCrossCorrelation(S, G):
    #FFTLen = 2*np.maximum(s.shape[0], g.shape[0])
    #Phi = np.fft.fft(g, n=FFTLen) * np.conj(np.fft.fft(s, n=FFTLen))
    Phi = G * np.conj(S)
    if UsePHATWeighting: Phi /= np.abs(Phi) # PHAT weighting
    return np.fft.irfft(Phi).real

def ApplyFFTAndBandpassFilter(x, LowCutoffFrequency = Constants.BandpassLowCutoffFrequency, HighCutoffFrequency = Constants.BandpassHighCutoffFrequency, SamplingRate = Constants.SAMPLE_RATE):
    FFTLen = 2*x.shape[0]
    Deltaf = SamplingRate / FFTLen
    idx1 = int(np.floor(LowCutoffFrequency / Deltaf))
    idx2 = int(np.ceil(HighCutoffFrequency / Deltaf))
    X = np.fft.rfft(x, n=FFTLen, axis=0)
    X[:idx1] = 0.0
    X[idx2:] = 0.0
    return X

def EvaluateDelay(x, y):
    X = ApplyFFTAndBandpassFilter(x)
    Y = ApplyFFTAndBandpassFilter(y)
    ### evaluate generalized cross-correlation and find delay
    phi = EvaluateGeneralizedCrossCorrelation(X, Y)
    FFTLen = phi.shape[0]
    Delay = np.argmax(phi)
    if Delay > FFTLen/2:
        Delay -= FFTLen
    return Delay

class CGeneralizedCrossCorrelationStorage(object):

    def __init__(self, m1, m2, phi):
        self.__m1 = m1
        self.__m2 = m2
        self.__phi = phi

    def GetM1(self):
        return self.__m1
    
    def GetM2(self):
        return self.__m2
    
    def GetPhi(self):
        return self.__phi

def EvaluateGeneralizedCrossCorrelationForAllMicrophonePairs(RecordedSignals):
    RecordedSignalsSpectra = ApplyFFTAndBandpassFilter(RecordedSignals)
    CrossCorrelationValues = []
    for m1 in range(len(Geometry.CoordinatesOfMicrophones)):
        for m2 in range(m1+1, len(Geometry.CoordinatesOfMicrophones)):
            CrossCorrelation = EvaluateGeneralizedCrossCorrelation(RecordedSignalsSpectra[:, m1], RecordedSignalsSpectra[:, m2])
            CrossCorrelationValues.append(CGeneralizedCrossCorrelationStorage(m1, m2, CrossCorrelation))
    return CrossCorrelationValues

def EvalResponseForPoint(ColumnIndex, RowIndex, CrossCorrelationValues, SamplingRate):  
    Response = 0.0
    for CrossCorrelationStorage in CrossCorrelationValues:
        m1 = CrossCorrelationStorage.GetM1()
        m2 = CrossCorrelationStorage.GetM2()
        phi = CrossCorrelationStorage.GetPhi()
        DelayDifference = Geometry.AGeometry.GetDelay(m1, m2, ColumnIndex, RowIndex)
        SampleDifference = int(DelayDifference * SamplingRate)
        if SampleDifference < 0:
            SampleDifference += phi.shape[0]
        Response += phi[SampleDifference]
    return Response

def EvalSteeringResponsePower(RecordedSignals, SamplingRate):
    SRP = np.zeros((Geometry.SrpGridX, Geometry.SrpGridY))
    CrossCorrelationValues = EvaluateGeneralizedCrossCorrelationForAllMicrophonePairs(RecordedSignals)    
    for ColumnIndex in range(SRP.shape[1]):
        for RowIndex in range(SRP.shape[0]):            
            Response = EvalResponseForPoint(ColumnIndex, RowIndex, CrossCorrelationValues, SamplingRate)
            SRP[RowIndex, ColumnIndex] = Response
    return SRP

if __name__ == '__main__':
    # Example usage of the SRP class
    sample_rate = 48000  # Sample rate in Hz
    

    import cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    for n in range(10):
        Block = np.random.randn(1024, 16)  # Simulated block of audio data
        SRP_output = EvalSteeringResponsePower(Block, sample_rate)
    profiler.disable()
    profiler.print_stats(sort='time')

