# class for automatic gain control (AGC) of audio signals
# input blocks of size BLOCK_SIZE*NumOfChannels, output blocks of same size
# normalisation to zero mean and unit variance
# learn the mean and std of the signal over several blocks, and apply to the next block
import numpy as np

# define smallest positive floating point number to avoid division by zero
EPSILON = np.finfo(float).eps
DynamicRangeIndB = 60

class AutomaticGainControl(object):

    def __init__(self, alpha=0.99):
        self.__counter = 0
        self.__mean = 0
        self.__std = 1
        self.__alpha = alpha
        self.__STDNoiseFloor = 1 / np.sqrt(10 ** (DynamicRangeIndB / 10)) 

    def process(self, block):
        alpha = self.__counter / (self.__counter + 1)
        if alpha > self.__alpha:
            alpha = self.__alpha
        else:
            self.__counter += 1

        # Update mean and std using exponential moving average
        current_mean = np.mean(block, axis=0)
        current_std = np.std(block, axis=0)

        self.__mean = alpha * self.__mean + (1 - alpha) * current_mean
        self.__std = alpha * self.__std + (1 - alpha) * current_std


        # block is of size BLOCK_SIZE * NumOfChannels, where channels are columns
        # self.__mean is of size NumOfChannels, self.__std is of size NumOfChannels
        # apply normalisation to each channel
        normalized_block = (block - np.outer(np.ones(block.shape[0]), self.__mean)) / np.outer(np.ones(block.shape[0]), self.__std + EPSILON)
        # Add small noise to avoid exact zeros
        normalized_block += np.random.randn(*block.shape) * self.__STDNoiseFloor  
        return normalized_block

    def GetMean(self):
        return self.__mean

    def GetSTD(self):
        return self.__std

# calling the script stand alone:
if __name__ == '__main__':
    # test the AGC class
    TargetMean = 2.0
    TargetSTD = 3.0
    NUM_CHANNELS = 16
    BlockSize = 1024
    agc = AutomaticGainControl(alpha=0.99)
    # generate a random block of size 1024 * 16
    for n in range(5):
        block = np.random.randn(BlockSize, NUM_CHANNELS) * TargetSTD + TargetMean
        normalized_block = agc.process(block)
        assert np.allclose(np.mean(normalized_block, axis=0), 0.0, atol=1e-1), "Mean is not close to zero"
        assert np.allclose(np.std(normalized_block, axis=0), 1.0, atol=1e-1), "Std is not close to one"

    assert np.allclose(agc.GetMean(), TargetMean, atol=1e-1), "AGC mean is not close to target mean"
    assert np.allclose(agc.GetSTD(), TargetSTD, atol=1e-1), "AGC std is not close to target std"