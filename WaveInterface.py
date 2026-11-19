import numpy as np

# define a class, which gets the stereo buffer of the microphone, append it to the whole recording and finally save the wavefile
class CWaveFileInterface():
    
    def __init__(self, filename, samplerate, channels):
        self.filename = filename
        self.samplerate = samplerate
        self.channels = channels
        self.recording = []

    def append_data(self, data):
        """
        Appends the recorded audio data to the internal recording buffer.
        
        Parameters:
        - data: The recorded audio data to append.
        """
        # convert to int16
        data = (data * 32767).astype(np.int16)
        self.recording.append(data)

    def save_to_file(self):
        """
        Saves the recorded audio data to a wave file.
        """
        import wave
        with wave.open(self.filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # Assuming 16-bit audio
            wf.setframerate(self.samplerate)
            wf.writeframes(b''.join([d.tobytes() for d in self.recording]))