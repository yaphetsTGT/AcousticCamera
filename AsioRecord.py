import sounddevice as sd
import numpy as np
import WaveInterface

DurationOfRecording = 5
SamplingRate = 16000
NumberOfChannels = 2
WaveFileInterface = WaveInterface.CWaveFileInterface("output.wav", SamplingRate, NumberOfChannels)

def ListAllMicrofoneInterfaces():
    """
    Lists all available microphone interfaces on the system.
    """
    result = []
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            #print(f"Microphone Interface {idx}: {device['name']} - Input Channels: {device['max_input_channels']}")
            result.append(device)
    return result

def SelectScarlettMicrophoneInterface(DeviceList):
    """
    Selects the Scarlett microphone interface from the provided device list.
    
    Parameters:
    - DeviceList: A list of available microphone interfaces.
    
    Returns:
    - The selected Scarlett microphone interface, or None if not found.
    """
    for device in DeviceList:
        if "Scarlett" in device['name']:
            print(f"Selected Scarlett Microphone Interface: {device['name']}")
            return device
    print("Scarlett Microphone Interface not found.")
    return None

def callback(indata, frames, time, status):
    """
    Callback function to process the recorded audio data.
    
    Parameters:
    - indata: The recorded audio data.
    - frames: Number of frames recorded.
    - time: Time information for the recording.
    - status: Status of the recording (e.g., if there were any errors).
    """
    if status:
        print(f"Status: {status}")
    ConvertIndataToNumpyArray(indata)

def ConvertIndataToNumpyArray(indata):
    """
    Converts the recorded audio data (indata) to a NumPy array.
    
    Parameters:
    - indata: The recorded audio data.
    
    Returns:
    - A NumPy array containing the audio data.
    """
    y = np.array(indata)
    ### dithering
    Bits = 16
    NoiseFactor = 1 / float(2 ** (Bits - 1) + 1.0)
    y += (np.random.rand(y.shape[0], y.shape[1]) - 0.5) * NoiseFactor  # Add a small random noise to avoid zero values
    ### do anything
    print(10*np.log10(np.mean(y**2, axis = 0)))  # Print the power of the audio signal in dB
    WaveFileInterface.append_data(y)  # Append the audio data to the wave file interface
    return y

def OpenDeviceAndRecordAudioWithCallback(device, callback, duration=5, samplerate=44100, channels=1, blocksize=1024):
    """
    Opens the default microphone device and records audio for a specified duration using a callback function.
    
    Parameters:
    - callback: A function that will be called with the recorded audio data.
    - duration: Duration of the recording in seconds (default is 5 seconds).
    - samplerate: Sampling rate for the recording (default is 44100 Hz).
    - channels: Number of audio channels (default is 1 for mono).
    - device: The microphone device to use for recording.
    - blocksize: The block size for the audio stream (default is 1024).
    """
    try:
        with sd.InputStream(device=device, callback=callback, channels=channels, samplerate=samplerate, blocksize=blocksize):
            print(f"Recording for {duration} seconds...")
            sd.sleep(int(duration * 1000))  # Convert duration to milliseconds
            WaveFileInterface.save_to_file()  # Save the recorded audio to a wave file
            print("Recording finished.")
    except Exception as e:
        print(f"An error occurred while recording audio: {e}")

MaximumMicrofoneDistance = 0.1

Overlap = 0.9
SpeedOfSound = 343
#BlockSize = int(MaximumMicrofoneDistance / SpeedOfSound * SamplingRate * (1 - Overlap))
BlockSize = 4096

ListAllMicrofoneInterfaces = ListAllMicrofoneInterfaces()
ScarlettMicrophoneInterface = SelectScarlettMicrophoneInterface(ListAllMicrofoneInterfaces)
OpenDeviceAndRecordAudioWithCallback(device = ScarlettMicrophoneInterface["index"], callback = callback, duration=DurationOfRecording, samplerate=SamplingRate, channels=NumberOfChannels, blocksize=BlockSize)
