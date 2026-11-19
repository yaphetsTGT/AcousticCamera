# source: 

# Simple Live Acoustic Camera using BeamformerBase
import acoular as ac
import numpy as np
import cv2
import sounddevice as sd
import time
import AutomaticGainControl
import SRP
import Constants

# Hacky
from pylab import imshow, show

# Disable Acoular's HDF5 caching for live processing
ac.config.global_caching = 'none'

# -----------------------------------------------------------------------------
# HARDWARE CONFIGURATION
# -----------------------------------------------------------------------------
UMA16_DEVICE_INDEX = None  # Line (MCHStreamer Multi-channels), Windows WASAPI
NUM_CHANNELS = 16
CAMERA_INDEX = 0

# -----------------------------------------------------------------------------
# ACOUSTIC GRID & PROCESSING SETTINGS
# -----------------------------------------------------------------------------
#TargetFrequencies = [500, 1000, 2000, 4000]
BLOCK_SIZE = 1024      # 
BLEND_ALPHA = 0.75    # Video transparency

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def find_uma16_device():
    """Find and verify the UMA-16 audio device."""
    global UMA16_DEVICE_INDEX

    devices = sd.query_devices()
    
    if UMA16_DEVICE_INDEX is None:
        # Search for a device with NUM_CHANNELS input channels
        print(f"\nSearching for a device with {NUM_CHANNELS} input channels...")
        for idx, dev in enumerate(devices):
            if dev['max_input_channels'] >= NUM_CHANNELS:
                print(f"Found suitable device at index {idx}:")
                print(f"  Name: {dev['name']}")
                print(f"  Channels: {dev['max_input_channels']} in")
                print(f"  Host API: {sd.query_hostapis(dev['hostapi'])['name']}")
                print(f"  Bit resolution: {dev['default_samplerate']} Hz")
                UMA16_DEVICE_INDEX = idx
                return True
        print(f"⚠️  No device with {NUM_CHANNELS} input channels found!")
        return False
    else:
        print(f"\nUsing audio device {UMA16_DEVICE_INDEX}:")
        if UMA16_DEVICE_INDEX < len(devices):
            dev = devices[UMA16_DEVICE_INDEX]
            print(f"  Name: {dev['name']}")
            print(f"  Channels: {dev['max_input_channels']} in")
            print(f"  Host API: {sd.query_hostapis(dev['hostapi'])['name']}")
            
            if dev['max_input_channels'] < NUM_CHANNELS:
                print(f"\n⚠️  WARNING: Device only has {dev['max_input_channels']} input channels!")
                return False
            return True
        else:
            print(f"⚠️  Device {UMA16_DEVICE_INDEX} not found!")
            return False

def find_camera():
    """Find and verify the camera."""
    print(f"\nUsing camera {CAMERA_INDEX}")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  Resolution: {width}x{height}")
        cap.release()
        return True
    else:
        print(f"⚠️  Camera {CAMERA_INDEX} not found!")
        return False

# -----------------------------------------------------------------------------
# LIVE CAPTURE & PROCESSING LOOP
# -----------------------------------------------------------------------------
def main():
    # # Verify audio device and camera
    if not find_uma16_device():
        return
    if not find_camera():
        return
    
    # Open camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_INDEX}.")
        return

    print("\nSetting up Acoular processing pipeline...")    
    print("\nStarting simple live acoustic camera...")
    print(f"Microphone array: {NUM_CHANNELS} channels at {Constants.SAMPLE_RATE} Hz")
    print(f"Block size: {BLOCK_SIZE} samples")
    print("Press 'q' to quit\n")

    AAGC = AutomaticGainControl.AutomaticGainControl(alpha=0.99)  # Initialize AGC
      
    frame_count = 0
    prev_time = time.perf_counter()
    fps = 0.0
    try:
        while True:
            # Get camera frame
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed")
                break
            
            # Recreate audio pipeline for each frame (chunked pseudo-realtime)
            # This captures fresh audio data for each visualization update
            audio_source = ac.SoundDeviceSamplesGenerator(
                device=UMA16_DEVICE_INDEX, 
                num_channels=NUM_CHANNELS,
                sample_freq=Constants.SAMPLE_RATE,
                num_samples=BLOCK_SIZE * 5  # Capture N blocks per frame
            )

            output = 0.0
            for block in audio_source.result(BLOCK_SIZE):
                # non-overlapping blocks of audio data
                #print(block.shape) # 1024 * 16
                #print('minimum = ', np.min(np.abs(block)), 'maximum = ', np.max(np.abs(block)), 'mean = ', np.mean(block), 'std = ', np.std(block))
                tmp = AAGC.process(block)  # Apply AGC to the block
                assert tmp.shape[0] == BLOCK_SIZE and tmp.shape[1] == NUM_CHANNELS, "AGC output shape mismatch"
                tmp = SRP.EvalSteeringResponsePower(tmp, SamplingRate=Constants.SAMPLE_RATE)  # Apply SRP processing to the AGC output
                #assert np.amin(tmp) >= 0, "SRP output contains negative values"
                output += tmp

            heatmap = 10*np.log10(output + 1e-12)  # Convert to dB scale, avoid log(0)
            heatmap = np.flipud(np.fliplr(heatmap))
            
            max_db = np.max(heatmap)
            min_db = np.min(heatmap)
            db_range = max_db - min_db
            heatmap_clipped = np.clip(heatmap, min_db, max_db)
            heatmap_norm = ((heatmap_clipped - min_db) / db_range * 255).astype(np.uint8)
            
            # Resize to match camera frame and apply colormap (bicubic interpolation)
            heatmap_resized = cv2.resize(heatmap_norm, (frame.shape[1], frame.shape[0]), 
                                        interpolation=cv2.INTER_CUBIC)
            heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            
            # Blend with camera frame
            blended_frame = cv2.addWeighted(frame, 1 - BLEND_ALPHA, heatmap_color, BLEND_ALPHA, 0)
            
            # Add overlay text
            cv2.putText(blended_frame, f"Simple Acoustic Camera", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(blended_frame, f"Max: {max_db:.1f} dB SPL", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(blended_frame, f"Block size: {BLOCK_SIZE}", (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # Smoothed framerate estimate.
            now = time.perf_counter()
            dt = now - prev_time
            if dt > 0:
                inst_fps = 1.0 / dt
                fps = inst_fps if fps == 0.0 else (0.9 * fps + 0.1 * inst_fps)
            prev_time = now
            cv2.putText(blended_frame, f"FPS: {fps:.1f}", (20, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display result
            cv2.imshow('Simple Live Acoustic Camera (UMA-16)', blended_frame)
            
            frame_count += 1
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nAcoustic camera stopped after {frame_count} frames")

if __name__ == '__main__':
    #import cProfile
    #cProfile.run('main()', sort='time')
    main()
