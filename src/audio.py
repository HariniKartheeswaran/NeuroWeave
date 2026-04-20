import wave
import struct
import math
import numpy as np

def generate_neural_scream(history, path, duration=5.0, sample_rate=44100):
    """
    Translates the neural network's confidence history into a sonic experience.
    Higher attack confidence -> Higher screeching pitch.
    Lower confidence -> More distortion/noise.
    """
    num_samples = int(duration * sample_rate)
    audio_data = []
    
    # Smooth frequency mapping
    freqs = [200 + (p * 2800) for p in history]
    
    phase = 0.0
    for i in range(num_samples):
        progress = i / num_samples
        
        # Determine index and interpolate linearly
        idx_float = progress * (len(history) - 1)
        idx_low = int(math.floor(idx_float))
        idx_high = min(idx_low + 1, len(history) - 1)
        fraction = idx_float - idx_low
        
        f = freqs[idx_low] * (1 - fraction) + freqs[idx_high] * fraction
        conf = history[idx_low] * (1 - fraction) + history[idx_high] * fraction
        
        # Wobble
        wobble = math.sin(2 * math.pi * 8 * (i / sample_rate)) * (1.0 - conf) * 100
        
        phase += 2 * math.pi * (f + wobble) / sample_rate
        sample = math.sin(phase)
        
        # Neural Noise (distortion based on uncertainty)
        noise = (np.random.rand() * 2 - 1) * (1.0 - conf) * 0.5
        
        sample_val = max(min(sample + noise, 1.0), -1.0)
        
        # 16-bit PCM conversion
        pcm = int(sample_val * 32767)
        audio_data.append(pcm)
        
    with wave.open(path, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        
        # Batch writing is faster
        packed_data = struct.pack('<' + 'h'*len(audio_data), *audio_data)
        wav_file.writeframes(packed_data)
