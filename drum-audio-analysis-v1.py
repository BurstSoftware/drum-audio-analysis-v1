import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

def analyze_audio_attributes(audio_data, sr):
    """Analyze various sonic qualities of the audio."""
    # Basic attributes
    duration = librosa.get_duration(y=audio_data, sr=sr)
    amplitude = np.abs(audio_data)
    max_amplitude = np.max(amplitude)
    mean_amplitude = np.mean(amplitude)
    rms = librosa.feature.rms(y=audio_data)[0]
    mean_rms = np.mean(rms)
    
    # Frequency analysis
    fft = np.fft.fft(audio_data)
    freqs = np.fft.fftfreq(len(fft)) * sr
    mag_spectrum = np.abs(fft)
    
    # Only take positive frequencies
    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    mag_spectrum = mag_spectrum[pos_mask]
    
    # Find dominant frequencies
    peak_indices = np.argsort(mag_spectrum)[-5:]  # Top 5 peaks
    dominant_freqs = freqs[peak_indices]
    dominant_mags = mag_spectrum[peak_indices]
    
    # Spectral centroid
    spec_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
    mean_spec_centroid = np.mean(spec_centroid)
    
    # Tempo estimation
    tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
    
    return {
        'duration': duration,
        'max_amplitude': max_amplitude,
        'mean_amplitude': mean_amplitude,
        'mean_rms': mean_rms,
        'dominant_frequencies': list(zip(dominant_freqs, dominant_mags)),
        'mean_spectral_centroid': mean_spec_centroid,
        'estimated_tempo': float(tempo)
    }

def plot_analysis(audio_data, sr, analysis_results):
    """Create visualizations of the audio analysis."""
    # Create figure with subplots
    fig = plt.figure(figsize=(12, 8))
    
    # Waveform
    ax1 = plt.subplot(2, 2, 1)
    librosa.display.waveshow(audio_data, sr=sr, ax=ax1)
    ax1.set_title('Waveform')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    
    # Amplitude envelope
    ax2 = plt.subplot(2, 2, 2)
    rms = librosa.feature.rms(y=audio_data)[0]
    times = librosa.times_like(rms, sr=sr)
    ax2.plot(times, rms)
    ax2.set_title('RMS Energy')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('RMS')
    
    # Spectrogram
    ax3 = plt.subplot(2, 2, 3)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    librosa.display.specshow(D, y_axis='linear', x_axis='time', sr=sr, ax=ax3)
    ax3.set_title('Spectrogram')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Frequency (Hz)')
    
    # Frequency spectrum
    ax4 = plt.subplot(2, 2, 4)
    fft = np.fft.fft(audio_data)
    freqs = np.fft.fftfreq(len(fft)) * sr
    mag_spectrum = np.abs(fft)
    pos_mask = freqs >= 0
    ax4.semilogx(freqs[pos_mask], mag_spectrum[pos_mask])
    ax4.set_title('Frequency Spectrum')
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Magnitude')
    ax4.set_xlim([20, sr/2])  # Focus on audible range
    
    plt.tight_layout()
    return fig

def main():
    st.title("Audio Analysis Tool")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an audio file (WAV format)", type=['wav'])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load and process audio
        try:
            # Load audio file
            audio_data, sr = librosa.load("temp_audio.wav")
            
            # Display audio player
            st.audio("temp_audio.wav", format='audio/wav')
            
            # Process audio
            if st.button("Analyze Audio"):
                with st.spinner("Analyzing audio..."):
                    # Analyze audio attributes
                    analysis_results = analyze_audio_attributes(audio_data, sr)
                    
                    # Display results
                    st.subheader("Audio Analysis Results")
                    
                    st.write(f"Duration: {analysis_results['duration']:.2f} seconds")
                    st.write(f"Maximum Amplitude: {analysis_results['max_amplitude']:.4f}")
                    st.write(f"Mean Amplitude: {analysis_results['mean_amplitude']:.4f}")
                    st.write(f"Mean RMS Energy: {analysis_results['mean_rms']:.4f}")
                    st.write(f"Mean Spectral Centroid: {analysis_results['mean_spectral_centroid']:.2f} Hz")
                    st.write(f"Estimated Tempo: {analysis_results['estimated_tempo']:.2f} BPM")
                    
                    st.subheader("Dominant Frequencies")
                    for freq, mag in sorted(analysis_results['dominant_frequencies'], key=lambda x: x[1], reverse=True):
                        st.write(f"Frequency: {freq:.2f} Hz, Magnitude: {mag:.2f}")
                    
                    # Display visualizations
                    st.subheader("Visual Analysis")
                    fig = plot_analysis(audio_data, sr, analysis_results)
                    st.pyplot(fig)
                    
                    # Save option
                    results_str = (
                        f"Audio Analysis Results\n"
                        f"Duration: {analysis_results['duration']:.2f} seconds\n"
                        f"Maximum Amplitude: {analysis_results['max_amplitude']:.4f}\n"
                        f"Mean Amplitude: {analysis_results['mean_amplitude']:.4f}\n"
                        f"Mean RMS Energy: {analysis_results['mean_rms']:.4f}\n"
                        f"Mean Spectral Centroid: {analysis_results['mean_spectral_centroid']:.2f} Hz\n"
                        f"Estimated Tempo: {analysis_results['estimated_tempo']:.2f} BPM\n"
                        f"\nDominant Frequencies:\n"
                        + "\n".join(f"Freq: {freq:.2f} Hz, Mag: {mag:.2f}" 
                                  for freq, mag in sorted(analysis_results['dominant_frequencies'], 
                                                        key=lambda x: x[1], reverse=True))
                    )
                    st.download_button(
                        label="Download Analysis",
                        data=results_str,
                        file_name="audio_analysis.txt",
                        mime="text/plain"
                    )
                    
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
        
        # Clean up
        if os.path.exists("temp_audio.wav"):
            os.remove("temp_audio.wav")

if __name__ == "__main__":
    main()
