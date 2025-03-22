import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pandas as pd
import os

def analyze_audio_attributes(audio_data, sr):
    """Analyze various sonic qualities of the audio."""
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
    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    mag_spectrum = mag_spectrum[pos_mask]
    
    peak_indices = np.argsort(mag_spectrum)[-5:]
    dominant_freqs = freqs[peak_indices]
    dominant_mags = mag_spectrum[peak_indices]
    
    spec_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
    mean_spec_centroid = np.mean(spec_centroid)
    
    tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    
    # Additional features
    zcr = np.mean(librosa.zero_crossings(audio_data)) * sr
    spec_flatness = np.mean(librosa.feature.spectral_flatness(y=audio_data))
    
    return {
        'duration': duration,
        'max_amplitude': max_amplitude,
        'mean_amplitude': mean_amplitude,
        'mean_rms': mean_rms,
        'dominant_frequencies': list(zip(dominant_freqs, dominant_mags)),
        'mean_spectral_centroid': mean_spec_centroid,
        'estimated_tempo': float(tempo),
        'beat_times': beat_times,
        'zero_crossing_rate': zcr,
        'spectral_flatness': spec_flatness,
        'freqs': freqs,
        'mag_spectrum': mag_spectrum
    }

def plot_analysis(audio_data, sr, analysis_results, min_freq, max_freq):
    """Create visualizations of the audio analysis."""
    fig = plt.figure(figsize=(12, 8))
    
    # Waveform with beat overlay
    ax1 = plt.subplot(2, 2, 1)
    librosa.display.waveshow(audio_data, sr=sr, ax=ax1)
    for beat_time in analysis_results['beat_times']:
        ax1.axvline(x=beat_time, color='g', linestyle='--', alpha=0.5, label='Beats' if beat_time == analysis_results['beat_times'][0] else "")
    ax1.set_title('Waveform with Beats')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    
    # RMS Energy
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
    
    # Frequency Spectrum with range filter
    ax4 = plt.subplot(2, 2, 4)
    freq_mask = (analysis_results['freqs'] >= min_freq) & (analysis_results['freqs'] <= max_freq)
    ax4.semilogx(analysis_results['freqs'][freq_mask], analysis_results['mag_spectrum'][freq_mask])
    ax4.set_title(f'Frequency Spectrum ({min_freq}-{max_freq} Hz)')
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Magnitude')
    ax4.set_xlim([max(20, min_freq), min(sr/2, max_freq)])
    
    plt.tight_layout()
    return fig

def main():
    st.title("Enhanced Audio Analysis Tool")
    
    # Help section
    with st.expander("How to Use & Feature Explanations"):
        st.write("""
        Upload a WAV file to analyze its sonic properties. Features include:
        - **Duration**: Length of the audio in seconds.
        - **Amplitude**: Loudness metrics (max and mean).
        - **RMS Energy**: Average energy over time (loudness).
        - **Spectral Centroid**: Center of frequency mass (brightness).
        - **Dominant Frequencies**: Strongest frequency components.
        - **Tempo**: Estimated beats per minute (BPM).
        - **Zero Crossing Rate**: Noisiness/percussiveness (crossings/sec).
        - **Spectral Flatness**: Tonal (0) vs. noisy (1) quality.
        Use the sliders to filter the frequency spectrum and explore the visualizations!
        """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an audio file (WAV format)", type=['wav'])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            # Load audio file
            audio_data, sr = librosa.load("temp_audio.wav")
            duration = librosa.get_duration(y=audio_data, sr=sr)
            
            # Display audio player
            st.audio("temp_audio.wav", format='audio/wav')
            
            # Frequency range sliders
            st.subheader("Frequency Range Filter")
            min_freq = st.slider("Min Frequency (Hz)", 20, int(sr/2), 20)
            max_freq = st.slider("Max Frequency (Hz)", min_freq, int(sr/2), int(sr/2))
            
            # Advanced features toggle
            show_advanced = st.checkbox("Show Advanced Features", value=False)
            
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
                    
                    # Advanced features
                    if show_advanced:
                        st.subheader("Advanced Features")
                        st.write(f"Zero Crossing Rate: {analysis_results['zero_crossing_rate']:.2f} crossings/sec")
                        st.write(f"Spectral Flatness: {analysis_results['spectral_flatness']:.4f} (0 = tonal, 1 = noisy)")
                    
                    # Visualizations
                    st.subheader("Visual Analysis")
                    fig = plot_analysis(audio_data, sr, analysis_results, min_freq, max_freq)
                    st.pyplot(fig)
                    
                    # Export options
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
                    if show_advanced:
                        results_str += (
                            f"\nAdvanced Features:\n"
                            f"Zero Crossing Rate: {analysis_results['zero_crossing_rate']:.2f} crossings/sec\n"
                            f"Spectral Flatness: {analysis_results['spectral_flatness']:.4f}\n"
                        )
                    
                    st.download_button(
                        label="Download Analysis (TXT)",
                        data=results_str,
                        file_name="audio_analysis.txt",
                        mime="text/plain"
                    )
                    
                    # CSV Export - Separate DataFrames for sample-based and frame-based data
                    # Sample-based data (Time and Amplitude)
                    time_samples = librosa.times_like(audio_data, sr=sr)
                    df_samples = pd.DataFrame({
                        'Time (s)': time_samples,
                        'Amplitude': np.abs(audio_data)
                    })
                    
                    # Frame-based data (RMS)
                    rms = librosa.feature.rms(y=audio_data)[0]
                    time_frames = librosa.times_like(rms, sr=sr)
                    df_rms = pd.DataFrame({
                        'Time (s)': time_frames,
                        'RMS': rms
                    })
                    
                    # Provide two separate CSV downloads
                    st.download_button(
                        label="Download Sample Data (CSV)",
                        data=df_samples.to_csv(index=False),
                        file_name="audio_sample_data.csv",
                        mime="text/csv"
                    )
                    
                    st.download_button(
                        label="Download RMS Data (CSV)",
                        data=df_rms.to_csv(index=False),
                        file_name="audio_rms_data.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
        
        # Clean up
        if os.path.exists("temp_audio.wav"):
            os.remove("temp_audio.wav")

if __name__ == "__main__":
    main()
