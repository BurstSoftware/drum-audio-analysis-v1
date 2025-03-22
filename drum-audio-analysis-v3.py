import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

def analyze_audio_attributes(audio_data, sr):
    """Analyze frequency-related attributes of the audio."""
    # Basic attributes
    duration = librosa.get_duration(y=audio_data, sr=sr)
    
    # Frequency analysis
    fft = np.fft.fft(audio_data)
    freqs = np.fft.fftfreq(len(fft)) * sr
    mag_spectrum = np.abs(fft)
    
    # Only take positive frequencies
    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    mag_spectrum = mag_spectrum[pos_mask]
    
    # Find top 50 loudest frequencies
    peak_indices = np.argsort(mag_spectrum)[-50:]  # Top 50 peaks
    top_freqs = freqs[peak_indices]
    top_mags = mag_spectrum[peak_indices]
    
    # Sort by frequency for consistent plotting
    sorted_indices = np.argsort(top_freqs)
    top_freqs = top_freqs[sorted_indices]
    top_mags = top_mags[sorted_indices]
    
    # Bin frequencies into 50 ranges for the frequency range visualization
    freq_bins = np.linspace(20, sr/2, 50)  # 50 bins from 20 Hz to Nyquist frequency
    freq_bin_indices = np.digitize(freqs, freq_bins)
    binned_mags = np.zeros(len(freq_bins) - 1)
    for i in range(1, len(freq_bins)):
        bin_mask = freq_bin_indices == i
        if np.any(bin_mask):
            binned_mags[i-1] = np.mean(mag_spectrum[bin_mask])
    
    return {
        'duration': duration,
        'freqs': freqs,
        'mag_spectrum': mag_spectrum,
        'top_freqs': top_freqs,
        'top_mags': top_mags,
        'freq_bins': freq_bins,
        'binned_mags': binned_mags
    }

def plot_analysis(audio_data, sr, analysis_results):
    """Create visualizations of the frequency analysis."""
    # Create figure with three subplots
    fig = plt.figure(figsize=(15, 12))
    
    # Plot 1: Top 50 loudest frequencies
    ax1 = plt.subplot(3, 1, 1)
    ax1.stem(analysis_results['top_freqs'], analysis_results['top_mags'], linefmt='b-', markerfmt='bo', basefmt='r-')
    ax1.set_title('Top 50 Loudest Frequencies')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude')
    ax1.set_xscale('log')
    ax1.set_xlim([20, sr/2])
    ax1.grid(True)
    
    # Plot 2: Top 50 frequency ranges
    ax2 = plt.subplot(3, 1, 2)
    bin_centers = (analysis_results['freq_bins'][:-1] + analysis_results['freq_bins'][1:]) / 2
    ax2.bar(bin_centers, analysis_results['binned_mags'], width=np.diff(analysis_results['freq_bins']), align='center', color='purple', alpha=0.7)
    ax2.set_title('Top 50 Frequency Ranges (Binned)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Average Magnitude')
    ax2.set_xscale('log')
    ax2.set_xlim([20, sr/2])
    ax2.grid(True)
    
    # Plot 3: All frequencies in the audio file
    ax3 = plt.subplot(3, 1, 3)
    ax3.semilogx(analysis_results['freqs'], analysis_results['mag_spectrum'], color='blue')
    ax3.set_title('Full Frequency Spectrum')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Magnitude')
    ax3.set_xlim([20, sr/2])
    ax3.grid(True)
    
    plt.tight_layout()
    return fig

def main():
    st.title("Drum Audio Frequency Analysis Tool")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a drum audio file (WAV format)", type=['wav'])
    
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
            if st.button("Analyze Frequencies"):
                with st.spinner("Analyzing audio..."):
                    # Analyze audio attributes
                    analysis_results = analyze_audio_attributes(audio_data, sr)
                    
                    # Display results
                    st.subheader("Frequency Analysis Results")
                    
                    st.write(f"Duration: {analysis_results['duration']:.2f} seconds")
                    
                    st.subheader("Top 50 Loudest Frequencies")
                    for freq, mag in zip(analysis_results['top_freqs'], analysis_results['top_mags']):
                        st.write(f"Frequency: {freq:.2f} Hz, Magnitude: {mag:.2f}")
                    
                    # Display visualizations
                    st.subheader("Visual Analysis")
                    fig = plot_analysis(audio_data, sr, analysis_results)
                    st.pyplot(fig)
                    
                    # Save option
                    results_str = (
                        f"Drum Audio Frequency Analysis\n"
                        f"Duration: {analysis_results['duration']:.2f} seconds\n"
                        f"\nTop 50 Loudest Frequencies:\n"
                        + "\n".join(f"Freq: {freq:.2f} Hz, Mag: {mag:.2f}" 
                                  for freq, mag in zip(analysis_results['top_freqs'], analysis_results['top_mags']))
                    )
                    st.download_button(
                        label="Download Analysis",
                        data=results_str,
                        file_name="frequency_analysis.txt",
                        mime="text/plain"
                    )
                    
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
        
        # Clean up
        if os.path.exists("temp_audio.wav"):
            os.remove("temp_audio.wav")

if __name__ == "__main__":
    main()
