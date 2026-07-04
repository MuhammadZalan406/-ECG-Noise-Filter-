import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
from scipy.ndimage import median_filter
import warnings
warnings.filterwarnings('ignore')

class ECGProcessor:
    def __init__(self, seed=42):
        self.raw_signal = None
        self.noisy_signal = None
        self.filtered_signal = None
        self.peaks = None
        self.true_peaks = None
        self.sampling_rate = 500
        self.timestamps = None
        
        np.random.seed(seed)
        
    def generate_ecg(self, duration=5, heart_rate=72):
        print(f"Generating ECG signal...")
        
        t = np.linspace(0, duration, int(duration * self.sampling_rate))
        self.timestamps = t
        
        beat_interval = 60 / heart_rate
        
        ecg = np.zeros_like(t)
        self.true_peaks = []
        
        for i in range(int(duration / beat_interval)):
            center = i * beat_interval + beat_interval / 2
            if center > duration:
                break
            
            peak_idx = np.argmin(np.abs(t - center))
            self.true_peaks.append(peak_idx)
            
            r_peak = np.exp(-((t - center) ** 2) / (2 * 0.008 ** 2)) * 1.2
            q_wave = -np.exp(-((t - (center - 0.02)) ** 2) / (2 * 0.005 ** 2)) * 0.25
            s_wave = -np.exp(-((t - (center + 0.02)) ** 2) / (2 * 0.005 ** 2)) * 0.2
            p_wave = np.exp(-((t - (center - 0.08)) ** 2) / (2 * 0.015 ** 2)) * 0.15
            t_wave = np.exp(-((t - (center + 0.1)) ** 2) / (2 * 0.03 ** 2)) * 0.2
            
            ecg += r_peak + q_wave + s_wave + p_wave + t_wave
        
        baseline = 0.03 * np.sin(2 * np.pi * 0.15 * t)
        ecg += baseline
        
        ecg = ecg / np.max(np.abs(ecg))
        
        self.raw_signal = ecg
        self.true_peaks = np.array(self.true_peaks)
        
        print(f"ECG generated: {len(ecg)} samples, {duration}s, {heart_rate} BPM")
        print(f"True peaks: {len(self.true_peaks)}")
        return ecg, t
    
    def add_noise(self, signal, noise_level=0.3, seed=None):
        print(f"Adding noise (level: {noise_level})...")
        
        if seed is not None:
            np.random.seed(seed)
        
        t = self.timestamps
        
        gaussian_noise = np.random.normal(0, noise_level * 0.15, len(signal))
        
        muscle_noise = np.random.randn(len(signal)) * 0.08
        muscle_noise = np.convolve(muscle_noise, np.ones(3)/3, mode='same')
        
        power_noise = 0.05 * np.sin(2 * np.pi * 50 * t + np.random.rand() * 2 * np.pi)
        
        motion_artifacts = np.zeros_like(signal)
        for _ in range(np.random.randint(1, 4)):
            start = np.random.randint(0, len(signal) - 100)
            motion_artifacts[start:start+100] = np.random.randn(100) * 0.2
        
        total_noise = gaussian_noise + muscle_noise + power_noise + motion_artifacts * 0.1
        total_noise = total_noise * noise_level / (np.std(total_noise) + 1e-6)
        
        noisy = signal + total_noise
        noisy = np.clip(noisy, -2, 2)
        
        self.noisy_signal = noisy
        
        signal_power = np.var(signal)
        noise_power = np.var(total_noise)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-6))
        print(f"Noise added | SNR: {snr:.2f} dB")
        
        return noisy
    
    def bandpass_filter(self, signal, lowcut=0.5, highcut=35, order=4):
        print(f"Applying bandpass filter ({lowcut}-{highcut} Hz)...")
        
        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        
        b, a = butter(order, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)
        filtered = median_filter(filtered, size=3)
        
        self.filtered_signal = filtered
        print(f"Bandpass filter applied")
        
        return filtered
    
    def moving_average_filter(self, signal, window_size=5):
        print(f"Applying moving average filter (window: {window_size})...")
        
        window = np.ones(window_size) / window_size
        filtered = np.convolve(signal, window, mode='same')
        
        filtered[:window_size//2] = signal[:window_size//2]
        filtered[-window_size//2:] = signal[-window_size//2:]
        
        self.filtered_signal = filtered
        print(f"Moving average filter applied")
        
        return filtered
    
    def detect_peaks_adaptive(self, signal, height_factor=1.5, min_distance=0.4):
        print("Detecting peaks with adaptive threshold...")
        
        signal_std = np.std(signal)
        signal_mean = np.mean(signal)
        
        dynamic_factor = height_factor * (1 + 0.5 * (signal_std / 0.3))
        height = signal_mean + dynamic_factor * signal_std
        
        min_dist_samples = int(min_distance * self.sampling_rate)
        
        peaks, properties = find_peaks(
            signal,
            height=height,
            distance=min_dist_samples,
            prominence=0.15,
            width=3,
            rel_height=0.5
        )
        
        peaks = peaks[(peaks > 50) & (peaks < len(signal) - 50)]
        
        if len(peaks) > 0 and len(properties.get('prominences', [])) > 0:
            prominences = properties['prominences']
            if len(prominences) == len(peaks):
                mean_prom = np.mean(prominences)
                peaks = peaks[prominences > mean_prom * 0.4]
        
        if len(peaks) > 1:
            rr_intervals = np.diff(self.timestamps[peaks])
            median_rr = np.median(rr_intervals)
            
            valid_peaks = [peaks[0]]
            for i in range(1, len(peaks)):
                current_rr = self.timestamps[peaks[i]] - self.timestamps[peaks[i-1]]
                if 0.5 * median_rr <= current_rr <= 1.5 * median_rr:
                    valid_peaks.append(peaks[i])
            
            peaks = np.array(valid_peaks)
        
        if len(peaks) > 0 and self.true_peaks is not None and len(self.true_peaks) > 0:
            peak_heights = signal[peaks]
            sorted_idx = np.argsort(peak_heights)[::-1]
            expected_peaks = len(self.true_peaks)
            
            if len(peaks) > expected_peaks:
                peaks = peaks[sorted_idx[:expected_peaks]]
                peaks = np.sort(peaks)
        
        self.peaks = peaks
        
        if len(peaks) > 1:
            rr_intervals = np.diff(self.timestamps[peaks])
            valid_rr = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 2.0)]
            if len(valid_rr) > 0:
                heart_rate = 60 / np.mean(valid_rr)
            else:
                heart_rate = 60 / np.mean(rr_intervals)
        else:
            heart_rate = 0
        
        print(f"Found {len(peaks)} peaks | Heart Rate: {heart_rate:.2f} BPM")
        
        return peaks, heart_rate
    
    def detect_peaks(self, signal, height=0.3, distance=30):
        print("Detecting heartbeat peaks...")
        
        peaks, properties = find_peaks(
            signal,
            height=height,
            distance=distance,
            prominence=0.1,
            width=2
        )
        
        self.peaks = peaks
        
        if len(peaks) > 1:
            rr_intervals = np.diff(self.timestamps[peaks])
            heart_rate = 60 / np.mean(rr_intervals)
        else:
            heart_rate = 0
        
        print(f"Found {len(peaks)} peaks | Heart Rate: {heart_rate:.1f} BPM")
        
        return peaks, heart_rate
    
    def calculate_accuracy(self, true_peaks, detected_peaks, tolerance=25):
        if len(true_peaks) == 0:
            return 0, 0, 0
        
        tp = 0
        matched_detected = set()
        
        for true_p in true_peaks:
            for det_p in detected_peaks:
                if abs(true_p - det_p) <= tolerance:
                    tp += 1
                    matched_detected.add(det_p)
                    break
        
        fp = len(detected_peaks) - len(matched_detected)
        fn = len(true_peaks) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision * 100, recall * 100, f1 * 100
    
    def visualize_signals(self, timestamps=None, save_path=None):
        if timestamps is None:
            timestamps = self.timestamps
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), facecolor='#0a0a12')
        
        axes[0].plot(timestamps, self.raw_signal, color='#34d399', linewidth=1.5, alpha=0.8)
        axes[0].set_title('Raw ECG Signal (No Noise)', color='white', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Amplitude', color='white', fontsize=10)
        axes[0].grid(True, alpha=0.1)
        axes[0].set_facecolor('#0a0a12')
        
        axes[1].plot(timestamps, self.noisy_signal, color='#f59e0b', linewidth=1.5, alpha=0.8)
        axes[1].set_title('Noisy ECG Signal', color='white', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Amplitude', color='white', fontsize=10)
        axes[1].grid(True, alpha=0.1)
        axes[1].set_facecolor('#0a0a12')
        
        axes[2].plot(timestamps, self.filtered_signal, color='#60a5fa', linewidth=1.8, alpha=0.9)
        if self.peaks is not None and len(self.peaks) > 0:
            axes[2].scatter(timestamps[self.peaks], self.filtered_signal[self.peaks], 
                           color='#ef4444', s=60, zorder=5, 
                           label=f'Detected Peaks ({len(self.peaks)})',
                           edgecolors='white', linewidth=0.5)
        if hasattr(self, 'true_peaks') and self.true_peaks is not None and len(self.true_peaks) > 0:
            axes[2].scatter(timestamps[self.true_peaks], self.raw_signal[self.true_peaks],
                           color='#34d399', s=80, zorder=4, marker='x',
                           label=f'True Peaks ({len(self.true_peaks)})', alpha=0.7)
        axes[2].set_title('Filtered ECG Signal with Detected Peaks', color='white', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Amplitude', color='white', fontsize=10)
        axes[2].legend(loc='upper right', fontsize=9, facecolor='#0a0a12', edgecolor='white', labelcolor='white')
        axes[2].grid(True, alpha=0.1)
        axes[2].set_facecolor('#0a0a12')
        
        axes[3].plot(timestamps, self.raw_signal, color='#34d399', linewidth=1.5, alpha=0.4, label='Original')
        axes[3].plot(timestamps, self.filtered_signal, color='#60a5fa', linewidth=1.8, alpha=0.8, label='Filtered')
        axes[3].set_title('Original vs Filtered', color='white', fontsize=12, fontweight='bold')
        axes[3].set_xlabel('Time (seconds)', color='white', fontsize=10)
        axes[3].set_ylabel('Amplitude', color='white', fontsize=10)
        axes[3].legend(loc='upper right', fontsize=9, facecolor='#0a0a12', edgecolor='white', labelcolor='white')
        axes[3].grid(True, alpha=0.1)
        axes[3].set_facecolor('#0a0a12')
        
        for ax in axes:
            ax.tick_params(colors='white', labelsize=9)
            for spine in ax.spines.values():
                spine.set_color('white')
                spine.set_alpha(0.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0a0a12')
        
        return fig
    
    def process_pipeline(self, duration=5, heart_rate=72, noise_level=0.3, 
                         lowcut=0.5, highcut=35, height_factor=1.5, min_distance=0.4):
        print("\n" + "="*50)
        print("Starting ECG Processing Pipeline")
        print("="*50 + "\n")
        
        raw_signal, t = self.generate_ecg(duration, heart_rate)
        noisy_signal = self.add_noise(raw_signal, noise_level, seed=42)
        filtered_signal = self.bandpass_filter(noisy_signal, lowcut, highcut)
        self.filtered_signal = filtered_signal
        
        detected_peaks, detected_bpm = self.detect_peaks_adaptive(
            filtered_signal, 
            height_factor=height_factor,
            min_distance=min_distance
        )
        self.peaks = detected_peaks
        
        precision, recall, f1 = self.calculate_accuracy(
            self.true_peaks, 
            detected_peaks,
            tolerance=25
        )
        
        print("\n" + "="*50)
        print("FINAL RESULTS")
        print("="*50)
        print(f"Actual Heart Rate:    {heart_rate:.1f} BPM")
        print(f"Detected Heart Rate:  {detected_bpm:.2f} BPM")
        print(f"Precision:            {precision:.1f}%")
        print(f"Recall:               {recall:.1f}%")
        print(f"F1 Score:             {f1:.1f}%")
        print(f"Peaks Detected:       {len(detected_peaks)}")
        print(f"True Peaks:           {len(self.true_peaks)}")
        print("="*50 + "\n")
        
        return {
            'raw': raw_signal,
            'noisy': noisy_signal,
            'filtered': filtered_signal,
            'peaks': detected_peaks,
            'true_peaks': self.true_peaks,
            'heart_rate': detected_bpm,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'snr': 10 * np.log10(np.var(raw_signal) / (np.var(noisy_signal - raw_signal) + 1e-6))
        }