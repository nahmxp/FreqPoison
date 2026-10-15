"""FreqPoison signal processing: attack primitives, codec paths, signal metrics, SAS.

All attacks operate on mono float32 audio at FS = 16 kHz, in this fixed order:
  input (TTS 22.05 kHz) -> polyphase resample to 16 kHz -> RMS-normalise to -23 dBFS
  -> attack primitive(s) -> [optional codec / resampling path] -> peak guard -> ASR.
The peak guard divides by max|x|/0.99 only if the peak exceeds 0.99 (records the factor).
"""
import numpy as np, scipy.signal as ss, subprocess, tempfile, os, soundfile as sf, librosa

FS = 16000
NOTCH_F = (350, 700, 1400, 2800)


def rfft_band_mask(n, lo, hi, fs=FS):
    f = np.fft.rfftfreq(n, 1 / fs)
    return (f >= lo) & (f <= hi)


# ---------------------------------------------------------------- attacks
def sei(x, gain_db=8.0, lo=50, hi=300):
    """Sub-band Energy Injection, Eq. (5): whole-utterance FFT gain on [lo, hi] Hz."""
    X = np.fft.rfft(x)
    X[rfft_band_mask(len(x), lo, hi)] *= 10 ** (gain_db / 20)
    return np.fft.irfft(X, n=len(x)).astype(np.float32)


def hfi(x, gain_db=12.0, lo=6000, hi=8000):
    """High-Frequency Injection: whole-utterance FFT gain on [lo, hi] Hz (same form as Eq. 5)."""
    return sei(x, gain_db, lo, hi)


def prp(x, sigma=0.8, lo=1000, hi=4000, rng=None):
    """Phase Randomization Perturbation, Eq. (6): whole-utterance FFT; in [lo, hi] Hz the
    phase is replaced by phi ~ N(0, sigma^2) and the magnitude |X(f)| is kept."""
    rng = np.random.default_rng() if rng is None else rng
    X = np.fft.rfft(x)
    m = rfft_band_mask(len(x), lo, hi)
    X[m] = np.abs(X[m]) * np.exp(1j * rng.normal(0.0, sigma, m.sum()))
    return np.fft.irfft(X, n=len(x)).astype(np.float32)


def canf(x, freqs=NOTCH_F, bw=80.0):
    """Codec-Aware Notch Filtering, Eq. (7): cascade of 2nd-order IIR notches,
    Q = f0 / 80 (i.e. a fixed -3 dB bandwidth of 80 Hz), causal lfilter."""
    y = x.astype(np.float64)
    for f0 in freqs:
        b, a = ss.iirnotch(f0, f0 / bw, fs=FS)
        y = ss.lfilter(b, a, y)
    return y.astype(np.float32)


def cfp(x, rng=None):
    """Combined FreqPoison: SEI(5 dB) -> PRP(sigma=0.8) -> CANF."""
    return canf(prp(sei(x, 5.0), 0.8, rng=rng))


def awgn_at_snr(x, snr_db, rng):
    """Control: white Gaussian noise added at a prescribed SNR."""
    p = np.mean(x.astype(np.float64) ** 2)
    n = rng.normal(0, np.sqrt(p / 10 ** (snr_db / 10)), len(x))
    return (x + n).astype(np.float32)


# ---------------------------------------------------------------- codec / resampling paths
def _ffmpeg(args, x, fs_in=FS):
    with tempfile.TemporaryDirectory() as d:
        i, o = os.path.join(d, "i.wav"), os.path.join(d, "o.wav")
        sf.write(i, x, fs_in, subtype="FLOAT")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", i] + args + [o], check=True)
        y, fs = sf.read(o, dtype="float32")
    return y, fs


def opus_roundtrip(x, kbps=16):
    """Encode with libopus (VBR, application=voip) at `kbps`, decode back to 16 kHz."""
    with tempfile.TemporaryDirectory() as d:
        i, e, o = (os.path.join(d, n) for n in ("i.wav", "e.ogg", "o.wav"))
        sf.write(i, x, FS, subtype="FLOAT")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", i, "-c:a", "libopus", "-b:a", f"{kbps}k",
                        "-application", "voip", e], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", e, "-ar", str(FS), "-ac", "1", o], check=True)
        y, _ = sf.read(o, dtype="float32")
    return fit_len(y, len(x))


def g711_path(x, antialias=True):
    """Telephony path: 16 kHz -> 8 kHz -> G.711 mu-law encode/decode -> 16 kHz.
    antialias=True uses a polyphase low-pass resampler; antialias=False keeps every
    second sample (no low-pass), so content in 4-8 kHz folds into 0-4 kHz."""
    x8 = ss.resample_poly(x, 1, 2) if antialias else x[::2]
    mu = 255.0
    x8 = np.clip(x8, -1, 1)
    q = np.round((np.sign(x8) * np.log1p(mu * np.abs(x8)) / np.log1p(mu)) * 127) / 127  # 8-bit companding
    y8 = np.sign(q) * ((1 + mu) ** np.abs(q) - 1) / mu
    y = ss.resample_poly(y8, 2, 1)
    return fit_len(y.astype(np.float32), len(x))


def fit_len(y, n):
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


def peak_guard(x):
    p = np.max(np.abs(x))
    if p > 0.99:
        return (x * (0.99 / p)).astype(np.float32), p / 0.99
    return x.astype(np.float32), 1.0


# ---------------------------------------------------------------- metrics
def snr_db(x, y):
    return 10 * np.log10(np.sum(x.astype(np.float64) ** 2) / max(np.sum((y - x).astype(np.float64) ** 2), 1e-20))


def stft_lsd(x, y, n_fft=512, hop=128):
    """Frame-averaged log-spectral distance (dB) on the STFT power spectrum."""
    X = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop)) ** 2 + 1e-10
    Y = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2 + 1e-10
    d = 10 * np.log10(X) - 10 * np.log10(Y)
    return float(np.mean(np.sqrt(np.mean(d ** 2, axis=0))))


def global_lsd(x, y):
    """LSD (dB) on the whole-utterance DFT magnitude, as literally written in Eq. (4)."""
    X = np.abs(np.fft.rfft(x)) + 1e-10
    Y = np.abs(np.fft.rfft(y)) + 1e-10
    d = 20 * np.log10(X) - 20 * np.log10(Y)
    return float(np.sqrt(np.mean(d ** 2)))


_MEL = librosa.filters.mel(sr=FS, n_fft=400, n_mels=80, fmax=8000)


def whisper_logmel(x):
    """Whisper front end: 25 ms Hann STFT, 10 ms hop, 80 Slaney mel bands, log10,
    dynamic-range clamp to (max - 8), affine scaling (x + 4) / 4."""
    S = np.abs(librosa.stft(x, n_fft=400, hop_length=160, window="hann", center=True)) ** 2
    L = np.log10(np.maximum(_MEL @ S, 1e-10))
    L = np.maximum(L, L.max() - 8.0)
    return (L + 4.0) / 4.0


def mel_distance(x, y):
    """Mean absolute difference between Whisper log-mel features of x and y."""
    return float(np.mean(np.abs(whisper_logmel(x) - whisper_logmel(y))))


def band_envelope_corr(x, y, lo=1000, hi=4000):
    """Correlation of short-time band energy envelopes (10 ms frames) in [lo, hi] Hz."""
    def env(s):
        S = np.abs(librosa.stft(s, n_fft=400, hop_length=160)) ** 2
        f = librosa.fft_frequencies(sr=FS, n_fft=400)
        return np.log10(S[(f >= lo) & (f <= hi)].sum(0) + 1e-10)
    a, b = env(x), env(y)
    return float(np.corrcoef(a, b)[0, 1])


def band_energy_db(x, lo, hi):
    X = np.abs(np.fft.rfft(x)) ** 2
    return float(10 * np.log10(X[rfft_band_mask(len(x), lo, hi)].sum() + 1e-20))


# ---------------------------------------------------------------- SAS defence, Eq. (9)
SAS_W = dict(alpha=3.0, beta=5.0, gamma=2.0, delta=2.0)
SAS_T = dict(low=0.15, high=0.05, sf=0.30)


def sas_features(x):
    P = np.abs(np.fft.rfft(x.astype(np.float64))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / FS)
    tot = P[(f >= 20)].sum()
    r_low = P[(f >= 20) & (f < 300)].sum() / tot
    r_high = P[f > 6000].sum() / tot
    band = P[(f >= 50) & (f <= 7900)] + 1e-20
    sfl = np.exp(np.mean(np.log(band))) / np.mean(band)
    voids = []
    for f0 in NOTCH_F:   # notch void: power within +-40 Hz vs. flanks 100-200 Hz away
        c = P[np.abs(f - f0) <= 40].mean()
        fl = P[(np.abs(f - f0) >= 100) & (np.abs(f - f0) <= 200)].mean()
        voids.append(max(0.0, 1.0 - c / fl))
    return dict(r_low=r_low, r_high=r_high, sf=sfl, n_score=float(np.mean(voids)))


def sas_score(feat):
    w, t = SAS_W, SAS_T
    return (w["alpha"] * max(0, feat["r_low"] - t["low"]) + w["beta"] * max(0, feat["r_high"] - t["high"])
            + w["gamma"] * max(0, feat["sf"] - t["sf"]) + w["delta"] * feat["n_score"])
