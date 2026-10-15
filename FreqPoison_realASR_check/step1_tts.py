"""Step 1: synthesise clean speech for every document with three Piper voices."""
import os, json, numpy as np, scipy.signal as ss, soundfile as sf, sherpa_onnx
from math import gcd
from corpus import DOCS, DOC_IDS

M = "/home/claude/models/"
VOICES = {"lessac": "vits-piper-en_US-lessac-medium/en_US-lessac-medium",
          "ryan": "vits-piper-en_US-ryan-medium/en_US-ryan-medium",
          "amy": "vits-piper-en_US-amy-low/en_US-amy-low"}
OUT = "/home/claude/exp/audio/clean"
os.makedirs(OUT, exist_ok=True)
meta = {}
for v, path in VOICES.items():
    d = os.path.dirname(M + path)
    tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(model=M + path + ".onnx", tokens=d + "/tokens.txt",
                                                   data_dir=d + "/espeak-ng-data", noise_scale=0.667,
                                                   noise_scale_w=0.8, length_scale=1.0), num_threads=2)))
    for i in DOC_IDS:
        a = tts.generate(DOCS[i][2], sid=0, speed=1.0)
        x = np.asarray(a.samples, dtype=np.float64)
        g = gcd(16000, a.sample_rate)
        x = ss.resample_poly(x, 16000 // g, a.sample_rate // g)
        x = x * (10 ** (-23 / 20) / np.sqrt(np.mean(x ** 2)))      # -23 dBFS RMS
        sf.write(f"{OUT}/{v}_{i}.wav", x.astype(np.float32), 16000, subtype="FLOAT")
        meta[f"{v}_{i}"] = dict(voice=v, doc=i, sr_tts=a.sample_rate, dur=len(x) / 16000,
                                words=len(DOCS[i][2].split()))
    print(v, "done")
json.dump(meta, open("/home/claude/exp/audio/clean_meta.json", "w"), indent=1)
d = [m["dur"] for m in meta.values()]; w = [m["words"] for m in meta.values()]
print("clips", len(d), "dur %.1f-%.1f s mean %.1f" % (min(d), max(d), np.mean(d)), "words %d-%d" % (min(w), max(w)))
