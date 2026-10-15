"""Step 2: apply each condition to each clean clip, measure the signal, transcribe with Whisper.
Results are appended to results/clips.jsonl (one line per clip x condition x model); reruns resume."""
import os, sys, json, time, numpy as np, soundfile as sf, sherpa_onnx
sys.path.insert(0, "/home/claude/exp")
import fp_signal as fs
from pesq import pesq
from pystoi import stoi
from corpus import DOC_IDS

M = "/home/claude/models/"
RES = "/home/claude/exp/results/clips.jsonl"
os.makedirs(os.path.dirname(RES), exist_ok=True)
VOICES = ["lessac", "ryan", "amy"]


def rng_for(seed, voice, doc):
    return np.random.default_rng(2024 + 1000 * seed + 100 * VOICES.index(voice) + DOC_IDS.index(doc))


def S(seed):  # seeded PRP / CFP helpers
    return lambda x, v, d: fs.prp(x, 0.8, rng=rng_for(seed, v, d))


COND = {
    "clean": lambda x, v, d: x,
    "sei": lambda x, v, d: fs.sei(x, 8.0),
    "hfi": lambda x, v, d: fs.hfi(x, 12.0),
    "canf": lambda x, v, d: fs.canf(x),
    "sei5": lambda x, v, d: fs.sei(x, 5.0),
}
for s in range(3):
    COND[f"prp_s{s}"] = (lambda s: lambda x, v, d: fs.prp(x, 0.8, rng=rng_for(s, v, d)))(s)
    COND[f"cfp_s{s}"] = (lambda s: lambda x, v, d: fs.cfp(x, rng=rng_for(s, v, d)))(s)
# CFP ablation (CFP parameters)
COND["sei5+prp"] = lambda x, v, d: fs.prp(fs.sei(x, 5.0), 0.8, rng=rng_for(0, v, d))
COND["sei5+canf"] = lambda x, v, d: fs.canf(fs.sei(x, 5.0))
COND["prp+canf"] = lambda x, v, d: fs.canf(fs.prp(x, 0.8, rng=rng_for(0, v, d)))
# codec / resampling paths
COND["clean+opus"] = lambda x, v, d: fs.opus_roundtrip(x)
COND["canf+opus"] = lambda x, v, d: fs.opus_roundtrip(fs.canf(x))
COND["clean+g711"] = lambda x, v, d: fs.g711_path(x, True)
COND["canf+g711"] = lambda x, v, d: fs.g711_path(fs.canf(x), True)
COND["hfi+g711"] = lambda x, v, d: fs.g711_path(fs.hfi(x), True)
COND["clean+naive8k"] = lambda x, v, d: fs.g711_path(x, False)
COND["hfi+naive8k"] = lambda x, v, d: fs.g711_path(fs.hfi(x), False)
# SEI gain sweep (adaptive attacker / threshold analysis)
for g in (2, 4, 6, 12):
    COND[f"sei_g{g}"] = (lambda g: lambda x, v, d: fs.sei(x, float(g)))(g)
# matched-SNR white-noise controls are generated from the measured SNR of the attack
NOISE_OF = {"noise_sei": "sei", "noise_hfi": "hfi", "noise_canf": "canf",
            "noise_prp": "prp_s0", "noise_cfp": "cfp_s0"}


def recognizer(name, int8):
    d = M + f"sherpa-onnx-whisper-{name}/"
    q = ".int8" if int8 else ""
    return sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=d + f"{name}-encoder{q}.onnx", decoder=d + f"{name}-decoder{q}.onnx",
        tokens=d + f"{name}-tokens.txt", num_threads=2, decoding_method="greedy_search",
        language="en", task="transcribe")


def transcribe(rec, x):
    s = rec.create_stream(); s.accept_waveform(fs.FS, x); rec.decode_stream(s)
    return s.result.text.strip()


def done_keys():
    if not os.path.exists(RES): return set()
    return {(r["cond"], r["voice"], r["doc"], r["model"]) for r in map(json.loads, open(RES))}


def run(jobs):
    """jobs: list of (cond, voices, model_name, int8)."""
    done = done_keys(); recs = {}
    snr_cache = {}
    if os.path.exists(RES):
        for r in map(json.loads, open(RES)):
            snr_cache[(r["cond"], r["voice"], r["doc"])] = r["snr"]
    with open(RES, "a") as out:
        for cond, voices, mname, int8 in jobs:
            t0 = time.time(); n = 0
            key_m = mname + ("-int8" if int8 else "")
            if key_m not in recs: recs[key_m] = recognizer(mname, int8)
            rec = recs[key_m]
            for v in voices:
                for d in DOC_IDS:
                    if (cond, v, d, key_m) in done: continue
                    x, _ = sf.read(f"/home/claude/exp/audio/clean/{v}_{d}.wav", dtype="float32")
                    if cond in NOISE_OF:
                        snr = snr_cache[(NOISE_OF[cond], v, d)]
                        y = fs.awgn_at_snr(x, snr, rng_for(7, v, d))
                    else:
                        y = COND[cond](x, v, d)
                    y, pk = fs.peak_guard(fs.fit_len(np.asarray(y, np.float32), len(x)))
                    t1 = time.time(); hyp = transcribe(rec, y); t_asr = time.time() - t1
                    r = dict(cond=cond, voice=v, doc=d, model=key_m, hyp=hyp, peak_div=float(pk), t_asr=t_asr)
                    if (cond, v, d) in snr_cache:      # signal metrics only once per clip x condition
                        r["snr"] = snr_cache[(cond, v, d)]
                    else:
                        yc = y * pk                       # undo peak guard for distortion metrics
                        r.update(snr=(None if cond == 'clean' else float(fs.snr_db(x, yc))), lsd_stft=fs.stft_lsd(x, yc), lsd_global=fs.global_lsd(x, yc),
                                 mel_dist=fs.mel_distance(x, yc), env_corr=fs.band_envelope_corr(x, yc),
                                 e_low=fs.band_energy_db(yc, 50, 300) - fs.band_energy_db(x, 50, 300),
                                 e_high=fs.band_energy_db(yc, 6000, 8000) - fs.band_energy_db(x, 6000, 8000),
                                 e_alias=fs.band_energy_db(yc, 0, 2000) - fs.band_energy_db(x, 0, 2000))
                        try: r["pesq_wb"] = float(pesq(fs.FS, x, y, "wb"))
                        except Exception as e: r["pesq_wb"] = None
                        r["stoi"] = float(stoi(x, y, fs.FS, extended=False))
                        t2 = time.time(); feat = fs.sas_features(y); r["sas"] = fs.sas_score(feat)
                        r["t_sas"] = time.time() - t2; r.update({f"sas_{k}": float(v_) for k, v_ in feat.items()})
                        snr_cache[(cond, v, d)] = r["snr"]
                    out.write(json.dumps(r, default=float) + "\n"); out.flush(); n += 1
            print(f"{cond:14s} {key_m:12s} {n:4d} clips {time.time()-t0:6.1f}s", flush=True)


if __name__ == "__main__":
    ALL = VOICES; L = ["lessac"]
    main = ["clean", "sei", "hfi", "canf", "prp_s0", "cfp_s0", "sei5", "prp_s1", "prp_s2", "cfp_s1", "cfp_s2",
            "noise_sei", "noise_hfi", "noise_canf", "noise_prp", "noise_cfp",
            "sei5+prp", "sei5+canf", "prp+canf",
            "clean+opus", "canf+opus", "clean+g711", "canf+g711", "hfi+g711", "clean+naive8k", "hfi+naive8k"]
    jobs = [(c, ALL, "base.en", False) for c in main]
    jobs += [(f"sei_g{g}", L, "base.en", False) for g in (2, 4, 6, 12)]
    for m in ("tiny.en", "small.en"):
        jobs += [(c, L, m, False) for c in ["clean", "sei", "hfi", "canf", "prp_s0", "cfp_s0"]]
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        jobs = [(c, ["lessac"], "base.en", False) for c in ["clean", "prp_s0"]]
    run(jobs)
