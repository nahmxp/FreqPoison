"""Extended analysis of the FreqPoison simulation (revision for the Springer version).

Uses the notebook's own functions (loaded verbatim via nbrun.py) and adds:
  A. multi-seed repetition (20 seeds) with 95% confidence intervals
  B. signal-level checks: gain effects, PRP, Whisper log-mel input, codec and resampling paths
  C. stage-wise analysis of the attack chain and a keyword-error analysis
  D. defence evaluation (SAS, TCS) with a calibration/test split and baseline detectors
All outputs go to out/results.json; figures to out/*.pdf|png.
"""
import json, math, random, time, sys, re, os
import numpy as np, scipy.stats as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nbrun
import fp_signal as fs

ns = nbrun.load(upto=7)          # environment, corpus, synthesis, metrics, attacks, ASR sim, RAG
CORPUS, QUERIES = ns["CORPUS"], ns["QUERIES"]
ATTACKS = ["CLEAN", "SEI", "PRP", "CANF", "HFI", "CFP"]
SR = ns["SR"]
OUT = {}
SEEDS = list(range(2024, 2044))   # 20 seeds; 2024 is the notebook's seed


# ------------------------------------------------------------------ A. experiment loop (as notebook cell 16)
def run(seed, keep_audio=False):
    clean_texts = [d["text"] for d in CORPUS]
    doc_ids = [d["id"] for d in CORPUS]
    res, audio_store = {}, {}
    for attack in ATTACKS:
        np.random.seed(seed); random.seed(seed)
        poisoned, wers, snrs, lsds = [], [], [], []
        for doc in CORPUS:
            audio = ns["synthesize_speech"](doc["text"])
            if attack == "CLEAN":
                meta, y = {"snr": 35.0, "lsd": 0.0}, audio
            else:
                y, meta = ns["ATTACK_FNS"][attack](audio)
            if keep_audio: audio_store[(attack, doc["id"])] = (audio, y)
            tr = ns["simulate_asr_transcription"](doc["text"], attack, meta["snr"])
            poisoned.append(tr)
            wers.append(ns["compute_wer"](doc["text"].lower(), tr.lower()))
            snrs.append(meta["snr"]); lsds.append(meta["lsd"])
        vec, Xc, Xp = ns["build_shared_tfidf"](clean_texts, poisoned)
        drifts = [ns["semantic_drift"](Xc[i].toarray()[0], Xp[i].toarray()[0]) for i in range(len(CORPUS))]
        q = {}
        for qq in QUERIES:
            r = ns["retrieve_topk"](qq["text"], vec, Xp, doc_ids, k=5)
            q[qq["id"]] = dict(ranked=[d for d, _ in r],
                               p1=ns["precision_at_k"](r, qq["relevant"], 1), p3=ns["precision_at_k"](r, qq["relevant"], 3),
                               r3=ns["recall_at_k"](r, qq["relevant"], 3), ndcg=ns["ndcg_at_k"](r, qq["relevant"], 3))
        res[attack] = dict(transcripts=poisoned, wer=wers, snr=snrs, lsd=lsds, drift=drifts, q=q,
                           pesq=[ns["pesq_proxy"](s) for s in snrs])
    return (res, audio_store) if keep_audio else res


def summary(r):
    qs = r["q"].values()
    return dict(WER=np.mean(r["wer"]), WER_sd=np.std(r["wer"]), SNR=np.mean(r["snr"]), LSD=np.mean(r["lsd"]),
                PESQ=np.mean(r["pesq"]), Drift=np.mean(r["drift"]),
                P1=np.mean([x["p1"] for x in qs]), P3=np.mean([x["p3"] for x in qs]),
                R3=np.mean([x["r3"] for x in qs]), nDCG=np.mean([x["ndcg"] for x in qs]))


t0 = time.time()
RUNS = {s: run(s) for s in SEEDS}
OUT["runtime_20_seeds_s"] = time.time() - t0
S = {s: {a: summary(RUNS[s][a]) for a in ATTACKS} for s in SEEDS}
OUT["seed2024"] = {a: {k: float(v) for k, v in S[2024][a].items()} for a in ATTACKS}


def ci(v):
    v = np.asarray(v, float); m = v.mean()
    h = st.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else 0
    return [float(m), float(m - h), float(m + h)]


OUT["multiseed"] = {a: {k: ci([S[s][a][k] for s in SEEDS]) for k in S[2024][a]} for a in ATTACKS}
# paired differences vs CLEAN, per seed
OUT["multiseed_delta"] = {}
for a in ATTACKS[1:]:
    d = {}
    for k in ["WER", "P1", "P3", "R3", "nDCG", "Drift"]:
        diffs = [S[s][a][k] - S[s]["CLEAN"][k] for s in SEEDS]
        d[k] = ci(diffs) + [int(sum(x < 0 for x in diffs)), int(sum(x == 0 for x in diffs)), int(sum(x > 0 for x in diffs))]
    OUT["multiseed_delta"][a] = d

# ------------------------------------------------------------------ additivity of WER increases (seed-averaged)
dW = {a: OUT["multiseed_delta"][a]["WER"][0] for a in ATTACKS[1:]}
p = [S[s]["CLEAN"]["WER"] for s in SEEDS]
base = float(np.mean(p))
indep = 1 - np.prod([1 - (OUT["multiseed"][a]["WER"][0]) for a in ["SEI", "PRP", "CANF"]])
OUT["additivity"] = dict(sum_components_pp=100 * (dW["SEI"] + dW["PRP"] + dW["CANF"]), cfp_pp=100 * dW["CFP"],
                         max_component_pp=100 * max(dW["SEI"], dW["PRP"], dW["CANF"]),
                         independent_union_wer=float(indep), independent_union_pp=100 * (float(indep) - base),
                         cfp_wer=OUT["multiseed"]["CFP"]["WER"][0])

# ------------------------------------------------------------------ explain r_base vs measured WER
# expected per-word edit probability of the sampler for CLEAN (r_eff = r_base since SNR = 35 dB > 25 dB):
words = [w for d in CORPUS for w in d["text"].split()]
cw = [w.strip(".,;:").lower() for w in words]
rb = ns["BASE_WER_BY_ATTACK"]["CLEAN"]
# deletion fires for r < 0.60 r (when word not in map) ... the code's branch structure gives:
exp_edit = np.mean([(0.60 * rb if True else 0) + 0.15 * rb + (0.10 * rb if len(c) > 4 else 0) for c in cw])
OUT["clean_rate_explanation"] = dict(r_base=rb, n_words=len(words), expected_edit_prob=float(exp_edit),
                                     mean_wer_20_seeds=OUT["multiseed"]["CLEAN"]["WER"])

# ------------------------------------------------------------------ B. signal-level checks (seed 2024 audio)
res24, AUD = run(2024, keep_audio=True)


def gm_snr(x, y):
    g = np.dot(x, y) / np.dot(y, y)
    return float(10 * np.log10(np.sum(x ** 2) / np.sum((x - g * y) ** 2))), float(20 * np.log10(abs(g)))


sig = {}
for a in ATTACKS[1:]:
    rows = []
    for d in CORPUS:
        x, y = AUD[(a, d["id"])]
        x = x.astype(np.float64); y = y.astype(np.float64)
        s_gm, gdb = gm_snr(x, y)
        Lx, Ly = fs.whisper_logmel(x.astype(np.float32)), fs.whisper_logmel(y.astype(np.float32))
        Lxr, Lyr = fs.whisper_logmel(x.astype(np.float32)), fs.whisper_logmel((y * np.std(x) / np.std(y)).astype(np.float32))
        X = np.abs(np.fft.rfft(x)); Y = np.abs(np.fft.rfft(y)); f = np.fft.rfftfreq(len(x), 1 / SR)
        band = (f >= 1000) & (f <= 4000)
        rows.append(dict(snr=ns["snr_db"](x, y), snr_gm=s_gm, level_db=float(20 * np.log10(np.std(y) / np.std(x))),
                         crest_db=float(20 * np.log10(np.max(np.abs(y)) / np.std(y))),
                         crest_clean_db=float(20 * np.log10(np.max(np.abs(x)) / np.std(x))),
                         mel_mad=float(np.mean(np.abs(Lx - Ly))), mel_mad_levelmatched=float(np.mean(np.abs(Lxr - Lyr))),
                         band_mag_corr=float(np.corrcoef(X[band], Y[band])[0, 1]),
                         env_corr=fs.band_envelope_corr(x.astype(np.float32), y.astype(np.float32)),
                         d_low=fs.band_energy_db(y, 50, 300) - fs.band_energy_db(x, 50, 300) - 20 * np.log10(np.std(y) / np.std(x)),
                         d_high=fs.band_energy_db(y, 6000, 8000) - fs.band_energy_db(x, 6000, 8000) - 20 * np.log10(np.std(y) / np.std(x))))
    sig[a] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
OUT["signal_checks"] = sig

# mel filters by band (Whisper: 80 Slaney mel bands, 0-8 kHz, n_fft 400)
cent = (fs._MEL * np.linspace(0, 8000, 201)).sum(1) / fs._MEL.sum(1)
OUT["mel_filters"] = dict(below_300=int((cent < 300).sum()), from_6000=int((cent >= 6000).sum()),
                          between_1000_4000=int(((cent >= 1000) & (cent <= 4000)).sum()))

# codec and resampling paths (seed 2024, all 15 docs)
def bandpow(x, f0, w=40):
    X = np.abs(np.fft.rfft(x)) ** 2; f = np.fft.rfftfreq(len(x), 1 / SR)
    return X[np.abs(f - f0) <= w].sum()


codec = {}
for path, fn in [("direct", lambda z: z), ("opus16k", lambda z: fs.opus_roundtrip(z, 16)),
                 ("g711", lambda z: fs.g711_path(z, True))]:
    depth = {f0: [] for f0 in fs.NOTCH_F}
    for d in CORPUS:
        x, y = AUD[("CANF", d["id"])]
        x = x.astype(np.float32); y = (y * np.std(x) / np.std(y)).astype(np.float32)   # level-matched
        xp, yp = fn(x), fn(y)
        for f0 in fs.NOTCH_F:
            depth[f0].append(10 * np.log10(bandpow(yp, f0) / bandpow(xp, f0)))
    codec[path] = {str(k): float(np.mean(v)) for k, v in depth.items()}
OUT["canf_notch_depth_db"] = codec

alias = {}
for path, aa in [("antialiased", True), ("naive_decimation", False)]:
    vals = []
    for d in CORPUS:
        x, y = AUD[("HFI", d["id"])]
        x = x.astype(np.float32); y = y.astype(np.float32)
        xp, yp = fs.g711_path(x, aa), fs.g711_path(y, aa)
        vals.append(fs.band_energy_db(yp, 0, 2000) - fs.band_energy_db(xp, 0, 2000))
    alias[path] = float(np.mean(vals))
x, y = AUD[("HFI", CORPUS[0]["id"])]
alias["direct_16k_0_2k_change_db"] = float(np.mean([fs.band_energy_db(AUD[("HFI", d["id"])][1], 0, 2000) -
                                                   fs.band_energy_db(AUD[("HFI", d["id"])][0], 0, 2000) for d in CORPUS]))
alias["hf_band_share_clean_pct"] = float(np.mean([100 * 10 ** ((fs.band_energy_db(AUD[("HFI", d["id"])][0], 6000, 8000)
                                       - fs.band_energy_db(AUD[("HFI", d["id"])][0], 0, 8000)) / 10) for d in CORPUS]))
OUT["hfi_alias_0_2k_change_db"] = alias

# ------------------------------------------------------------------ C. stage-wise analysis
STOP = set("what are the and is before an or how must be when a for exist".split())
qterms = set(w for q in QUERIES for w in re.findall(r"[a-z0-9]+", q["text"].lower()) if w not in STOP)
OUT["query_terms"] = sorted(qterms)


def kw_counts(t):
    toks = re.findall(r"[a-z0-9]+", t.lower())
    return {k: toks.count(k) for k in qterms}


rows = []          # per doc x attack x seed
runrows = []       # per attack x seed
for s in SEEDS:
    for a in ATTACKS[1:]:
        r, c = RUNS[s][a], RUNS[s]["CLEAN"]
        kerr = []
        for i, d in enumerate(CORPUS):
            ref, hyp = kw_counts(d["text"]), kw_counts(r["transcripts"][i])
            nref = sum(ref.values())
            lost = sum(max(0, ref[k] - hyp[k]) for k in qterms)
            ker = lost / nref if nref else 0.0
            kerr.append((lost, nref))
            rows.append(dict(seed=s, attack=a, wer=r["wer"][i], drift=r["drift"][i], snr=r["snr"][i], ker=ker))
        for qq in QUERIES:
            rel = qq["relevant"]
            idx = [k for k, d in enumerate(CORPUS) if d["id"] in rel]
            runrows.append(dict(seed=s, attack=a, q=qq["id"],
                                d_ndcg=r["q"][qq["id"]]["ndcg"] - c["q"][qq["id"]]["ndcg"],
                                d_p3=r["q"][qq["id"]]["p3"] - c["q"][qq["id"]]["p3"],
                                rel_drift=float(np.mean([r["drift"][k] for k in idx])),
                                rel_wer=float(np.mean([r["wer"][k] for k in idx])),
                                rel_ker=float(np.mean([rows[-15 + k]["ker"] for k in idx]))))
W = np.array([x["wer"] for x in rows]); D = np.array([x["drift"] for x in rows]); SN = np.array([x["snr"] for x in rows])
K = np.array([x["ker"] for x in rows])
lab = np.array([x["attack"] for x in rows])
OUT["stage"] = {}
OUT["stage"]["snr_wer_spearman_pooled"] = float(st.spearmanr(SN, W)[0])
OUT["stage"]["snr_wer_spearman_within_attack"] = {a: float(st.spearmanr(SN[lab == a], W[lab == a])[0]) for a in ATTACKS[1:]}
# share of WER variance explained by the attack label alone (eta^2)
grand = W.mean(); ss_tot = ((W - grand) ** 2).sum()
ss_b = sum((lab == a).sum() * (W[lab == a].mean() - grand) ** 2 for a in ATTACKS[1:])
OUT["stage"]["wer_eta2_attack_label"] = float(ss_b / ss_tot)
OUT["stage"]["wer_drift_spearman_pooled"] = float(st.spearmanr(W, D)[0])
OUT["stage"]["wer_drift_spearman_within_attack"] = {a: float(st.spearmanr(W[lab == a], D[lab == a])[0]) for a in ATTACKS[1:]}
RR = runrows
dn = np.array([x["d_ndcg"] for x in RR]); rd = np.array([x["rel_drift"] for x in RR])
rw = np.array([x["rel_wer"] for x in RR]); rk = np.array([x["rel_ker"] for x in RR])
OUT["stage"]["reldrift_dndcg_spearman"] = float(st.spearmanr(rd, dn)[0])
OUT["stage"]["relwer_dndcg_spearman"] = float(st.spearmanr(rw, dn)[0])
OUT["stage"]["relker_dndcg_spearman"] = float(st.spearmanr(rk, dn)[0])
OUT["stage"]["frac_query_cases_unchanged_ndcg"] = float(np.mean(np.abs(dn) < 1e-9))
OUT["stage"]["frac_query_cases_improved_ndcg"] = float(np.mean(dn > 1e-9))
OUT["stage"]["frac_query_cases_degraded_ndcg"] = float(np.mean(dn < -1e-9))
OUT["stage"]["per_attack_query_cases"] = {a: dict(degraded=float(np.mean(dn[[x["attack"] == a for x in RR]] < -1e-9)),
                                                  improved=float(np.mean(dn[[x["attack"] == a for x in RR]] > 1e-9)))
                                          for a in ATTACKS[1:]}
# partial association: within-WER-bin keyword error
OUT["stage"]["ker_mean_by_attack"] = {a: float(K[lab == a].mean()) for a in ATTACKS[1:]}

# controlled text-only intervention: corrupt clean text at fixed per-word rate, no audio involved
dose = {}
clean_texts = [d["text"] for d in CORPUS]; doc_ids = [d["id"] for d in CORPUS]
for rate in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    vals = []
    for s in SEEDS:
        random.seed(s)
        ns["BASE_WER_BY_ATTACK"]["_DOSE"] = rate
        tr = [ns["simulate_asr_transcription"](d["text"], "_DOSE", 35.0) for d in CORPUS]
        vec, Xc, Xp = ns["build_shared_tfidf"](clean_texts, tr)
        w = np.mean([ns["compute_wer"](d["text"].lower(), t.lower()) for d, t in zip(CORPUS, tr)])
        p3 = np.mean([ns["precision_at_k"](ns["retrieve_topk"](q["text"], vec, Xp, doc_ids, 5), q["relevant"], 3) for q in QUERIES])
        nd = np.mean([ns["ndcg_at_k"](ns["retrieve_topk"](q["text"], vec, Xp, doc_ids, 5), q["relevant"], 3) for q in QUERIES])
        vals.append((w, p3, nd))
    v = np.array(vals)
    dose[str(rate)] = dict(wer=ci(v[:, 0]), p3=ci(v[:, 1]), ndcg=ci(v[:, 2]))
OUT["dose_response"] = dose

# ------------------------------------------------------------------ D. defence evaluation
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
CAL, TEST = SEEDS[:10], SEEDS[10:]
import io, contextlib
ns["ATTACKS"] = ATTACKS
for src in nbrun.CODE[12:14]:
    with contextlib.redirect_stdout(io.StringIO()):
        exec(src, ns)
sasf = ns["spectral_anomaly_score"]; tcsf = ns["transcript_consistency_score"]
feat = {}   # (seed, attack, doc) -> dict
for s in SEEDS:
    for a in ATTACKS:
        np.random.seed(s); random.seed(s)
        for i, d in enumerate(CORPUS):
            x = ns["synthesize_speech"](d["text"])
            y = x if a == "CLEAN" else ns["ATTACK_FNS"][a](x)[0]
            t1 = time.time(); f_ = sasf(y); tsas = time.time() - t1
            meta_txt = re.split(r"(?<=\.)\s", d["text"])[0]          # metadata: first sentence
            t2 = time.time(); tcs = tcsf(meta_txt, RUNS[s][a]["transcripts"][i]); ttcs = time.time() - t2
            vocab = set(re.findall(r"[a-z0-9]+", " ".join(clean_texts).lower()))
            toks = re.findall(r"[a-z0-9]+", RUNS[s][a]["transcripts"][i].lower())
            feat[(s, a, d["id"])] = dict(sas=f_["anomaly_score"], low=f_["low_ratio"], high=f_["high_ratio"],
                                         flat=f_["flatness"], notch=f_["notch_score"],
                                         rms=float(20 * np.log10(np.std(y) + 1e-12)), tcs=tcs,
                                         oov=float(np.mean([t not in vocab for t in toks])) if toks else 1.0,
                                         t_sas=tsas, t_tcs=ttcs)
OUT["defense_cost_ms"] = dict(sas=1000 * float(np.mean([v["t_sas"] for v in feat.values()])),
                              tcs=1000 * float(np.mean([v["t_tcs"] for v in feat.values()])))


def get(seeds, a, k):
    return np.array([feat[(s, a, d["id"])][k] for s in seeds for d in CORPUS])


def detector_scores(seeds, a, name, iso=None, med_rms=None):
    if name == "SAS": return get(seeds, a, "sas")
    if name == "Flatness": return get(seeds, a, "flat")
    if name == "Energy": return np.abs(get(seeds, a, "rms") - med_rms)
    if name == "IsoForest":
        X = np.c_[get(seeds, a, "low"), get(seeds, a, "high"), get(seeds, a, "flat"), get(seeds, a, "notch"), get(seeds, a, "rms")]
        return -iso.score_samples(X)
    if name == "TCS": return -get(seeds, a, "tcs")           # low consistency = suspicious
    if name == "OOV": return get(seeds, a, "oov")


Xcal = np.c_[get(CAL, "CLEAN", "low"), get(CAL, "CLEAN", "high"), get(CAL, "CLEAN", "flat"), get(CAL, "CLEAN", "notch"), get(CAL, "CLEAN", "rms")]
iso = IsolationForest(n_estimators=200, random_state=0).fit(Xcal)
med_rms = float(np.median(get(CAL, "CLEAN", "rms")))
det = {}
for name in ["SAS", "Energy", "Flatness", "IsoForest", "TCS", "OOV"]:
    c_cal = detector_scores(CAL, "CLEAN", name, iso, med_rms); c_test = detector_scores(TEST, "CLEAN", name, iso, med_rms)
    thr = float(np.quantile(c_cal, 0.95))               # 5% FPR on calibration clean
    row = dict(thr_cal=thr, fpr_test_cal=float(np.mean(c_test > thr)))
    if name == "SAS": row["fpr_fixed"] = float(np.mean(c_test > 0.05))
    if name == "TCS": row["fpr_fixed"] = float(np.mean(-c_test < 0.40))
    for a in ATTACKS[1:]:
        at = detector_scores(TEST, a, name, iso, med_rms)
        yv = np.r_[np.zeros(len(c_test)), np.ones(len(at))]; sv = np.r_[c_test, at]
        row[a] = dict(auroc=float(roc_auc_score(yv, sv)), tpr_cal=float(np.mean(at > thr)))
        if name == "SAS": row[a]["tpr_fixed"] = float(np.mean(at > 0.05))
        if name == "TCS": row[a]["tpr_fixed"] = float(np.mean(-at < 0.40))
    det[name] = row
OUT["detection"] = det
OUT["sas_clean_stats"] = dict(mean=float(get(TEST, "CLEAN", "sas").mean()), min=float(get(TEST, "CLEAN", "sas").min()),
                              low_ratio_mean=float(get(TEST, "CLEAN", "low").mean()))
# threshold sensitivity for SAS (test split, all attacks pooled)
c_test = get(TEST, "CLEAN", "sas"); allatk = np.concatenate([get(TEST, a, "sas") for a in ATTACKS[1:]])
OUT["sas_threshold_sweep"] = [dict(thr=float(t), fpr=float(np.mean(c_test > t)), tpr=float(np.mean(allatk > t)))
                              for t in np.quantile(c_test, [0.5, 0.8, 0.9, 0.95, 0.99, 1.0])]

# adaptive attacker: SEI gain sweep against the calibrated SAS threshold
thr = det["SAS"]["thr_cal"]; sweep = []
for g in [1, 2, 3, 4, 5, 6, 8]:
    sc, wv = [], []
    for s in TEST:
        np.random.seed(s); random.seed(s)
        for d in CORPUS:
            x = ns["synthesize_speech"](d["text"])
            y, m = ns["attack_SEI"](x, gain_db=g)
            sc.append(sasf(y)["anomaly_score"])
            tr = ns["simulate_asr_transcription"](d["text"], "SEI", m["snr"])
            wv.append(ns["compute_wer"](d["text"].lower(), tr.lower()))
    sweep.append(dict(gain_db=g, tpr=float(np.mean(np.array(sc) > thr)), sim_wer=float(np.mean(wv))))
OUT["sei_gain_sweep"] = sweep

json.dump(OUT, open("out/results.json", "w"), indent=1, default=float)
import pickle; pickle.dump(dict(RUNS=RUNS, rows=rows, runrows=runrows), open("out/runs.pkl", "wb"))
print("done in %.1f s" % (time.time() - t0))
