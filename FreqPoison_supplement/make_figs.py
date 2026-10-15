"""Regenerate the paper figures from the notebook code (seed 2024) and the 20-seed runs."""
import json, os, pickle, random, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nbrun, fp_signal as fs

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
                     "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 0.8,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
ns = nbrun.load(upto=7)
R = json.load(open("out/results.json"))
P = pickle.load(open("out/runs.pkl", "rb"))
ATT = ["CLEAN", "SEI", "PRP", "CANF", "HFI", "CFP"]
COL = {"CLEAN": "#4d4d4d", "SEI": "#e08214", "PRP": "#b2182b", "CANF": "#2166ac", "HFI": "#7b3294", "CFP": "#1b7837"}
OUT = "out/"
W = 4.6  # inches, svmult text width ~ 11.7 cm

# ---------------------------------------------------------------- Fig. 2: spectra (seed 2024, D01)
np.random.seed(2024); random.seed(2024)
x = ns["synthesize_speech"](ns["CORPUS"][0]["text"])
ys = {"CLEAN": x}
for a in ATT[1:]:
    np.random.seed(2024)  # PRP draws from the global generator
    ys[a] = ns["ATTACK_FNS"][a](x)[0]
fig, axs = plt.subplots(2, 3, figsize=(W, 2.9), sharex=True, sharey=True)
f = np.fft.rfftfreq(len(x), 1 / 16000) / 1000


def smooth_db(sig):
    m = 20 * np.log10(np.abs(np.fft.rfft(sig)) + 1e-8)
    k = 41
    return np.convolve(m, np.ones(k) / k, mode="same")


ref = smooth_db(x)
bands = {"SEI": [(0.05, 0.3)], "PRP": [(1, 4)], "HFI": [(6, 8)], "CFP": [(0.05, 0.3), (1, 4)]}
for ax, a in zip(axs.flat, ATT):
    if a != "CLEAN":
        ax.plot(f, ref, color="#b0b0b0", lw=0.6, label="clean")
    ax.plot(f, smooth_db(ys[a]), color=COL[a], lw=0.7, label=a)
    for lo, hi in bands.get(a, []):
        ax.axvspan(lo, hi, color=COL[a], alpha=0.10, lw=0)
    if a in ("CANF", "CFP"):
        for fn in (0.35, 0.7, 1.4, 2.8):
            ax.axvline(fn, color=COL[a], lw=0.4, ls=":")
    r = R["seed2024"][a]
    ax.set_title(a, pad=2)
    if a != "CLEAN":
        ax.text(0.97, 0.95, f"SNR {r['SNR']:.1f} dB\nLSD {r['LSD']:.2f} dB", transform=ax.transAxes, ha="right", va="top",
                fontsize=6.5, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", lw=0.4))
    ax.set_xlim(0, 8); ax.set_ylim(-30, 70); ax.grid(alpha=0.25, lw=0.4)
for ax in axs[1]: ax.set_xlabel("Frequency (kHz)")
for ax in axs[:, 0]: ax.set_ylabel("Magnitude (dB)")
fig.tight_layout(pad=0.3)
fig.savefig(OUT + "fig2_spectra.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig. 3: Whisper log-mel input (level matched)
fig, axs = plt.subplots(1, 3, figsize=(W, 1.45), sharey=True)
for ax, a in zip(axs, ["CLEAN", "PRP", "CFP"]):
    y = ys[a].astype(np.float64); y = (y * np.std(x) / np.std(y)).astype(np.float32)
    L = fs.whisper_logmel(y)
    t = np.arange(L.shape[1]) / 100
    ax.imshow(L, origin="lower", aspect="auto", extent=[0, t[-1], 0, 80], cmap="magma", vmin=-1.2, vmax=1.5)
    ax.set_title(a + (" (level-matched)" if a != "CLEAN" else ""), pad=2)
    ax.set_xlabel("Time (s)")
axs[0].set_ylabel("Mel band")
fig.tight_layout(pad=0.3)
fig.savefig(OUT + "fig3_logmel.pdf", dpi=300); plt.close(fig)

# ---------------------------------------------------------------- Fig. 4: results, 20-seed mean +- 95% CI, seed 2024 marker
fig, axs = plt.subplots(1, 4, figsize=(W, 1.55))
for ax, (k, lab) in zip(axs, [("WER", "WER"), ("P1", "P@1"), ("P3", "P@3"), ("nDCG", "nDCG@3")]):
    m = [R["multiseed"][a][k][0] for a in ATT]
    lo = [R["multiseed"][a][k][0] - R["multiseed"][a][k][1] for a in ATT]
    s24 = [R["seed2024"][a][k] for a in ATT]
    ax.bar(range(6), m, yerr=lo, color=[COL[a] for a in ATT], width=0.65, error_kw=dict(lw=0.6, capsize=1.5))
    ax.plot(range(6), s24, "kD", ms=2.2, label="seed 2024")
    ax.axhline(m[0], color="k", lw=0.5, ls="--")
    ax.set_xticks(range(6)); ax.set_xticklabels(ATT, rotation=60, fontsize=6)
    ax.set_title(lab, pad=2); ax.set_ylim(0, 0.6 if k == "WER" else 1.0); ax.grid(axis="y", alpha=0.25, lw=0.4)
axs[0].legend(loc="upper left", frameon=False, handletextpad=0.2, borderaxespad=0.1)
fig.tight_layout(pad=0.3, w_pad=0.4)
fig.savefig(OUT + "fig4_results.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig. 5: stage-wise analysis
rows = P["rows"]
fig, axs = plt.subplots(1, 3, figsize=(W, 1.65))
for a in ATT[1:]:
    rr = [r for r in rows if r["attack"] == a]
    axs[0].scatter([r["snr"] for r in rr], [r["wer"] for r in rr], s=1.2, color=COL[a], alpha=0.35, lw=0, label=a)
    axs[1].scatter([r["wer"] for r in rr], [r["drift"] for r in rr], s=1.2, color=COL[a], alpha=0.35, lw=0)
axs[0].set_xlabel("SNR (dB)"); axs[0].set_ylabel("WER (per document)")
axs[0].set_title("(a) SNR vs. WER", pad=2)
axs[1].set_xlabel("WER (per document)"); axs[1].set_ylabel("Drift")
axs[1].set_title("(b) WER vs. drift", pad=2)
d = R["dose_response"]; ks = sorted(d, key=float)
wv = [d[k]["wer"][0] for k in ks]
for key, lab, c in [("p3", "P@3", "#2166ac"), ("ndcg", "nDCG@3", "#b2182b")]:
    m = np.array([d[k][key][0] for k in ks]); l = np.array([d[k][key][1] for k in ks]); h = np.array([d[k][key][2] for k in ks])
    axs[2].plot(wv, m, "o-", ms=2, color=c, label=lab); axs[2].fill_between(wv, l, h, color=c, alpha=0.15, lw=0)
axs[2].set_xlabel("WER (text-only corruption)"); axs[2].set_title("(c) Text-stage intervention", pad=2)
axs[2].legend(frameon=False, loc="lower left"); axs[2].set_ylim(0.4, 0.75)
leg = axs[0].legend(frameon=False, markerscale=3, fontsize=5.5, loc="upper right", handletextpad=0.1, borderaxespad=0.1)
for lh in leg.legend_handles: lh.set_alpha(1)
for ax in axs: ax.grid(alpha=0.25, lw=0.4)
fig.tight_layout(pad=0.3, w_pad=0.5)
fig.savefig(OUT + "fig5_stages.pdf", dpi=300); plt.close(fig)
print("figures written")
