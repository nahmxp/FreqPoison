# FreqPoison: supplementary code

Everything needed to reproduce the numbers and figures in the paper.

| File | What it does |
|---|---|
| `FreqPoison_Experiment.ipynb` | The original Colab notebook (seed 2024 gives Table 3 exactly). |
| `nbrun.py` | Runs the notebook's code cells as is (no edits), so the scripts below use the same functions. |
| `sim_extend.py` | 20 seeds with 95% CIs (Table 4), signal checks (Table 5), stage-wise analysis, text-only intervention, SAS/TCS evaluation with baselines (Table 6), SEI gain sweep. Writes `out/results.json` and `out/runs.pkl`. About 5 minutes on a laptop CPU. |
| `make_figs.py` | Builds Figs. 2 to 4 from the outputs above. |
| `fp_signal.py` | Helper signal functions: Whisper-style log-mel, Opus and G.711 round trips (needs `ffmpeg` with libopus), band energies. |
| `out/results.json` | The results used in the paper. |

Install the dependencies, then run the scripts in order:

```
pip install numpy scipy scikit-learn jiwer matplotlib librosa soundfile
python sim_extend.py
python make_figs.py
```
