# FreqPoison

Research code for **FreqPoison: Frequency-Domain Adversarial Poisoning of Audio-RAG Pipelines for Large Language Models**. The project explores how frequency-domain changes to audio documents can affect speech recognition and downstream retrieval in an audio-RAG pipeline.

This repository contains two related experiment tracks:

| Directory | Contents |
|---|---|
| [`FreqPoison_supplement/`](./FreqPoison_supplement/) | Main experiment notebook, simulation and multi-seed analysis scripts, signal-processing helpers, and paper results. |
| [`FreqPoison_realASR_check/`](./FreqPoison_realASR_check/) | Partial real-ASR follow-up using Piper TTS and Whisper models through sherpa-onnx. See its README for progress and details. |

## Quick start: supplementary experiment

The supplementary scripts use Python and the packages listed below. From the project root:

```bash
python -m pip install numpy scipy scikit-learn jiwer matplotlib librosa soundfile
cd FreqPoison_supplement
python sim_extend.py
python make_figs.py
```

`sim_extend.py` runs the extended simulation and writes results to `FreqPoison_supplement/out/`. `make_figs.py` generates figures from those results. The original notebook is [`FreqPoison_Experiment.ipynb`](./FreqPoison_supplement/FreqPoison_Experiment.ipynb) and can also be opened in Google Colab.

## Real-ASR follow-up

The real-ASR scripts synthesize clean clips, apply signal conditions, and transcribe them with sherpa-onnx Whisper models. They require separately downloaded TTS/ASR models and additional packages; consult [`FreqPoison_realASR_check/README.md`](./FreqPoison_realASR_check/README.md) before running them.

**Portability note:** the current real-ASR scripts contain machine-specific `/home/claude/...` paths for model, audio, and result locations. Update those paths for your environment before running the scripts. This track is documented as partial; its README lists completed and outstanding experiments.

## Repository contents and status

- `FreqPoison_supplement/out/results.json` contains supplementary experiment results.
- `FreqPoison_realASR_check/results/clips.jsonl` contains the real-ASR clip results collected so far.
- Generated audio and external model files are not included in this repository.

The supplementary simulation and real-ASR check are distinct experiment tracks; results from one should not be interpreted as results from the other. No license is specified in this repository.
