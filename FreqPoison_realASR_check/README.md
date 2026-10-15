# Real-ASR check (partial, for the journal follow-up)

`step1_tts.py` synthesizes 60 policy statements (`corpus.py`, 20 queries with relevance labels) with three Piper voices.
`step2_asr.py` applies each condition and transcribes with Whisper (sherpa-onnx ONNX exports of tiny.en, base.en and small.en). It records SNR, frame LSD, PESQ-WB, STOI, Whisper log-mel distance and SAS features per clip in `results/clips.jsonl`, and resumes where it left off.
`peek.py` prints WER per condition.

Done so far (base.en): clean, SEI, HFI, CANF (3 voices), plus PRP and CFP (lessac and ryan).
Not run yet: the other seeds, noise controls, ablations, codec paths, the SEI sweep, tiny/small models, and the retrieval evaluation.
Models come from GitHub releases at k2-fsa/sherpa-onnx (asr-models, tts-models).
