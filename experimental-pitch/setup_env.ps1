# Creates experimental-pitch\.venv (Python 3.11, CPU torch) isolated from the app's .venv.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
uv venv --python 3.11 .venv
$py = ".venv\Scripts\python.exe"
uv pip install --python $py torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python $py "basic-pitch[onnx]" torchcrepe pesto-pitch piano_transcription_inference onnx librosa pretty_midi mir_eval matplotlib soundfile mido
# basic-pitch pulls TensorFlow 2.14, which crashes with NumPy 2; we use its ONNX model instead.
uv pip uninstall --python $py tensorflow tensorflow-intel tensorflow-estimator tensorboard tensorflow-io-gcs-filesystem

# piano_transcription_inference downloads its checkpoint with wget, which Windows lacks.
$ckDir = Join-Path $env:USERPROFILE "piano_transcription_inference_data"
$ck = Join-Path $ckDir "note_F1=0.9677_pedal_F1=0.9186.pth"
if (-not (Test-Path $ck)) {
    New-Item -ItemType Directory -Force $ckDir | Out-Null
    curl.exe -L -o $ck "https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
}

# ---- drums ----
New-Item -ItemType Directory -Force third_party | Out-Null
if (-not (Test-Path third_party\ADTOF-pytorch)) {
    git clone https://github.com/xavriley/ADTOF-pytorch third_party\ADTOF-pytorch
}
uv pip install --python $py -e third_party\ADTOF-pytorch libnmfd

# YourMT3+ lives in its own venv: mt3-infer needs transformers 4.45.x
# (5.x lacks model_parallel_utils, 4.49 breaks its cache handling).
uv venv --python 3.11 .venv-mt3
$py3 = ".venv-mt3\Scripts\python.exe"
uv pip install --python $py3 torch torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install --python $py3 "mt3-infer==0.2.0" "transformers==4.45.1" librosa pretty_midi
# mt3-infer's auto-download git-clones the whole HF Space; fetch just the checkpoint.
$mt3 = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops"
$mt3Ck = "third_party\mt3_checkpoints\yourmt3\$mt3\last.ckpt"
if (-not (Test-Path $mt3Ck)) {
    New-Item -ItemType Directory -Force (Split-Path $mt3Ck) | Out-Null
    curl.exe -L -o $mt3Ck "https://huggingface.co/spaces/mimbres/YourMT3/resolve/main/amt/logs/2024/$mt3/checkpoints/last.ckpt"
}
