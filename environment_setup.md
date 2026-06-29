# MeTMaP Reproduction Environment Setup

This project uses a Python environment for reproducing the core idea of MeTMaP: detecting false vector matching problems with metamorphic testing.

## Recommended Runtime

- OS: Windows 10/11
- Python: 3.11.x
- Hardware: NVIDIA GPU is recommended for larger embedding batches
- Optional: 8 GB RAM or above

Python 3.14 is not recommended because several machine-learning libraries may not provide compatible wheels yet.

## GPU Setup Commands

Install Python 3.11 first, then run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## Verification

After installation, verify the core packages:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
python -c "import sentence_transformers, faiss, pandas, sklearn; print('environment ok')"
```

## Planned Experiment Stack

- `torch`: GPU acceleration for embedding generation
- `sentence-transformers`: embedding models
- `faiss-cpu`: vector index and similarity search
- `pandas`: experiment records and result tables
- `scikit-learn`: metric calculation and data splitting
- `matplotlib`: result visualization

## Notes

`faiss-cpu` is used for index/search compatibility on Windows. The expensive step in this reproduction is usually embedding generation, which can still run on GPU through PyTorch and `sentence-transformers`.

If GPU installation fails, use the CPU fallback:

```powershell
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio
pip install -r requirements.txt
```
