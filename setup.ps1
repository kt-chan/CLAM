# Check if Conda is available
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Error "Conda is not installed or not available in PATH."
    exit 1
}

# Check if requirements.txt exists
if (-not (Test-Path requirements.txt)) {
    Write-Error "requirements.txt not found in current directory."
    exit 1
}

# Create or check Conda environment
$venvPath = ".\venv"
if (Test-Path $venvPath) {
    Write-Host "Virtual environment exists at $venvPath. Checking if Python version is up to date..."
    conda update --prefix $venvPath --yes python=3.10
} else {
    Write-Host "Creating conda environment at $venvPath..."
    conda create --prefix $venvPath --yes python=3.10
}

# Get the Python executable path
$pythonPath = Join-Path $venvPath "python.exe"
if (-not (Test-Path $pythonPath)) {
    $pythonPath = Join-Path $venvPath "bin/python"
}


# Activate the conda environment and install packages using pip
Write-Host "Installing packages from requirements.txt..."
conda activate $venvPath
pip install --upgrade -r requirements.txt

python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'PyTorch version: {torch.__version__}')"