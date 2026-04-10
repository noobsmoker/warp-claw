#!/bin/bash
# M1-specific environment setup for Warp-Claw

set -e

echo "🤖 Warp-Claw M1 Setup"
echo "===================="

# Detect chip
if [[ $(uname -m) != "arm64" ]]; then
    echo "⚠️  Warning: Not running on ARM64 (Apple Silicon)"
fi

# Check for Apple Silicon
if [[ $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "") == *"Apple"* ]]; then
    echo "✅ Detected Apple Silicon"
fi

# Homebrew (x86 vs ARM)
if [[ $(which brew 2>/dev/null) == *"/opt/homebrew"* ]]; then
    echo "✅ Using ARM64 Homebrew"
    BREW_PREFIX="/opt/homebrew"
elif [[ $(which brew 2>/dev/null) == *"/usr/local"* ]]; then
    echo "✅ Using x86_64 Homebrew (Rosetta)"
    BREW_PREFIX="/usr/local"
else
    echo "❌ Homebrew not found. Install from https://brew.sh"
    exit 1
fi

# Python environment
echo ""
echo "🐍 Setting up Python..."

# Check for python
if command -v python3 &> /dev/null; then
    PYVER=$(python3 --version)
    echo "✅ Python: $PYVER"
else:
    echo "❌ Python not found"
    exit 1
fi

# Create venv
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Created virtual environment"
fi

# Activate venv
source venv/bin/activate

# Install PyTorch with MPS support
echo ""
echo "🔥 Installing PyTorch with MPS..."
pip install torch torchvision torchaudio

# Verify MPS available
python3 -c "import torch; print(f'✅ MPS available: {torch.backends.mps.is_available()}')" || true

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create data directories
echo ""
echo "📁 Creating directories..."
mkdir -p data/{models,knowledge,cache}
mkdir -p logs

# Download default model
echo ""
echo "⬇️  Downloading default model..."
python3 scripts/download_models.py qwen-0.5b

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  make run        # Start API server"
echo "  make dashboard # Start dashboard"
echo "  make test      # Run tests"