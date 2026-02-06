#!/usr/bin/env python3
"""
Setup script for Kyutai TTS model files

This script downloads and organizes the required model files for the Pipecat voice pipeline:
1. Kyutai TTS 1.6B model files 
2. Voice embedding files
3. Sets up proper directory structure

The total download size is approximately 4-5GB.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and handle errors"""
    print(f"Running: {description or ' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return None

def download_with_git_lfs(repo_url, target_dir):
    """Download a HuggingFace repo with git-lfs support"""
    if os.path.exists(target_dir):
        print(f"Directory {target_dir} already exists, skipping download")
        return True
        
    # Clone the repo
    cmd = ["git", "clone", repo_url, target_dir]
    result = run_command(cmd, f"Cloning {repo_url}")
    
    if result is None:
        return False
        
    # Pull LFS files (must run inside the cloned repo directory)
    cmd = ["git", "-C", target_dir, "lfs", "pull"]
    result = run_command(cmd, f"Pulling LFS files in {target_dir}")
    
    return result is not None

def download_with_huggingface_hub(repo_id, target_dir, filename=None):
    """Download files using huggingface_hub"""
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
        
        if filename:
            # Download specific file
            print(f"Downloading {filename} from {repo_id}")
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=target_dir
            )
            print(f"Downloaded: {file_path}")
        else:
            # Download entire repo
            print(f"Downloading all files from {repo_id}")
            repo_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=target_dir
            )
            print(f"Downloaded to: {repo_path}")
            
        return True
        
    except ImportError:
        print("huggingface_hub not available, falling back to git clone")
        return False
    except Exception as e:
        print(f"Error downloading from HuggingFace Hub: {e}")
        return False

def setup_model_directories():
    """Create model directory structure"""
    base_dir = Path(__file__).parent / "models"
    
    dirs = {
        "kyutai_tts": base_dir / "kyutai_tts",
        "voices": base_dir / "voices", 
        "cache": base_dir / "cache"
    }
    
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {path}")
    
    return dirs

def download_kyutai_tts_model(model_dir):
    """Download Kyutai TTS 1.6B model files"""
    print("\n=== Downloading Kyutai TTS 1.6B Model ===")
    
    repo_id = "kyutai/tts-1.6b-en_fr"
    
    # Try huggingface_hub first
    if download_with_huggingface_hub(repo_id, model_dir):
        return True
        
    # Fallback to git clone
    repo_url = f"https://huggingface.co/{repo_id}"
    return download_with_git_lfs(repo_url, model_dir)

def download_voice_models(voices_dir):
    """Download voice embedding models"""
    print("\n=== Downloading Voice Models ===")
    
    repo_id = "kyutai/tts-voices"
    
    # Try huggingface_hub first  
    if download_with_huggingface_hub(repo_id, voices_dir):
        return True
        
    # Fallback to git clone
    repo_url = f"https://huggingface.co/{repo_id}"
    return download_with_git_lfs(repo_url, voices_dir)

def verify_installations():
    """Verify that all required components are installed"""
    print("\n=== Verifying Installation ===")
    
    # Check moshi
    try:
        import moshi
        print("✓ moshi package available")
    except ImportError:
        print("✗ moshi package not found")
        return False
    
    # Check torch
    try:
        import torch
        print(f"✓ torch {torch.__version__} available")
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("! CUDA not available - will use CPU")
    except ImportError:
        print("✗ torch not found")
        return False
        
    # Check other dependencies
    try:
        import transformers
        print(f"✓ transformers {transformers.__version__} available")
    except ImportError:
        print("✗ transformers not found")
        
    try:
        from huggingface_hub import __version__
        print(f"✓ huggingface_hub {__version__} available")
    except ImportError:
        print("✗ huggingface_hub not found")
        
    return True

def create_test_script(models_dir):
    """Create a test script for the downloaded models"""
    test_script = Path(__file__).parent / "test_kyutai_models.py"
    
    script_content = f'''#!/usr/bin/env python3
"""
Test script for Kyutai TTS models
"""

import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "components"))

from kyutai_tts_service import KyutaiTTSService

async def test_models():
    """Test the downloaded models"""
    print("=== Testing Kyutai TTS Models ===")
    
    # Test service creation (model loads lazily)
    try:
        service = KyutaiTTSService()
        print("✓ KyutaiTTSService created")
        
        # Test basic functionality
        test_text = "Hello, this is a test of the Kyutai TTS system."
        print(f"Testing with text: {{test_text}}")
        
        # Note: This would actually generate audio in a real test
        print("Note: Full audio generation test requires running in pipeline context")
        
    except Exception as e:
        print(f"✗ Error testing simple service: {{e}}")
        
    print("\\nTo test full model integration:")
    print("1. Run python voice_bot_v2.py")
    print("2. Join Discord voice channel")  
    print("3. Speak to test the full pipeline")

if __name__ == "__main__":
    asyncio.run(test_models())
'''
    
    with open(test_script, 'w') as f:
        f.write(script_content)
        
    # Make executable
    os.chmod(test_script, 0o755)
    print(f"Created test script: {test_script}")

def main():
    """Main setup function"""
    print("Setting up Kyutai TTS v2 Voice Pipeline")
    print("=" * 50)
    
    # Verify installations first
    if not verify_installations():
        print("\nMissing dependencies. Please install requirements first:")
        print("pip install -r requirements.txt")
        sys.exit(1)
        
    # Setup directories
    dirs = setup_model_directories()
    
    # Download models
    print(f"\\nModel files will be downloaded to: {dirs['kyutai_tts'].parent}")
    print("This will download approximately 4-5GB of data.")
    
    response = input("Continue? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Setup cancelled.")
        return
        
    # Download Kyutai TTS model
    if not download_kyutai_tts_model(dirs["kyutai_tts"]):
        print("Failed to download Kyutai TTS model")
        sys.exit(1)
        
    # Download voice models  
    if not download_voice_models(dirs["voices"]):
        print("Failed to download voice models")
        sys.exit(1)
        
    # Create test script
    create_test_script(dirs["kyutai_tts"])
    
    print("\\n" + "=" * 50)
    print("Setup Complete!")
    print(f"Models downloaded to: {dirs['kyutai_tts'].parent}")
    print("\\nNext steps:")
    print("1. Set VOICE_BOT_TOKEN environment variable")
    print("2. Run: python test_kyutai_models.py")
    print("3. Run: python voice_bot_v2.py")
    print("\\nNote: First run will be slower as models load into GPU memory.")

if __name__ == "__main__":
    main()