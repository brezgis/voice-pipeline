#!/usr/bin/env python3
"""
Kyutai TTS 1.6B Speed Benchmark Script

Tests different n_q values, torch.compile optimizations, warm-up effects,
and text length impact on generation speed.
"""

import os
import sys
import time
import torch
import numpy as np
import subprocess
from pathlib import Path
import json
from typing import Dict, Any, List, Tuple
import gc

# Add the parent directory to Python path to import moshi
sys.path.append('/path/to/voice-pipeline/v2')

try:
    from moshi.models.loaders import CheckpointInfo
    from moshi.models.tts import DEFAULT_DSM_TTS_VOICE_REPO, TTSModel
    import soundfile as sf
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you're in the v2 environment: source /path/to/voice-pipeline/v2/venv/bin/activate")
    sys.exit(1)

# Benchmark configuration
BENCHMARK_CONFIGS = {
    'n_q_values': [32, 24, 16, 12, 8],
    'torch_compile_modes': [None, 'default', 'reduce-overhead'],
    'test_texts': {
        'short': "Hello! How are you doing today?",
        'medium': "This is a medium length text for testing the text-to-speech system. It should be long enough to see if there are any differences in performance based on text length, but not so long as to take forever to process.",
        'long': "This is a much longer text passage designed to test the performance characteristics of the text-to-speech system with extended content. We want to see if the generation speed remains consistent across different text lengths, or if there are any performance degradations or improvements as the text gets longer. The system should maintain a consistent realtime factor regardless of whether we're processing a short sentence or a longer paragraph like this one. Performance consistency is crucial for real-time streaming applications where users expect smooth, uninterrupted audio output."
    }
}

# Test voice (using the same one from the working code)
TEST_VOICE = "expresso/ex03-ex01_happy_001_channel1_334s.wav"

class GPUMonitor:
    """Monitor GPU memory usage during benchmarks"""
    
    @staticmethod
    def get_gpu_memory_usage() -> float:
        """Get current GPU memory usage in MB"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True
            )
            return float(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError):
            return 0.0
    
    @staticmethod
    def get_gpu_memory_by_process() -> List[Tuple[str, float]]:
        """Get memory usage by process in MB"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=pid,name,used_memory', '--format=csv,noheader'],
                capture_output=True, text=True, check=True
            )
            processes = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        name = parts[1]
                        memory_str = parts[2].replace(' MiB', '').replace(' MB', '')
                        memory = float(memory_str)
                        processes.append((name, memory))
            return processes
        except (subprocess.CalledProcessError, ValueError):
            return []

class KyutaiTTSBenchmark:
    """Benchmark runner for Kyutai TTS model"""
    
    def __init__(self):
        self.model = None
        self.checkpoint_info = None
        self.results = []
        
    def load_model(self, n_q: int, torch_compile_mode: str = None) -> None:
        """Load Kyutai TTS model with specified configuration"""
        print(f"Loading model with n_q={n_q}, torch_compile={torch_compile_mode}")
        
        # Clear any existing model
        if self.model is not None:
            del self.model
            self.model = None
            
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        # Load checkpoint info (only once)
        if self.checkpoint_info is None:
            print("Loading checkpoint info from HuggingFace...")
            self.checkpoint_info = CheckpointInfo.from_hf_repo("kyutai/tts-1.6b-en_fr")
        
        # Load model
        print("Creating TTS model...")
        self.model = TTSModel.from_checkpoint_info(
            self.checkpoint_info, 
            n_q=n_q, 
            temp=0.6, 
            device=torch.device("cuda")
        )
        
        # Apply torch.compile if requested
        if torch_compile_mode is not None:
            print(f"Applying torch.compile with mode: {torch_compile_mode}")
            if torch_compile_mode == 'default':
                self.model = torch.compile(self.model)
            else:
                self.model = torch.compile(self.model, mode=torch_compile_mode)
    
    def generate_audio(self, text: str, voice: str = TEST_VOICE) -> Tuple[np.ndarray, float, float]:
        """
        Generate audio and return (audio_array, generation_time, audio_duration)
        """
        # Prepare inputs
        entries = self.model.prepare_script([text], padding_between=1)
        voice_path = self.model.get_voice_path(voice)
        condition_attributes = self.model.make_condition_attributes([voice_path], cfg_coef=2.0)
        
        # Storage for audio frames
        pcms = []
        
        def _on_frame(frame):
            if (frame != -1).all():
                pcm = self.model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
                pcms.append(np.clip(pcm[0, 0], -1, 1))
        
        # Generate audio with timing
        start_time = time.time()
        
        with self.model.mimi.streaming(1):
            result = self.model.generate([entries], [condition_attributes], on_frame=_on_frame)
        
        generation_time = time.time() - start_time
        
        # Combine audio frames
        if pcms:
            audio = np.concatenate(pcms, axis=-1)
        else:
            audio = np.array([])
            
        audio_duration = len(audio) / self.model.mimi.sample_rate if len(audio) > 0 else 0.0
        
        return audio, generation_time, audio_duration
    
    def run_benchmark(self, n_q: int, text_name: str, text: str, torch_compile_mode: str = None, 
                     warmup: bool = False, save_audio: bool = True) -> Dict[str, Any]:
        """Run a single benchmark configuration"""
        
        config_name = f"nq{n_q}_{text_name}"
        if torch_compile_mode:
            config_name += f"_compile_{torch_compile_mode}"
        if warmup:
            config_name += "_warmup"
            
        print(f"\n{'='*60}")
        print(f"Running benchmark: {config_name}")
        print(f"Text: {text[:50]}{'...' if len(text) > 50 else ''}")
        print(f"{'='*60}")
        
        # Load model for this configuration
        self.load_model(n_q, torch_compile_mode)
        
        # Measure memory before generation
        memory_before = GPUMonitor.get_gpu_memory_usage()
        
        # Warmup run if requested
        if warmup:
            print("Running warmup...")
            _, _, _ = self.generate_audio(text)
            torch.cuda.synchronize()  # Ensure CUDA operations complete
        
        # Actual benchmark run
        print("Running benchmark...")
        audio, generation_time, audio_duration = self.generate_audio(text)
        
        # Measure memory after generation
        memory_after = GPUMonitor.get_gpu_memory_usage()
        memory_used = memory_after - memory_before
        
        # Calculate realtime factor
        realtime_factor = audio_duration / generation_time if generation_time > 0 else 0.0
        
        # Save audio sample
        audio_path = None
        if save_audio and len(audio) > 0:
            audio_filename = f"/tmp/kyutai_{config_name}.wav"
            sf.write(audio_filename, audio, self.model.mimi.sample_rate)
            audio_path = audio_filename
            print(f"Saved audio to: {audio_path}")
        
        # Evaluate audio quality subjectively (basic checks)
        quality_notes = self.evaluate_quality(audio, audio_duration)
        
        result = {
            'config': config_name,
            'n_q': n_q,
            'text_name': text_name,
            'text_length': len(text),
            'torch_compile': torch_compile_mode,
            'warmup': warmup,
            'generation_time': generation_time,
            'audio_duration': audio_duration,
            'realtime_factor': realtime_factor,
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_used_mb': memory_used,
            'quality_notes': quality_notes,
            'audio_path': audio_path,
            'timestamp': time.time()
        }
        
        # Print results
        print(f"Generation time: {generation_time:.2f}s")
        print(f"Audio duration: {audio_duration:.2f}s")
        print(f"Realtime factor: {realtime_factor:.2f}x")
        print(f"Memory used: {memory_used:.0f} MB")
        print(f"Quality: {quality_notes}")
        
        return result
    
    def evaluate_quality(self, audio: np.ndarray, duration: float) -> str:
        """Basic quality evaluation of generated audio"""
        if len(audio) == 0:
            return "FAILED: No audio generated"
        
        # Check for silence
        rms = np.sqrt(np.mean(audio**2))
        if rms < 0.001:
            return "POOR: Too quiet/silent"
        
        # Check for clipping
        clipped_samples = np.sum(np.abs(audio) > 0.99)
        clip_ratio = clipped_samples / len(audio)
        if clip_ratio > 0.1:
            return "POOR: Heavy clipping detected"
        
        # Check for reasonable duration
        if duration < 0.5:
            return "QUESTIONABLE: Very short audio"
        
        # Check for NaN or inf
        if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
            return "FAILED: NaN/Inf values in audio"
        
        # Basic quality assessment based on dynamic range
        dynamic_range = np.max(audio) - np.min(audio)
        if dynamic_range < 0.1:
            return "POOR: Low dynamic range"
        elif clip_ratio > 0.01:
            return "FAIR: Some clipping"
        else:
            return "GOOD: Clean audio"
    
    def run_all_benchmarks(self) -> List[Dict[str, Any]]:
        """Run the complete benchmark suite"""
        print("Starting Kyutai TTS 1.6B Benchmark Suite")
        print(f"GPU Memory available: {GPUMonitor.get_gpu_memory_usage():.0f} MB")
        
        all_results = []
        
        # 1. Test different n_q values with medium text (baseline)
        print("\n" + "="*80)
        print("PHASE 1: Testing different n_q values")
        print("="*80)
        
        for n_q in BENCHMARK_CONFIGS['n_q_values']:
            result = self.run_benchmark(
                n_q=n_q, 
                text_name='medium', 
                text=BENCHMARK_CONFIGS['test_texts']['medium']
            )
            all_results.append(result)
            self.results.append(result)
            
            # Small delay to let GPU settle
            time.sleep(2)
        
        # 2. Test torch.compile with best performing n_q
        print("\n" + "="*80)
        print("PHASE 2: Testing torch.compile optimizations")
        print("="*80)
        
        # Find best n_q from phase 1 (highest realtime factor)
        best_nq_result = max(all_results, key=lambda x: x['realtime_factor'])
        best_nq = best_nq_result['n_q']
        print(f"Using best n_q from phase 1: {best_nq} (realtime factor: {best_nq_result['realtime_factor']:.2f}x)")
        
        for compile_mode in ['default', 'reduce-overhead']:
            result = self.run_benchmark(
                n_q=best_nq,
                text_name='medium',
                text=BENCHMARK_CONFIGS['test_texts']['medium'],
                torch_compile_mode=compile_mode
            )
            all_results.append(result)
            self.results.append(result)
            time.sleep(2)
        
        # 3. Test warmup effect with best configuration
        print("\n" + "="*80)
        print("PHASE 3: Testing warmup effects")
        print("="*80)
        
        # Find best compile mode
        compile_results = [r for r in all_results if r['torch_compile'] is not None]
        if compile_results:
            best_compile_result = max(compile_results, key=lambda x: x['realtime_factor'])
            best_compile_mode = best_compile_result['torch_compile']
        else:
            best_compile_mode = None
            
        result = self.run_benchmark(
            n_q=best_nq,
            text_name='medium',
            text=BENCHMARK_CONFIGS['test_texts']['medium'],
            torch_compile_mode=best_compile_mode,
            warmup=True
        )
        all_results.append(result)
        self.results.append(result)
        
        # 4. Test text length impact
        print("\n" + "="*80)
        print("PHASE 4: Testing text length impact")
        print("="*80)
        
        for text_name, text in BENCHMARK_CONFIGS['test_texts'].items():
            if text_name == 'medium':  # Skip medium since we already tested it
                continue
                
            result = self.run_benchmark(
                n_q=best_nq,
                text_name=text_name,
                text=text,
                torch_compile_mode=best_compile_mode
            )
            all_results.append(result)
            self.results.append(result)
            time.sleep(2)
        
        return all_results

def main():
    """Main benchmark execution"""
    
    # Check environment
    if not os.path.exists('/path/to/voice-pipeline/v2/venv'):
        print("ERROR: v2 virtual environment not found!")
        print("Please run: source /path/to/voice-pipeline/v2/venv/bin/activate")
        sys.exit(1)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        sys.exit(1)
    
    print("KYUTAI TTS 1.6B SPEED BENCHMARK")
    print("="*50)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    # Initialize benchmark
    benchmark = KyutaiTTSBenchmark()
    
    try:
        # Run all benchmarks
        results = benchmark.run_all_benchmarks()
        
        # Save detailed results to JSON
        results_file = '/path/to/voice-pipeline/v2/benchmarks/kyutai_benchmark_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {results_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        # Find best configurations
        best_overall = max(results, key=lambda x: x['realtime_factor'])
        print(f"\nBest overall performance:")
        print(f"  Config: {best_overall['config']}")
        print(f"  Realtime factor: {best_overall['realtime_factor']:.2f}x")
        print(f"  Memory used: {best_overall['memory_used_mb']:.0f} MB")
        print(f"  Quality: {best_overall['quality_notes']}")
        
        # n_q comparison
        nq_results = [r for r in results if not r['torch_compile'] and not r['warmup'] and r['text_name'] == 'medium']
        print(f"\nn_q value comparison (medium text, no optimizations):")
        for result in sorted(nq_results, key=lambda x: x['n_q']):
            print(f"  n_q={result['n_q']:2d}: {result['realtime_factor']:.2f}x realtime, "
                  f"{result['memory_used_mb']:4.0f}MB, {result['quality_notes']}")
        
        # torch.compile comparison
        compile_results = [r for r in results if r['torch_compile'] and r['text_name'] == 'medium']
        if compile_results:
            print(f"\ntorch.compile comparison:")
            baseline = next((r for r in results if not r['torch_compile'] and not r['warmup'] 
                           and r['text_name'] == 'medium' and r['n_q'] == best_overall['n_q']), None)
            if baseline:
                print(f"  baseline: {baseline['realtime_factor']:.2f}x")
            for result in compile_results:
                speedup = result['realtime_factor'] / baseline['realtime_factor'] if baseline else 1.0
                print(f"  {result['torch_compile']}: {result['realtime_factor']:.2f}x ({speedup:.2f}x speedup)")
        
        print(f"\nAll benchmark audio samples saved to /tmp/kyutai_*.wav")
        
    except Exception as e:
        print(f"\nERROR during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())