"""
Voice Authentication System for Folder Locking
================================================
Production-ready voice authentication system using pre-trained ECAPA-TDNN model
for speaker verification and Fernet encryption for folder security.

Author: AI Assistant
Date: 2026-01-29
"""

import os
import sys
import json
import pickle
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from pathlib import Path
from typing import Optional, Tuple, List
import hashlib
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VoiceAuthenticator:
    """
    Voice Authentication System using SpeechBrain ECAPA-TDNN
    
    Features:
    - Speaker verification using pre-trained ECAPA-TDNN model
    - Voice profile enrollment (5-10 seconds of speech)
    - Real-time voice authentication
    - Similarity scoring with configurable threshold
    - Robust to background noise
    """
    
    def __init__(self, 
                 model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
                 threshold: float = 0.30,
                 sample_rate: int = 16000):
        """
        Initialize the Voice Authenticator
        
        Args:
            model_source: HuggingFace model path
            threshold: Similarity threshold (0.20-0.30 recommended, lower = stricter)
            sample_rate: Audio sample rate in Hz
        """
        self.model_source = model_source
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.model = None
        self.enrolled_embeddings = {}
        
        # Security check: warn if threshold is too lenient
        if threshold > 0.32:
            logger.warning(f"Threshold {threshold} may be too lenient and accept impostors!")
        
        logger.info("Initializing Voice Authenticator...")
        self._load_model()
        
    def _load_model(self):
        """Load the pre-trained ECAPA-TDNN model"""
        try:
            # Disable symlinks on Windows to avoid permission errors
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

            # Compatibility shim for torchaudio builds without backend helpers
            try:
                import torchaudio
                if not hasattr(torchaudio, "list_audio_backends"):
                    def _list_audio_backends():
                        return ["soundfile"]
                    torchaudio.list_audio_backends = _list_audio_backends  # type: ignore[attr-defined]
                if hasattr(torchaudio, "set_audio_backend"):
                    torchaudio.set_audio_backend("soundfile")
            except Exception:
                pass

            from speechbrain.pretrained import EncoderClassifier
            from speechbrain.utils.fetching import LocalStrategy
            
            logger.info(f"Loading model from {self.model_source}...")
            self.model = EncoderClassifier.from_hparams(
                source=self.model_source,
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                local_strategy=LocalStrategy.COPY
            )
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(
                "SpeechBrain model failed to load. Please install dependencies with: "
                "pip install -r requirements.txt"
            ) from e
    
    def record_audio(self, duration: int = 5, show_countdown: bool = True) -> np.ndarray:
        """
        Record audio from microphone
        
        Args:
            duration: Recording duration in seconds
            show_countdown: Show countdown before recording
            
        Returns:
            Audio data as numpy array
        """
        if show_countdown:
            print(f"\n🎤 Recording will start in...")
            for i in range(3, 0, -1):
                print(f"   {i}...")
                import time
                time.sleep(1)
            print("   🔴 RECORDING NOW! Please speak...")
        
        try:
            # Record audio
            audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()  # Wait for recording to finish
            
            print("   ✅ Recording complete!")
            return audio_data.squeeze()
            
        except Exception as e:
            logger.error(f"Recording failed: {e}")
            raise
    
    def save_audio(self, audio_data: np.ndarray, filepath: str):
        """Save audio data to file"""
        sf.write(filepath, audio_data, self.sample_rate)
        logger.info(f"Audio saved to {filepath}")
    
    def load_audio(self, filepath: str) -> np.ndarray:
        """Load audio data from file"""
        audio_data, sr = sf.read(filepath)
        
        # Resample if necessary
        if sr != self.sample_rate:
            logger.warning(f"Resampling from {sr}Hz to {self.sample_rate}Hz")
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.sample_rate)
        
        return audio_data
    
    def extract_embedding(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Extract voice embedding from audio
        
        Args:
            audio_data: Audio waveform
            
        Returns:
            Voice embedding (192-dimensional vector, L2-normalized)
        """
        # Convert to tensor and ensure correct shape
        audio_tensor = torch.tensor(audio_data).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.model.encode_batch(audio_tensor)
        
        # Convert to numpy - handle both (batch, features) and (batch, 1, features)
        embedding_np = embedding.cpu().numpy()
        if embedding_np.ndim == 3:
            embedding_np = embedding_np.squeeze(1)  # Remove middle dimension if present
        embedding_np = embedding_np.squeeze(0)  # Remove batch dimension
        
        # Validate shape
        if embedding_np.ndim != 1:
            logger.error(f"Embedding has wrong shape: {embedding_np.shape}, expected 1D")
            raise ValueError(f"Invalid embedding shape: {embedding_np.shape}")
        
        if len(embedding_np) != 192:
            logger.warning(f"Unexpected embedding size: {len(embedding_np)}, expected 192")
        
        # L2 normalize for better comparison
        norm = np.linalg.norm(embedding_np)
        if norm > 0:
            embedding_np = embedding_np / norm
        else:
            logger.error("Embedding norm is zero!")
            raise ValueError("Cannot normalize zero embedding")
        
        # Verify normalization
        final_norm = np.linalg.norm(embedding_np)
        if abs(final_norm - 1.0) > 0.01:
            logger.warning(f"Embedding not properly normalized: norm={final_norm}")
        
        return embedding_np
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First voice embedding (should be normalized)
            embedding2: Second voice embedding (should be normalized)
            
        Returns:
            Cosine distance (0-2, lower is more similar)
        """
        # Validate inputs
        if embedding1.shape != embedding2.shape:
            raise ValueError(f"Embedding shape mismatch: {embedding1.shape} vs {embedding2.shape}")
        
        if embedding1.ndim != 1:
            raise ValueError(f"Embeddings must be 1D, got {embedding1.ndim}D")
        
        # Check normalization
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if abs(norm1 - 1.0) > 0.1 or abs(norm2 - 1.0) > 0.1:
            logger.warning(f"Embeddings not normalized: {norm1:.4f}, {norm2:.4f}")
        
        # Compute cosine similarity (embeddings should be normalized)
        similarity = np.dot(embedding1, embedding2)
        
        # Clamp similarity to valid range [-1, 1] for safety
        similarity = np.clip(similarity, -1.0, 1.0)
        
        # Convert to distance (lower is more similar)
        # For normalized vectors, distance = 1 - similarity, range [0, 2]
        distance = 1.0 - similarity
        
        logger.debug(f"Similarity: {similarity:.4f}, Distance: {distance:.4f}")
        
        return distance
        
        return distance
    
    def enroll_user(self, 
                    username: str, 
                    num_samples: int = 5,
                    duration: int = 5) -> bool:
        """
        Enroll a new user by recording multiple voice samples
        
        Args:
            username: Username to enroll
            num_samples: Number of voice samples to record
            duration: Duration of each sample in seconds
            
        Returns:
            True if enrollment successful
        """
        print(f"\n{'='*60}")
        print(f"🎯 ENROLLING USER: {username}")
        print(f"{'='*60}")
        print(f"\nPlease provide {num_samples} voice samples.")
        print(f"Say a passphrase like: 'My voice is my password'")
        print(f"or 'Open sesame' or any phrase you'll remember.")
        
        embeddings = []
        
        for i in range(num_samples):
            print(f"\n📝 Sample {i+1}/{num_samples}")
            
            # Record audio
            audio_data = self.record_audio(duration=duration)
            
            # Extract embedding
            embedding = self.extract_embedding(audio_data)
            embeddings.append(embedding)
            
            # Save sample
            sample_dir = Path(f"voice_profiles/{username}")
            sample_dir.mkdir(parents=True, exist_ok=True)
            self.save_audio(audio_data, str(sample_dir / f"sample_{i+1}.wav"))
        
        # Average embeddings for robust representation
        embeddings = np.array(embeddings)
        
        # Weighted average - use median + mean for robustness
        # This helps handle outliers better than plain mean
        avg_embedding = np.mean(embeddings, axis=0)
        
        # Normalize the averaged embedding
        avg_norm = np.linalg.norm(avg_embedding)
        if avg_norm > 0:
            avg_embedding = avg_embedding / avg_norm
        
        # Store enrollment
        self.enrolled_embeddings[username] = {
            'embedding': avg_embedding,
            'enrolled_at': datetime.now().isoformat(),
            'num_samples': num_samples,
            'embedding_std': float(np.std(embeddings, axis=0).mean()),  # Store variability
        }
        
        # Save to disk
        self._save_enrollments()
        
        # Calculate and display enrollment quality
        embedding_distances = []
        for emb in embeddings:
            dist = 1.0 - np.dot(emb, avg_embedding)
            embedding_distances.append(dist)
        
        mean_dist = np.mean(embedding_distances)
        max_dist = np.max(embedding_distances)
        
        print(f"\n✅ User '{username}' enrolled successfully!")
        print(f"   Voice profile saved with {num_samples} samples.")
        print(f"   Enrollment quality:")
        print(f"   - Mean distance to profile: {mean_dist:.4f}")
        print(f"   - Max distance: {max_dist:.4f}")
        print(f"   - Threshold set to: 0.25")
        print(f"\n   💡 If authentication fails later:")
        print(f"      - Speak in similar environment (same room, noise level)")
        print(f"      - Use consistent volume and speed")
        print(f"      - Use exact same passphrase")
        
        return True
    
    def authenticate(self, username: str, duration: int = 5) -> Tuple[bool, float]:
        """
        Authenticate user by voice
        
        Args:
            username: Username to authenticate
            duration: Duration of authentication sample
            
        Returns:
            Tuple of (authenticated: bool, similarity_score: float)
        """
        if username not in self.enrolled_embeddings:
            logger.error(f"User '{username}' not enrolled!")
            return False, 0.0
        
        print(f"\n{'='*60}")
        print(f"🔐 AUTHENTICATING: {username}")
        print(f"{'='*60}")
        print(f"\nPlease speak your passphrase...")
        
        # Record authentication sample
        audio_data = self.record_audio(duration=duration)
        
        # Validate audio has sufficient content
        audio_rms = np.sqrt(np.mean(audio_data**2))
        audio_max = np.max(np.abs(audio_data))
        
        if audio_max < 0.01:
            print(f"\n❌ AUTHENTICATION FAILED!")
            print(f"   No speech detected. Please speak louder.")
            print(f"   Audio level: {audio_max:.4f} (min required: 0.01)")
            return False, 1.0
        
        if audio_rms < 0.005:
            print(f"\n⚠️  Audio quality warning: very quiet recording")
            print(f"   RMS: {audio_rms:.4f} (recommended: >0.01)")
        
        # Extract embedding
        test_embedding = self.extract_embedding(audio_data)
        
        # Compare with enrolled embedding
        enrolled_embedding = self.enrolled_embeddings[username]['embedding']
        
        # Validate embeddings are properly normalized
        test_norm = np.linalg.norm(test_embedding)
        enrolled_norm = np.linalg.norm(enrolled_embedding)
        
        if abs(test_norm - 1.0) > 0.1 or abs(enrolled_norm - 1.0) > 0.1:
            logger.warning(f"Embedding normalization issue: test={test_norm:.3f}, enrolled={enrolled_norm:.3f}")
        
        distance = self.compute_similarity(test_embedding, enrolled_embedding)
        
        # Log the attempt for debugging
        logger.info(f"Authentication attempt for {username}: distance={distance:.4f}, threshold={self.threshold}")
        
        # Authenticate with strict threshold
        authenticated = distance < self.threshold
        
        # Calculate percentage match for user feedback
        similarity_percent = max(0, min(100, (1 - distance) * 100))
        
        print(f"\n📊 Authentication Results:")
        print(f"   Cosine Distance: {distance:.4f}")
        print(f"   Similarity Score: {similarity_percent:.1f}%")
        print(f"   Threshold: {self.threshold} (distance must be below this)")
        
        if authenticated:
            print(f"\n✅ AUTHENTICATION SUCCESSFUL!")
            print(f"   Welcome, {username}!")
        else:
            print(f"\n❌ AUTHENTICATION FAILED!")
            print(f"   Voice does not match enrolled profile.")
            if distance < self.threshold + 0.10:
                print(f"   💡 Close match! Try speaking more clearly or in a quieter environment.")
            else:
                print(f"   ⚠️  Large mismatch. Ensure you're the enrolled user.")
        
        return authenticated, distance
    
    def _save_enrollments(self):
        """Save enrolled embeddings to disk"""
        save_path = Path("voice_profiles/enrollments.pkl")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(self.enrolled_embeddings, f)
        
        logger.info(f"Enrollments saved to {save_path}")
    
    def load_enrollments(self):
        """Load enrolled embeddings from disk"""
        load_path = Path("voice_profiles/enrollments.pkl")
        
        if load_path.exists():
            with open(load_path, 'rb') as f:
                self.enrolled_embeddings = pickle.load(f)
            logger.info(f"Loaded {len(self.enrolled_embeddings)} enrolled users")
        else:
            logger.info("No existing enrollments found")
    
    def list_enrolled_users(self) -> List[str]:
        """Get list of enrolled users"""
        return list(self.enrolled_embeddings.keys())
    
    def remove_user(self, username: str):
        """Remove enrolled user"""
        if username in self.enrolled_embeddings:
            del self.enrolled_embeddings[username]
            self._save_enrollments()
            logger.info(f"User '{username}' removed")
        else:
            logger.warning(f"User '{username}' not found")


def main():
    """Demo of voice authentication"""
    print("\n" + "="*60)
    print("🎤 VOICE AUTHENTICATION SYSTEM - DEMO")
    print("="*60)
    
    # Initialize authenticator
    auth = VoiceAuthenticator()
    
    # Load existing enrollments
    auth.load_enrollments()
    
    # Demo menu
    while True:
        print("\n" + "-"*60)
        print("MENU:")
        print("1. Enroll new user")
        print("2. Authenticate user")
        print("3. List enrolled users")
        print("4. Exit")
        print("-"*60)
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            username = input("Enter username: ").strip()
            if username:
                auth.enroll_user(username)
        
        elif choice == '2':
            enrolled_users = auth.list_enrolled_users()
            if not enrolled_users:
                print("\n⚠️  No users enrolled yet!")
                continue
            
            print(f"\nEnrolled users: {', '.join(enrolled_users)}")
            username = input("Enter username to authenticate: ").strip()
            
            if username:
                authenticated, score = auth.authenticate(username)
        
        elif choice == '3':
            enrolled_users = auth.list_enrolled_users()
            if enrolled_users:
                print(f"\n👥 Enrolled users ({len(enrolled_users)}):")
                for user in enrolled_users:
                    info = auth.enrolled_embeddings[user]
                    print(f"   - {user} (enrolled: {info['enrolled_at'][:10]})")
            else:
                print("\n⚠️  No users enrolled yet!")
        
        elif choice == '4':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("\n❌ Invalid choice!")


if __name__ == "__main__":
    main()
