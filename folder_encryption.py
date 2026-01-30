"""
Folder Encryption Module
========================
Secure folder encryption using Fernet (AES-128) symmetric encryption.

Features:
- Encrypt entire folders recursively
- Decrypt folders with authentication
- Preserve folder structure
- Secure key management
"""

import os
import shutil
import base64
from pathlib import Path
from typing import Optional, List
import json
import hashlib
import logging
from datetime import datetime

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

class FolderEncryption:
    """
    Folder encryption system using Fernet symmetric encryption
    """
    
    def __init__(self):
        """Initialize the encryption system"""
        self.fernet = None
        self.key = None
        
    def generate_key_from_password(self, password: str, salt: bytes = None) -> tuple:
        """
        Generate encryption key from password using PBKDF2
        
        Args:
            password: User password
            salt: Salt for key derivation (generated if None)
            
        Returns:
            Tuple of (key, salt)
        """
        if salt is None:
            salt = os.urandom(16)
        
        # Use PBKDF2 to derive key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = kdf.derive(password.encode())
        
        return key, salt
    
    def generate_key(self) -> bytes:
        """
        Generate a new encryption key
        
        Returns:
            Encryption key
        """
        return Fernet.generate_key()
    
    def save_key(self, key: bytes, filepath: str, password: Optional[str] = None):
        """
        Save encryption key to file
        
        Args:
            key: Encryption key
            filepath: Path to save key
            password: Optional password to encrypt the key
        """
        key_data = {
            'created_at': datetime.now().isoformat()
        }
        
        if password:
            # Encrypt the key with password-derived key
            derived_key, salt = self.generate_key_from_password(password)
            fernet_key = base64.urlsafe_b64encode(derived_key)
            fernet = Fernet(fernet_key)
            encrypted_key = fernet.encrypt(key)
            
            key_data['encrypted'] = True
            key_data['salt'] = salt.hex()
            key_data['key'] = encrypted_key.decode('utf-8')
        else:
            # Store plaintext key (demo / non-production)
            key_data['encrypted'] = False
            key_data['key'] = key.decode('utf-8')
        
        with open(filepath, 'w') as f:
            json.dump(key_data, f, indent=2)
        
        logger.info(f"Key saved to {filepath}")
    
    def load_key(self, filepath: str, password: Optional[str] = None) -> bytes:
        """
        Load encryption key from file
        
        Args:
            filepath: Path to key file
            password: Password if key is encrypted
            
        Returns:
            Encryption key
        """
        with open(filepath, 'r') as f:
            key_data = json.load(f)
        
        if key_data.get('encrypted'):
            if not password:
                raise ValueError("Password required to decrypt key file")
            salt = bytes.fromhex(key_data['salt'])
            derived_key, _ = self.generate_key_from_password(password, salt=salt)
            fernet_key = base64.urlsafe_b64encode(derived_key)
            fernet = Fernet(fernet_key)
            key = fernet.decrypt(key_data['key'].encode('utf-8'))
        else:
            key = key_data['key'].encode('utf-8')
        
        return key
    
    def set_key(self, key: bytes):
        """Set the encryption key"""
        self.key = key
        self.fernet = Fernet(key)
    
    def encrypt_file(self, filepath: str, output_path: Optional[str] = None) -> str:
        """
        Encrypt a single file
        
        Args:
            filepath: Path to file to encrypt
            output_path: Output path (defaults to filepath + .encrypted)
            
        Returns:
            Path to encrypted file
        """
        if not self.fernet:
            raise ValueError("Encryption key not set!")
        
        # Read file
        with open(filepath, 'rb') as f:
            file_data = f.read()
        
        # Encrypt
        encrypted_data = self.fernet.encrypt(file_data)
        
        # Determine output path
        if output_path is None:
            output_path = filepath + '.encrypted'
        
        # Write encrypted file
        with open(output_path, 'wb') as f:
            f.write(encrypted_data)
        
        return output_path
    
    def decrypt_file(self, filepath: str, output_path: Optional[str] = None) -> str:
        """
        Decrypt a single file
        
        Args:
            filepath: Path to encrypted file
            output_path: Output path (defaults to removing .encrypted)
            
        Returns:
            Path to decrypted file
        """
        if not self.fernet:
            raise ValueError("Encryption key not set!")
        
        # Read encrypted file
        with open(filepath, 'rb') as f:
            encrypted_data = f.read()
        
        # Decrypt
        try:
            decrypted_data = self.fernet.decrypt(encrypted_data)
        except Exception as e:
            raise ValueError(f"Decryption failed! Wrong key or corrupted file: {e}")
        
        # Determine output path
        if output_path is None:
            if filepath.endswith('.encrypted'):
                output_path = filepath[:-10]  # Remove .encrypted
            else:
                output_path = filepath + '.decrypted'
        
        # Write decrypted file
        with open(output_path, 'wb') as f:
            f.write(decrypted_data)
        
        return output_path
    
    def encrypt_folder(self, folder_path: str, delete_original: bool = False) -> dict:
        """
        Encrypt all files in a folder recursively
        
        Args:
            folder_path: Path to folder to encrypt
            delete_original: Whether to delete original files after encryption
            
        Returns:
            Dictionary with encryption statistics
        """
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            raise ValueError(f"Folder not found: {folder_path}")
        
        if not folder_path.is_dir():
            raise ValueError(f"Not a folder: {folder_path}")
        
        print(f"\n🔒 Encrypting folder: {folder_path}")
        
        stats = {
            'total_files': 0,
            'encrypted_files': 0,
            'failed_files': 0,
            'total_size': 0,
            'files': []
        }
        
        # Get all files recursively
        all_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                filepath = Path(root) / file
                # Skip already encrypted files
                if not str(filepath).endswith('.encrypted'):
                    all_files.append(filepath)
        
        stats['total_files'] = len(all_files)
        
        # Encrypt each file
        for filepath in all_files:
            try:
                # Get file size
                file_size = filepath.stat().st_size
                stats['total_size'] += file_size
                
                # Encrypt
                encrypted_path = self.encrypt_file(str(filepath))
                
                stats['encrypted_files'] += 1
                stats['files'].append({
                    'original': str(filepath),
                    'encrypted': encrypted_path,
                    'size': file_size
                })
                
                # Delete original if requested
                if delete_original:
                    filepath.unlink()
                
                print(f"   ✅ {filepath.name}")
                
            except Exception as e:
                logger.error(f"Failed to encrypt {filepath}: {e}")
                stats['failed_files'] += 1
                print(f"   ❌ {filepath.name} - {str(e)}")
        
        # Save encryption manifest
        manifest_path = folder_path / '.encryption_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump({
                'encrypted_at': datetime.now().isoformat(),
                'stats': stats,
                'folder': str(folder_path)
            }, f, indent=2)
        
        print(f"\n📊 Encryption complete:")
        print(f"   Total files: {stats['total_files']}")
        print(f"   Encrypted: {stats['encrypted_files']}")
        print(f"   Failed: {stats['failed_files']}")
        print(f"   Total size: {stats['total_size'] / 1024:.2f} KB")
        
        return stats
    
    def decrypt_folder(self, folder_path: str, delete_encrypted: bool = False) -> dict:
        """
        Decrypt all encrypted files in a folder recursively
        
        Args:
            folder_path: Path to folder to decrypt
            delete_encrypted: Whether to delete encrypted files after decryption
            
        Returns:
            Dictionary with decryption statistics
        """
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            raise ValueError(f"Folder not found: {folder_path}")
        
        print(f"\n🔓 Decrypting folder: {folder_path}")
        
        stats = {
            'total_files': 0,
            'decrypted_files': 0,
            'failed_files': 0,
            'files': []
        }
        
        # Get all encrypted files recursively
        encrypted_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.encrypted'):
                    filepath = Path(root) / file
                    encrypted_files.append(filepath)
        
        stats['total_files'] = len(encrypted_files)
        
        # Decrypt each file
        for filepath in encrypted_files:
            try:
                # Decrypt
                decrypted_path = self.decrypt_file(str(filepath))
                
                stats['decrypted_files'] += 1
                stats['files'].append({
                    'encrypted': str(filepath),
                    'decrypted': decrypted_path
                })
                
                # Delete encrypted file if requested
                if delete_encrypted:
                    filepath.unlink()
                
                print(f"   ✅ {filepath.name}")
                
            except Exception as e:
                logger.error(f"Failed to decrypt {filepath}: {e}")
                stats['failed_files'] += 1
                print(f"   ❌ {filepath.name} - {str(e)}")
        
        print(f"\n📊 Decryption complete:")
        print(f"   Total files: {stats['total_files']}")
        print(f"   Decrypted: {stats['decrypted_files']}")
        print(f"   Failed: {stats['failed_files']}")
        
        return stats
    
    def is_folder_encrypted(self, folder_path: str) -> bool:
        """
        Check if folder contains encrypted files
        
        Args:
            folder_path: Path to folder
            
        Returns:
            True if folder has encrypted files
        """
        folder_path = Path(folder_path)
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.encrypted'):
                    return True
        
        return False


def demo():
    """Demo of folder encryption"""
    print("\n" + "="*60)
    print("🔐 FOLDER ENCRYPTION - DEMO")
    print("="*60)
    
    # Initialize encryption
    enc = FolderEncryption()
    
    # Generate key
    key = enc.generate_key()
    enc.set_key(key)
    
    # Save key
    enc.save_key(key, 'encryption.key')
    print("\n✅ Encryption key generated and saved to 'encryption.key'")
    
    # Create demo folder
    demo_folder = Path("demo_secure_folder")
    demo_folder.mkdir(exist_ok=True)
    
    # Create some demo files
    (demo_folder / "document1.txt").write_text("This is a secret document!")
    (demo_folder / "document2.txt").write_text("Another confidential file.")
    
    subfolder = demo_folder / "subfolder"
    subfolder.mkdir(exist_ok=True)
    (subfolder / "secret.txt").write_text("Hidden information here.")
    
    print(f"\n📁 Created demo folder: {demo_folder}")
    
    # Encrypt folder
    enc.encrypt_folder(str(demo_folder))
    
    print("\n✅ Folder encrypted! Check the demo_secure_folder directory.")
    print("   Original files are preserved alongside encrypted versions.")
    

if __name__ == "__main__":
    demo()
