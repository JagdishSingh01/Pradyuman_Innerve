# 🚀 QUICK START GUIDE

## Get Started in 5 Minutes!

### Step 1: Installation

**Linux/macOS:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows:**
```cmd
setup.bat
```

**Manual Installation:**
```bash
pip install -r requirements.txt
```

> Note: Dependencies must be installed upfront. The app does not auto-install packages at runtime.

---

### Step 2: First Run

```bash
python main.py
```

You'll see the main menu:
```
MAIN MENU:
1. 👤 Enroll new user
2. 🔒 Lock folder
3. 🔓 Unlock folder
4. 📋 List locked folders
5. 👥 List enrolled users
6. 📜 Show access log
7. ℹ️  System information
8. 🚪 Exit
```

---

### Step 3: Enroll Your Voice (First Time Only)

1. Choose option `1` (Enroll new user)
2. Enter your username: `alice`
3. Get ready to record 3 voice samples
4. When prompted, say your passphrase clearly:
   - Example: "My voice is my password"
   - Or: "Open sesame"
   - Or any phrase you'll remember

**Tips:**
- Use a quiet environment
- Speak naturally and clearly
- Use the same phrase for all 3 samples
- Each recording is 5 seconds

---

### Step 4: Lock a Folder

1. Choose option `2` (Lock folder)
2. Enter your username: `alice`
3. Enter folder path: `/path/to/secret_folder`
4. When prompted, say your passphrase for authentication
5. Wait for encryption to complete

**Result:**
- All files in the folder are encrypted
- Original files are preserved (delete manually if desired)
- Folder is now protected by your voice!

---

### Step 5: Unlock a Folder

1. Choose option `3` (Unlock folder)
2. Enter your username: `alice`
3. Enter folder path: `/path/to/secret_folder`
4. Speak your passphrase for authentication
5. Wait for decryption to complete

**Result:**
- All files are decrypted
- Encrypted files are preserved (delete manually if desired)
- You can now access your files!

---

## Common Commands

### View Your Profile
```
Choose: 5. List enrolled users
```

### Check Locked Folders
```
Choose: 4. List locked folders
```

### View Access History
```
Choose: 6. Show access log
```

### System Information
```
Choose: 7. System information
```

---

## Troubleshooting

### "No module named 'speechbrain'"
```bash
pip install speechbrain
```

### "Could not open audio device"
- Check microphone permissions
- Try: `pip install pyaudio`
- Linux: `sudo apt-get install portaudio19-dev`

### Authentication Keeps Failing
- Ensure same microphone for enrollment and authentication
- Reduce background noise
- Speak clearly and consistently
- Try re-enrolling

### Need Help?
- Read README.md for detailed documentation
- Check TECHNICAL_DOCUMENTATION.md for advanced info
- Run examples: `python examples.py`

---

## Security Reminders

✅ **DO:**
- Enroll in a quiet environment
- Use a strong passphrase
- Backup your voice profiles
- Use full-disk encryption too

❌ **DON'T:**
- Share your voice profiles
- Use in very noisy environments
- Rely on this as your only security
- Forget to backup important data

---

## Next Steps

1. **Try Examples:**
   ```bash
   python examples.py
   ```

2. **Run Tests:**
   ```bash
   python test_suite.py
   ```

3. **Read Full Documentation:**
   - README.md - Complete user guide
   - TECHNICAL_DOCUMENTATION.md - Technical deep dive

4. **Customize:**
   - Adjust authentication threshold
   - Change recording duration
   - Configure sample requirements

---

## That's It! 🎉

You now have a voice-authenticated folder locking system!

**Stay secure! 🔒**
