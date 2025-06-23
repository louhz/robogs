# Robo-GS Installation Guidance

> **Official Release** - Updated for gsplat 1.0 and the newest mesh reconstruction method: **2DGS + Normal Constraint**

### Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   > This installs all real2sim necessary packages except for StableNormal and COLMAP

2. **Install FFmpeg:**
   ```bash
   # Ubuntu/Debian
   sudo apt install ffmpeg
   
   # macOS
   brew install ffmpeg
   
   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

## 📦 Reconstructed Assets

### Sample Results

We provide a **pre-reconstructed object** to help you understand the real2sim video capture process:

🔗 **[Download Sample Reconstructed Object](https://drive.google.com/file/d/1VVj6VdYO2MdxuC6HpMQBdeaHmR8LsskF/view?usp=drive_link)**

### Asset Structure

The final cleaned result is located at:
```
spray/spray/results/train/ours_7000/spray.ply
```

## 🎯 Workflow

1. **Capture** a 360° surrounded video of your target object
2. **Process** using Gaussian Splat and Mesh Extraction
3. **Generate** digital assets for simulation
4. **Deploy** with policy training and sim2real applications

## 📋 TODO

- [ ] Fix configuration and dataloader structure
- [ ] Improve gripper functionality
- [ ] Enhance object integration with MuJoCo backend