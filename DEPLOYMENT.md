# 🚀 Streamlit Cloud Deployment Guide

## Quick Fix for OpenCV Error

If you encountered the `ImportError` with cv2, you've already fixed it! The solution involves two files:

### 1. `packages.txt` ✅
This file tells Streamlit Cloud to install system-level dependencies:
```
libgl1-mesa-glx
libglib2.0-0
```

### 2. `requirements.txt` ✅
Changed `opencv-python` to `opencv-python-headless` (the headless version works on servers without displays).

---

## Deployment Steps

### Initial Setup

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/your-repo-name.git
   git push -u origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io/)
   - Sign in with GitHub
   - Click **"New app"**
   - Select your repository
   - Set **Main file path:** `app.py`
   - Click **"Deploy"**

### Important Files for Deployment

Make sure these files are in your repository:
- ✅ `app.py` - Main application
- ✅ `requirements.txt` - Python dependencies (with opencv-python-headless)
- ✅ `packages.txt` - System dependencies (NEW - fixes OpenCV error)
- ✅ `src/` folder - All source code
- ✅ `data.yaml` - Dataset configuration

### Files to EXCLUDE (.gitignore)

Create a `.gitignore` file with:
```
# Model weights (too large for git)
*.pt
yolov8*.pt

# Python cache
__pycache__/
*.pyc
*.pyo

# Data
data/raw/
runs/

# Environment
venv/
.env
```

> [!IMPORTANT]
> Model weights (`.pt` files) are too large for GitHub. Streamlit Cloud will download them automatically on first run.

---

## Troubleshooting

### Error: "File too large"
- Don't commit `.pt` model files
- Add them to `.gitignore`
- The app downloads them automatically

### Error: "Module not found"
- Check `requirements.txt` has all dependencies
- Make sure you're using `opencv-python-headless` not `opencv-python`

### Error: "Out of memory"
- Use `yolov8n.pt` (smallest model) as default
- Streamlit Cloud free tier has limited RAM

### Webcam doesn't work
- Desktop webcam mode won't work on cloud (requires local camera)
- Use "Browser Camera (Mobile/Tablet)" mode instead
- This works perfectly on Streamlit Cloud!

---

## After Deployment

Your app will be available at:
```
https://your-app-name.streamlit.app
```

You can share this URL with anyone! 🎉

### Managing Your App
- Click **"Manage app"** in the bottom-right of your deployed app
- View logs, reboot, or delete the app
- Check resource usage and errors

---

## Performance Tips

1. **Use caching:** The app already uses `@st.cache_resource` for model loading
2. **Choose smaller models:** `yolov8n.pt` is fastest for cloud deployment
3. **Limit video processing:** Long videos may timeout on free tier
4. **Use image mode:** Most reliable for cloud deployment

---

## Need Help?

- Check [Streamlit Community Forum](https://discuss.streamlit.io/)
- Review [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-community-cloud)
- Open an issue on your GitHub repo
