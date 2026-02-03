# 📱 Mobile Access Guide for Indoor Object Detector
This guide explains how to use your Object Detection app on your smartphone or tablet.

## Method 1: Local Network (Wi-Fi) - *Fastest & Easiest*
If your phone and computer are on the same Wi-Fi network, you can connect directly.

### Steps:
1.  **Start the App on your PC:**
    Open your terminal in the project folder and run:
    ```bash
    streamlit run app.py
    ```

2.  **Find the Network URL:**
    Look at the output in your terminal. You will see something like:
    ```
    You can now view your Streamlit app in your browser.
    
    Local URL: http://localhost:8501
    Network URL: http://192.168.1.5:8501  <-- THIS IS THE ONE YOU NEED
    ```

3.  **Open on Mobile:**
    - Open Chrome or Safari on your phone.
    - Type the **Network URL** (e.g., `http://192.168.1.5:8501`) into the address bar.

4.  **Use Mobile Mode:**
    - Go to the **Live Webcam** tab.
    - Select **Browser Camera (Mobile/Tablet)**.
    - Tap **Take Photo** to detect objects!

> [!NOTE]
> If the page doesn't load, check your firewall settings. You may need to allow Python/Streamlit through the Windows Firewall.

---

## Method 2: Streamlit Cloud - *Accessible Anywhere*
If you want to share the app with friends or use it outside your home Wi-Fi.

### Steps:
1.  **Push Code to GitHub:**
    Ensure your project is uploaded to a GitHub repository.

2.  **Deploy on Streamlit Cloud:**
    - Go to [share.streamlit.io](https://share.streamlit.io/).
    - Sign in with GitHub.
    - Click **New App**.
    - Select your repository and setting the main file path to `app.py`.
    - Click **Deploy**.

3.  **Share the Link:**
    Once deployed, you get a public URL (e.g., `https://my-object-detector.streamlit.app`) that works on any device!
