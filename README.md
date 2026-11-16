# 🛡️ Sentinel AI Camera - macOS Surveillance Terminal

Sentinel AI Camera is a local-first, real-time object detection web application designed to run entirely inside your browser. It utilizes your MacBook's built-in webcam and performs on-device computer vision inference using TensorFlow.js and the COCO-SSD model.

This project was built as a clean, highly aesthetic macOS MVP featuring glassmorphic controls, sound telemetry indicators, and an automated threat risk assessor.

---

## 🚀 Quick Start Instructions

Follow these simple steps to run Sentinel AI Camera locally on your Mac.

### 1. Prerequisites
Make sure you have [Node.js](https://nodejs.org/) (v18 or higher recommended) installed.

### 2. Installation
Open your terminal and navigate to the project directory, then run the installation command:

```bash
# Navigate to the Sentinel folder (if you are not already there)
cd /Users/horatiubudai/ceo/Pericle/Sentinel

# Install dependencies
npm install
```

### 3. Start the Development Server
Launch the local Vite server:

```bash
npm run dev
```

By default, the terminal will print an address like `http://localhost:5173`. Open this URL in Google Chrome, Safari, or Microsoft Edge.

### 4. Running surveillance
1. If prompted, click **Allow** when the browser requests access to your camera.
2. The AI Core will finish bootloading in a few seconds (you will see the "AI Core: Ready" status in the header).
3. Click the **"Start Surveillance"** button to activate the camera and begin live scanning.
4. Use the **Volume icon** in the top-right to mute/unmute warning sounds.

---

## 🛠️ Architecture & Core Components

Sentinel is structured as a single-page React + Vite + TypeScript application:

*   **[src/App.tsx](file:///Users/horatiubudai/ceo/Pericle/Sentinel/src/App.tsx)**: Main application logic, including the camera device stream loop, the TensorFlow detection interval loop, the sound synthesizers (Web Audio API), the threat level calculation, and the state managers.
*   **[src/App.css](file:///Users/horatiubudai/ceo/Pericle/Sentinel/src/App.css)**: Visual styles (HUD overlays, corners, threat metric badges, glassmorphic cards, layout).
*   **[src/index.css](file:///Users/horatiubudai/ceo/Pericle/Sentinel/src/index.css)**: App design system variables (color codes, cyberpunk fonts, custom scrollbars, animations).
*   **[index.html](file:///Users/horatiubudai/ceo/Pericle/Sentinel/index.html)**: Main entry document containing loaded Google Fonts (`Orbitron` & `Inter`).

---

## 🧠 AI Engine Logic & Model Replacement Guide

### Where the Object Detection Logic Lives

All machine learning execution is contained in **[src/App.tsx](file:///Users/horatiubudai/ceo/Pericle/Sentinel/src/App.tsx)**:
1.  **Loading the Model**: On mounting, `useEffect` triggers `cocoSsd.load({ base: 'mobilenet_v2' })` to initialize the neural network. Mobilenet V2 is optimized for browser/mobile CPU/GPU environments.
2.  **Surveillance Loop**: The `detectFrameLoop` method executes predictions via a `requestAnimationFrame` loop. To keep the UI running at a buttery-smooth 60fps, we throttle the AI detections to run once every **120ms** (approx. 8 times a second), which is optimal for macOS devices.
3.  **Inference Execution**: `model.detect(video)` parses the HTML5 `<video>` frames directly in the browser using the WebGL backend, returning classification classes, probability scores, and pixel boundary arrays (`bbox: [x, y, width, height]`).

---

### How to Upgrade/Replace with a Stronger AI Model

If you wish to scale Sentinel to a more powerful model, here are three progressive upgrade paths:

#### Option A: Transition to MediaPipe (Object Detector Task)
Google's MediaPipe is the successor to TensorFlow.js for lightweight on-device vision tasks. It runs faster on macOS WebGL/WebGPU.
1.  Install `@mediapipe/tasks-vision`:
    ```bash
    npm install @mediapipe/tasks-vision
    ```
2.  In `src/App.tsx`, replace the `cocoSsd` initialization with MediaPipe's `FilesetResolver` and `ObjectDetector`:
    ```typescript
    import { ObjectDetector, FilesetResolver } from '@mediapipe/tasks-vision';

    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    );
    const detector = await ObjectDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite",
        delegate: "GPU"
      },
      scoreThreshold: 0.5,
      runningMode: "VIDEO"
    });
    ```
3.  Run detection on video frames:
    ```typescript
    const result = detector.detectForVideo(videoElement, timestamp);
    ```

#### Option B: Deploy Custom YOLO models (e.g. YOLOv8 / YOLOv11)
To detect custom security threats (e.g., custom tools, specific badges, or posture):
1.  Train a custom model in PyTorch and export it to **TensorFlow.js graph model format** using `ultralytics`:
    ```bash
    yolo export model=yolov8n.pt format=tfjs
    ```
2.  Host the output files (`model.json` and weight `.bin` shards) locally in your Vite `/public` directory.
3.  Load the custom model in `App.tsx`:
    ```typescript
    import * as tf from '@tensorflow/tfjs';
    const customModel = await tf.loadGraphModel('/yolov8n_web_model/model.json');
    ```
4.  Write a custom post-processing utility to perform Non-Maximum Suppression (NMS) on the raw tensor predictions returned by `customModel.execute(tensor)`.

#### Option C: WebSocket / HTTP Backend (e.g. YOLOv8-PyTorch or Local LLaVA)
If you want to use heavy state-of-the-art vision models (like YOLOv8-X, RT-DETR, or local Vision-LLMs like LLaVA/Ollama) that exceed browser capacity:
1.  Run a local Python server (Flask/FastAPI) that captures video frames via WebSockets, runs inference on your Mac's M1/M2/M3 Unified Memory GPU using PyTorch, and sends back the detections.
2.  In `App.tsx`, instead of loading a local model, connect to the socket:
    ```typescript
    const socket = new WebSocket('ws://localhost:8000/stream');
    
    // Periodically send frame snips:
    canvasContext.drawImage(video, 0, 0, 640, 480);
    canvas.toBlob((blob) => socket.send(blob), 'image/jpeg');

    socket.onmessage = (event) => {
      const predictions = JSON.parse(event.data);
      drawBoundingBoxes(predictions);
      processModelDetections(predictions);
    };
    ```
