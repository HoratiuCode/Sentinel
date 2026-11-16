import { useState, useEffect, useRef } from 'react';
// Performance: TensorFlow.js + models are heavy (~1.3MB). Load them lazily via
// dynamic import inside initAI so the UI shell paints immediately instead of
// blocking on the full bundle. Types are imported separately (erased at build).
import type * as TfTypes from '@tensorflow/tfjs';
import type * as CocoSsdTypes from '@tensorflow-models/coco-ssd';
import type * as MobilenetTypes from '@tensorflow-models/mobilenet';
let tf: typeof TfTypes;
let cocoSsd: typeof CocoSsdTypes;
let mobilenet: typeof MobilenetTypes;
import { 
  Camera, 
  CameraOff, 
  Shield, 
  ShieldAlert, 
  ShieldCheck, 
  Activity, 
  Trash2, 
  Play, 
  Square, 
  Volume2, 
  VolumeX, 
  Info,
  Settings,
  Brain,
  Sliders,
  X,
  Check,
  ExternalLink
} from 'lucide-react';
import './App.css';

// Cosine similarity helper for high-dimensional embedding matching
function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Interfaces
interface LogEvent {
  id: string;
  timestamp: string;
  message: string;
  type: 'info' | 'low' | 'medium' | 'high';
  object?: string;
  confidence?: number;
}

interface DetectionSummary {
  [key: string]: number;
}

interface LearnedObject {
  id: string;
  label: string;
  embedding: number[];
  timestamp: string;
  thumbnail: string; // Base64 thumbnail data URL
}

interface RegisteredPerson {
  id: string;
  name: string;
  role: 'friendly' | 'danger';
  embedding: number[];
  thumbnail: string; // Base64 thumbnail data URL
  timestamp: string;
}

interface LabelingTarget {
  prediction: CocoSsdTypes.DetectedObject;
  thumbnailUrl: string;
  cropCanvas: HTMLCanvasElement;
}

// Pre-defined classes in COCO-SSD database that the user can pick to alert on
const STANDARD_ALERT_OPTIONS = [
  'person', 'knife', 'scissors', 'backpack', 'handbag', 'suitcase', 
  'cell phone', 'laptop', 'book', 'umbrella', 'bottle', 'cup'
];

export default function App() {
  // Hash Routing
  const [currentRoute, setCurrentRoute] = useState<string>(window.location.hash || '#surveillance');

  // AI Core Models
  const [model, setModel] = useState<CocoSsdTypes.ObjectDetection | null>(null);
  const [mobilenetModel, setMobilenetModel] = useState<any>(null);
  const [modelType, setModelType] = useState<'mobilenet_v1' | 'mobilenet_v2' | 'lite_mobilenet_v2'>('mobilenet_v2');
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [modelError, setModelError] = useState<string | null>(null);
  
  // Camera State
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isCameraLoading, setIsCameraLoading] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  
  // Real-time Detections
  const [currentDetections, setCurrentDetections] = useState<any[]>([]);
  const [detectionSummary, setDetectionSummary] = useState<DetectionSummary>({});
  const [inferenceTime, setInferenceTime] = useState<number>(0);
  const [fps, setFps] = useState<number>(0);

  // System Security Status
  const [riskLevel, setRiskLevel] = useState<'LOW' | 'MEDIUM' | 'HIGH'>('LOW');
  const [riskReason, setRiskReason] = useState('System secure. Camera offline.');
  const [events, setEvents] = useState<LogEvent[]>([]);
  const [isMuted, setIsMuted] = useState(false);

  // --- Dynamic Security Settings ---
  const [minConfidence, setMinConfidence] = useState<number>(0.35);
  const [detectionInterval, setDetectionInterval] = useState<number>(120);
  const [persistentPersonThreshold, setPersistentPersonThreshold] = useState<number>(5);
  const [multiplePeopleCount, setMultiplePeopleCount] = useState<number>(2);
  const [similarityThreshold, setSimilarityThreshold] = useState<number>(0.83);
  const [alertClasses, setAlertClasses] = useState<string[]>(['person', 'knife', 'scissors', 'suitcase']);
  
  // --- Few-Shot Interactive Learning ---
  const [isLabelingModalOpen, setIsLabelingModalOpen] = useState(false);
  const [labelingTarget, setLabelingTarget] = useState<LabelingTarget | null>(null);
  const [customLabelInput, setCustomLabelInput] = useState('');
  const [learnedObjects, setLearnedObjects] = useState<LearnedObject[]>([]);
  
  // --- Personnel Registry Database ---
  const [registeredPeople, setRegisteredPeople] = useState<RegisteredPerson[]>([]);
  const [uploadName, setUploadName] = useState('');
  const [uploadRole, setUploadRole] = useState<'friendly' | 'danger'>('friendly');

  // References
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationFrameId = useRef<number | null>(null);
  
  const lastRiskLevel = useRef<'LOW' | 'MEDIUM' | 'HIGH'>('LOW');
  const personPresentFrames = useRef<number>(0);
  const objectHistory = useRef<Record<string, number>>({});
  const lastDetectionTime = useRef<number>(0);
  const fpsFrameCount = useRef<number>(0);
  const fpsLastTime = useRef<number>(0);
  const allPredictionsRef = useRef<any[]>([]);
  // Improvement (B): persistent track IDs across frames via IoU association
  const trackedBoxesRef = useRef<{ id: number; bbox: number[]; class: string; age: number }[]>([]);
  const nextTrackId = useRef<number>(1);

  // Web Audio Alert warnings synthesizer
  const playBeep = (frequency = 800, duration = 0.1, type: OscillatorType = 'sine') => {
    if (isMuted) return;
    try {
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      const oscillator = audioCtx.createOscillator();
      const gainNode = audioCtx.createGain();
      
      oscillator.type = type;
      oscillator.frequency.value = frequency;
      
      gainNode.gain.setValueAtTime(0.04, audioCtx.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.00001, audioCtx.currentTime + duration);
      
      oscillator.connect(gainNode);
      gainNode.connect(audioCtx.destination);
      
      oscillator.start();
      oscillator.stop(audioCtx.currentTime + duration);
    } catch (e) {
      console.warn('Audio blocked or not supported by browser:', e);
    }
  };

  // Warning Sound Combos
  const triggerNewDetectionSound = (isHighPriority = false) => {
    if (isHighPriority) {
      playBeep(900, 0.15, 'square');
      setTimeout(() => playBeep(1200, 0.15, 'square'), 100);
    } else {
      playBeep(880, 0.08, 'sine');
      setTimeout(() => playBeep(1100, 0.08, 'sine'), 60);
    }
  };

  const triggerRiskChangeSound = (level: 'LOW' | 'MEDIUM' | 'HIGH') => {
    if (level === 'HIGH') {
      playBeep(523.25, 0.35, 'sawtooth'); // C5
      setTimeout(() => playBeep(493.88, 0.35, 'sawtooth'), 150); // B4
    } else if (level === 'MEDIUM') {
      playBeep(440, 0.25, 'triangle'); // A4
    } else {
      playBeep(587.33, 0.1, 'sine'); // D5
    }
  };

  // Unified settings state writer
  const updateSetting = (key: string, value: any) => {
    const currentSettings = {
      minConfidence,
      detectionInterval,
      persistentPersonThreshold,
      multiplePeopleCount,
      similarityThreshold,
      alertClasses
    };
    const updated = { ...currentSettings, [key]: value };
    
    // Update local state
    if (key === 'minConfidence') setMinConfidence(value);
    if (key === 'detectionInterval') setDetectionInterval(value);
    if (key === 'persistentPersonThreshold') setPersistentPersonThreshold(value);
    if (key === 'multiplePeopleCount') setMultiplePeopleCount(value);
    if (key === 'similarityThreshold') setSimilarityThreshold(value);
    if (key === 'alertClasses') setAlertClasses(value);
    
    // Save to storage
    localStorage.setItem('sentinel_settings', JSON.stringify(updated));
  };

  // Listen to hash changes for routing
  useEffect(() => {
    const handleHashChange = () => {
      setCurrentRoute(window.location.hash || '#surveillance');
    };
    
    if (!window.location.hash) {
      window.location.hash = '#surveillance';
    }
    
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  // Load settings and sync databases on mount
  useEffect(() => {
    // 1. Restore settings
    const storedSettings = localStorage.getItem('sentinel_settings');
    if (storedSettings) {
      try {
        const parsed = JSON.parse(storedSettings);
        if (parsed.minConfidence !== undefined) setMinConfidence(parsed.minConfidence);
        if (parsed.detectionInterval !== undefined) setDetectionInterval(parsed.detectionInterval);
        if (parsed.persistentPersonThreshold !== undefined) setPersistentPersonThreshold(parsed.persistentPersonThreshold);
        if (parsed.multiplePeopleCount !== undefined) setMultiplePeopleCount(parsed.multiplePeopleCount);
        if (parsed.similarityThreshold !== undefined) setSimilarityThreshold(parsed.similarityThreshold);
        if (parsed.alertClasses !== undefined) setAlertClasses(parsed.alertClasses);
      } catch (e) {
        console.error('Failed to parse stored settings:', e);
      }
    }

    // 2. Restore objects database
    const storedObjects = localStorage.getItem('sentinel_learned_objects');
    if (storedObjects) {
      try {
        setLearnedObjects(JSON.parse(storedObjects));
      } catch (e) {
        console.error('Failed to parse learned objects:', e);
      }
    }

    // 3. Restore personnel registry
    const storedPeople = localStorage.getItem('sentinel_registered_people');
    if (storedPeople) {
      try {
        setRegisteredPeople(JSON.parse(storedPeople));
      } catch (e) {
        console.error('Failed to parse registered people:', e);
      }
    }
  }, []);

  // Listen to storage changes to sync tabs in real time
  useEffect(() => {
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'sentinel_settings') {
        try {
          const parsed = JSON.parse(e.newValue || '{}');
          if (parsed.minConfidence !== undefined) setMinConfidence(parsed.minConfidence);
          if (parsed.detectionInterval !== undefined) setDetectionInterval(parsed.detectionInterval);
          if (parsed.persistentPersonThreshold !== undefined) setPersistentPersonThreshold(parsed.persistentPersonThreshold);
          if (parsed.multiplePeopleCount !== undefined) setMultiplePeopleCount(parsed.multiplePeopleCount);
          if (parsed.similarityThreshold !== undefined) setSimilarityThreshold(parsed.similarityThreshold);
          if (parsed.alertClasses !== undefined) setAlertClasses(parsed.alertClasses);
        } catch (err) {
          console.error('Failed to sync settings:', err);
        }
      } else if (e.key === 'sentinel_learned_objects') {
        try {
          setLearnedObjects(e.newValue ? JSON.parse(e.newValue) : []);
        } catch (err) {
          console.error('Failed to sync objects database:', err);
        }
      } else if (e.key === 'sentinel_registered_people') {
        try {
          setRegisteredPeople(e.newValue ? JSON.parse(e.newValue) : []);
        } catch (err) {
          console.error('Failed to sync personnel registry:', err);
        }
      }
    };
    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, []);

  // Initialize TensorFlow.js and load models
  useEffect(() => {
    const initAI = async () => {
      try {
        setIsModelLoading(true);
        setModel(null);
        setModelError(null);
        
        // Lazy-load TensorFlow + models (keeps initial bundle small)
        if (!tf) tf = await import('@tensorflow/tfjs');
        if (!cocoSsd) cocoSsd = await import('@tensorflow-models/coco-ssd');
        if (!mobilenet) mobilenet = await import('@tensorflow-models/mobilenet');
        
        await tf.ready();
        
        // Always load MobileNet v2 since it is used on settings page for uploads and camera page for frames
        if (!mobilenetModel) {
          const loadedMobileNet = await mobilenet.load({
            version: 2,
            alpha: 1.0
          });
          setMobilenetModel(loadedMobileNet);
        }
        
        // Load COCO-SSD only if active on surveillance page
        if (currentRoute === '#surveillance' && !model) {
          const loadedCoco = await cocoSsd.load({
            base: modelType
          });
          setModel(loadedCoco);
        }
        
        setIsModelLoading(false);
        addEventLog(`AI Core active. Loaded COCO-SSD (${modelType}) & MobileNet v2.`, 'info');
        playBeep(1000, 0.15, 'sine');
      } catch (err: any) {
        console.error('Failed to boot AI Core:', err);
        setModelError(`AI Engine core boot failed. Reason: ${err.message || err}`);
        setIsModelLoading(false);
      }
    };

    // Load MobileNet on settings page; Load both on surveillance page
    initAI();

    if (currentRoute === '#surveillance') {
      queryCameraDevices();
    }

    return () => {
      stopCamera();
    };
  }, [modelType, currentRoute]);

  // Query webcam devices
  const queryCameraDevices = async () => {
    try {
      const tempStream = await navigator.mediaDevices.getUserMedia({ video: true });
      tempStream.getTracks().forEach(track => track.stop());
      
      const allDevices = await navigator.mediaDevices.enumerateDevices();
      const videoInputs = allDevices.filter(device => device.kind === 'videoinput');
      setDevices(videoInputs);
      
      if (videoInputs.length > 0 && !selectedDeviceId) {
        setSelectedDeviceId(videoInputs[0].deviceId);
      }
    } catch (err) {
      console.warn('Failed to enumerate camera streams:', err);
    }
  };

  // Add event log helper
  const addEventLog = (
    message: string, 
    type: 'info' | 'low' | 'medium' | 'high', 
    objectName?: string, 
    confidence?: number
  ) => {
    const now = new Date();
    const timestamp = now.toTimeString().split(' ')[0];
    const newEvent: LogEvent = {
      id: Math.random().toString(36).substring(2, 9),
      timestamp,
      message,
      type,
      object: objectName,
      confidence
    };
    
    setEvents(prev => [newEvent, ...prev].slice(0, 100));
  };

  // Start Camera feed
  const startCamera = async () => {
    if (!model) return;
    
    setIsCameraLoading(true);
    setCameraError(null);
    stopCamera();
    
    try {
      const constraints = {
        video: selectedDeviceId 
          ? { deviceId: { exact: selectedDeviceId } } 
          : { facingMode: 'user' },
        audio: false
      };
      
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          if (videoRef.current) {
            videoRef.current.play();
            setIsCameraActive(true);
            setIsCameraLoading(false);
            
            if (canvasRef.current) {
              canvasRef.current.width = videoRef.current.videoWidth;
              canvasRef.current.height = videoRef.current.videoHeight;
            }
            
            addEventLog(`Perimeter camera active (${videoRef.current.videoWidth}x${videoRef.current.videoHeight})`, 'info');
            setRiskLevel('LOW');
            setRiskReason('System active. Perimeter secure.');
            
            lastDetectionTime.current = performance.now();
            fpsLastTime.current = performance.now();
            fpsFrameCount.current = 0;
            
            animationFrameId.current = requestAnimationFrame(detectFrameLoop);
          }
        };
      }
    } catch (err: any) {
      console.error('Camera startup failed:', err);
      setCameraError(
        err.name === 'NotAllowedError' 
          ? 'Webcam permission denied. Please allow camera access in browser and macOS System Settings.' 
          : `Webcam startup error: ${err.message}`
      );
      setIsCameraLoading(false);
    }
  };

  // Stop Camera surveillance
  const stopCamera = () => {
    if (animationFrameId.current) {
      cancelAnimationFrame(animationFrameId.current);
      animationFrameId.current = null;
    }
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    
    setIsCameraActive(false);
    setCurrentDetections([]);
    setDetectionSummary({});
    setRiskLevel('LOW');
    setRiskReason('System secure. Camera offline.');
    
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (ctx && canvas) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    
    personPresentFrames.current = 0;
    objectHistory.current = {};
    lastRiskLevel.current = 'LOW';
  };

  // Detection loop
  const detectFrameLoop = async (now: number) => {
    if (!videoRef.current || !model || !streamRef.current) return;
    
    // FPS counter
    fpsFrameCount.current++;
    if (now - fpsLastTime.current >= 1000) {
      setFps(Math.round((fpsFrameCount.current * 1000) / (now - fpsLastTime.current)));
      fpsFrameCount.current = 0;
      fpsLastTime.current = now;
    }

    const elapsed = now - lastDetectionTime.current;
    
    if (elapsed >= detectionInterval) {
      const video = videoRef.current;
      
      if (video.readyState === 4) {
        const startTime = performance.now();
        
        try {
          // Fix #1: wire the Min Detection Confidence slider into the model call.
          // Fall back to 0.10 floor so proposals still feed the override pass.
          const detectConf = Math.max(0.1, minConfidence);
          const rawPredictions = await model.detect(video, 30, detectConf);
          allPredictionsRef.current = rawPredictions;
          
          // Perform dynamic MobileNet custom label override logic
          const predictions = await overrideWithLearnedLabels(rawPredictions);

          // Improvement (B): associate boxes across frames via IoU for stable IDs
          const tracked = associateTracks(predictions);

          const endTime = performance.now();
          setInferenceTime(Math.round(endTime - startTime));
          
          // Draw overlays on screen
          drawBoundingBoxes(tracked);
          
          // Evaluate alarm contributions and logs
          processDetections(tracked);
        } catch (e) {
          console.error('Inference error in frame loop:', e);
        }
        
        lastDetectionTime.current = now;
      }
    }
    
    animationFrameId.current = requestAnimationFrame(detectFrameLoop);
  };

  // MobileNet cosine embedding comparison to override standard labels
  // Improvement (C): only run the heavy embedding pass on persons + the top-5
  // non-person boxes, and add an "unknown" band (D) so a weak match is not
  // force-labeled.
  const overrideWithLearnedLabels = async (predictions: CocoSsdTypes.DetectedObject[]): Promise<any[]> => {
    const video = videoRef.current;
    if (!video || !mobilenetModel) return predictions;

    const hasRegistry = registeredPeople.length > 0;
    const hasLearned = learnedObjects.length > 0;
    const needsEmbedding = hasRegistry || hasLearned;

    // Limit the expensive embedding pass to persons + top-5 non-person boxes
    const persons = predictions.filter(p => p.class.toLowerCase() === 'person');
    const others = predictions
      .filter(p => p.class.toLowerCase() !== 'person')
      .sort((a, b) => b.score - a.score)
      .slice(0, 5);
    const candidates = needsEmbedding ? [...persons, ...others] : [];

    // Keep index mapping so we can splice results back in order
    const candidateIndexes = predictions
      .map((p, i) => (candidates.includes(p) ? i : -1))
      .filter(i => i >= 0);

    const results = await Promise.all(
      predictions.map(async (pred, idx) => {
        const [x, y, w, h] = pred.bbox;
        if (!candidateIndexes.includes(idx) || w < 15 || h < 15) return pred;

        const cropCanvas = document.createElement('canvas');
        cropCanvas.width = 64;
        cropCanvas.height = 64;
        const ctx = cropCanvas.getContext('2d');
        if (!ctx) return pred;

        ctx.drawImage(video, x, y, w, h, 0, 0, 64, 64);

        const label = pred.class.toLowerCase();

        try {
          const imgTensor = tf.browser.fromPixels(cropCanvas);
          const activation = mobilenetModel.infer(imgTensor, true);
          const embeddingArray = Array.from(await activation.data()) as number[];

          imgTensor.dispose();
          activation.dispose();

          // 1. If it's a person, check custom personnel registry first
          if (label === 'person' && hasRegistry) {
            let bestSim = 0;
            let bestMatch: RegisteredPerson | null = null;

            for (const person of registeredPeople) {
              const sim = cosineSimilarity(embeddingArray, person.embedding);
              if (sim > bestSim) {
                bestSim = sim;
                bestMatch = person;
              }
            }

            if (bestSim >= similarityThreshold && bestMatch) {
              return {
                ...pred,
                class: bestMatch.name,
                score: bestSim,
                isCustom: true,
                isPersonIdentify: true,
                role: bestMatch.role
              };
            }

            // 1b. Uncertain band: weak match (similarity 0.70-0.83) -> "unknown"
            if (bestSim >= 0.70 && bestMatch) {
              return {
                ...pred,
                class: 'unknown',
                score: bestSim,
                isCustom: true,
                isPersonIdentify: false,
                role: bestMatch.role
              };
            }
          }

          // 2. Fallback or general match against learned objects database
          if (hasLearned) {
            let bestSim = 0;
            let bestMatch: LearnedObject | null = null;

            for (const obj of learnedObjects) {
              const sim = cosineSimilarity(embeddingArray, obj.embedding);
              if (sim > bestSim) {
                bestSim = sim;
                bestMatch = obj;
              }
            }

            if (bestSim >= similarityThreshold && bestMatch) {
              return {
                ...pred,
                class: bestMatch.label,
                score: bestSim,
                isCustom: true
              };
            }
          }
        } catch (err) {
          console.warn('Embedding evaluation exception:', err);
        }

        return pred;
      })
    );

    return results;
  };

  // Improvement (B): associate detections across frames by IoU overlap so each
  // real object keeps a stable trackId instead of flickering. Boxes that match
  // an existing track adopt its id; stale tracks decay and get dropped.
  const iou = (a: number[], b: number[]): number => {
    const [ax, ay, aw, ah] = a;
    const [bx, by, bw, bh] = b;
    const x1 = Math.max(ax, bx);
    const y1 = Math.max(ay, by);
    const x2 = Math.min(ax + aw, bx + bw);
    const y2 = Math.min(ay + ah, by + bh);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const union = aw * ah + bw * bh - inter;
    return union > 0 ? inter / union : 0;
  };

  const associateTracks = (predictions: any[]): any[] => {
    const prev = trackedBoxesRef.current;
    const matchedPrev = new Set<number>();
    const next: { id: number; bbox: number[]; class: string; age: number }[] = [];

    const out = predictions.map(pred => {
      const [x, y, w, h] = pred.bbox;
      let bestId = -1;
      let bestIoU = 0.3; // association threshold
      for (const t of prev) {
        const o = iou(pred.bbox, t.bbox);
        if (o > bestIoU) {
          bestIoU = o;
          bestId = t.id;
        }
      }
      let trackId: number;
      if (bestId >= 0) {
        matchedPrev.add(bestId);
        trackId = bestId;
      } else {
        trackId = nextTrackId.current++;
      }
      next.push({ id: trackId, bbox: [x, y, w, h], class: pred.class, age: 0 });
      return { ...pred, trackId };
    });

    // Carry over unmatched tracks (decay) so brief misses don't drop the label
    prev.forEach(t => {
      if (!matchedPrev.has(t.id)) {
        if (t.age < 5) {
          next.push({ ...t, age: t.age + 1 });
        }
      }
    });

    trackedBoxesRef.current = next;
    return out;
  };

  // Draw bounding boxes on Canvas overlay
  const drawBoundingBoxes = (predictions: any[]) => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!ctx || !canvas) return;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    predictions.forEach(prediction => {
      const [x, y, width, height] = prediction.bbox;
      const isPerson = prediction.class === 'person';
      
      const label = prediction.class.toLowerCase();
      const isAlert = alertClasses.includes(label);
      const isCustom = !!prediction.isCustom;
      const isPersonIdentify = !!prediction.isPersonIdentify;
      const role = prediction.role; // 'friendly' or 'danger'
      
      let color = '#ffffff'; // default white (monochrome)
      let bgLight = 'rgba(255, 255, 255, 0.04)';

      if (isPersonIdentify) {
        if (role === 'danger') {
          color = '#ffffff'; // danger target -> pure white
          bgLight = 'rgba(255, 255, 255, 0.08)';
        } else {
          color = '#d4d4d4'; // friendly target -> light grey
          bgLight = 'rgba(212, 212, 212, 0.06)';
        }
      } else if (isCustom) {
        color = '#a3a3a3'; // custom trained objects -> mid grey
        bgLight = 'rgba(163, 163, 163, 0.06)';
      } else if (isAlert) {
        color = '#ffffff'; // threat items -> pure white
        bgLight = 'rgba(255, 255, 255, 0.08)';
      } else if (isPerson) {
        color = '#a3a3a3'; // person -> mid grey
        bgLight = 'rgba(163, 163, 163, 0.06)';
      }
      
      // Box style
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.shadowBlur = 8;
      ctx.shadowColor = color;
      
      // Draw autofocus corners
      const len = Math.min(24, width / 4, height / 4);
      
      // Top Left
      ctx.beginPath();
      ctx.moveTo(x + len, y);
      ctx.lineTo(x, y);
      ctx.lineTo(x, y + len);
      ctx.stroke();
      
      // Top Right
      ctx.beginPath();
      ctx.moveTo(x + width - len, y);
      ctx.lineTo(x + width, y);
      ctx.lineTo(x + width, y + len);
      ctx.stroke();
      
      // Bottom Left
      ctx.beginPath();
      ctx.moveTo(x + len, y + height);
      ctx.lineTo(x, y + height);
      ctx.lineTo(x, y + height - len);
      ctx.stroke();
      
      // Bottom Right
      ctx.beginPath();
      ctx.moveTo(x + width - len, y + height);
      ctx.lineTo(x + width, y + height);
      ctx.lineTo(x + width, y + height - len);
      ctx.stroke();
      
      // Tint fill
      ctx.shadowBlur = 0;
      ctx.fillStyle = bgLight;
      ctx.fillRect(x, y, width, height);
      
      // Draw label
      let labelText = `${isCustom ? '🧠 ' : ''}${prediction.class.toUpperCase()} ${Math.round(prediction.score * 100)}%`;
      if (isPersonIdentify) {
        labelText = `👤 [${role.toUpperCase()}] ${prediction.class.toUpperCase()} ${Math.round(prediction.score * 100)}%`;
      }
      
      ctx.font = 'bold 11px Orbitron';
      
      const padding = 6;
      const textWidth = ctx.measureText(labelText).width;
      const labelHeight = 20;
      const labelY = y - labelHeight > 0 ? y - labelHeight : y;
      
      ctx.fillStyle = color;
      ctx.fillRect(x, labelY, textWidth + padding * 2, labelHeight);
      
      ctx.fillStyle = '#08090e';
      ctx.fillText(labelText, x + padding, labelY + 14);
      
      if (width > 80 && height > 60) {
        ctx.font = '9px system-ui';
        ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
        ctx.fillText('CLICK TO TEACH', x + 5, y + height - 8);
      }
    });
  };

  // Process risk metrics
  const processDetections = (predictions: any[]) => {
    setCurrentDetections(predictions);
    
    const summary: DetectionSummary = {};
    predictions.forEach(p => {
      const label = p.class.toLowerCase();
      summary[label] = (summary[label] || 0) + 1;
    });
    setDetectionSummary(summary);
    
    const activeClasses = predictions.map(p => p.class.toLowerCase());
    // Fix #2 / H: a recognized person's class becomes their name, so count
    // persons via the isPersonIdentify flag OR the literal 'person' class.
    const personCount = predictions.filter(p => p.class.toLowerCase() === 'person' || p.isPersonIdentify).length;
    
    // Check identified people
    const dangerPeople = predictions.filter(p => p.isPersonIdentify && p.role === 'danger');
    const friendlyPeopleCount = predictions.filter(p => p.isPersonIdentify && p.role === 'friendly').length;
    const unidentifiedPeopleCount = personCount - friendlyPeopleCount - dangerPeople.length;
    const hasUnidentifiedPerson = unidentifiedPeopleCount > 0;
    
    if (hasUnidentifiedPerson || dangerPeople.length > 0) {
      personPresentFrames.current += 1;
    } else {
      personPresentFrames.current = 0;
    }

    let computedRisk: 'LOW' | 'MEDIUM' | 'HIGH' = 'LOW';
    let computedReason = 'Surveillance active. Scanning perimeter.';
    
    // Check general alert category targets currently in frame
    const activeAlertTriggers = predictions.filter(p => {
      const label = p.class.toLowerCase();
      return alertClasses.includes(label) && label !== 'person';
    });

    const framesToEscalate = Math.round((persistentPersonThreshold * 1000) / detectionInterval);

    if (dangerPeople.length > 0) {
      computedRisk = 'HIGH';
      computedReason = `CRITICAL ALERT: Blacklisted threat target (${dangerPeople.map(d => d.class.toUpperCase()).join(', ')}) detected in secure zone!`;
    } else if (activeAlertTriggers.length > 0) {
      computedRisk = 'HIGH';
      computedReason = `CRITICAL DETECTED: Alert item (${activeAlertTriggers.map(t => t.class.toUpperCase()).join(', ')}) identified in zone.`;
    } else if (unidentifiedPeopleCount >= multiplePeopleCount) {
      // Improvement (F): require persistence before escalating multi-person to HIGH,
      // so someone briefly walking through frame doesn't fire a full alert.
      if (personPresentFrames.current > framesToEscalate) {
        computedRisk = 'HIGH';
        computedReason = `HIGH RISK: Multiple unidentified targets (${unidentifiedPeopleCount}) persistent in zone.`;
      } else {
        computedRisk = 'MEDIUM';
        computedReason = `MULTIPLE TARGETS: ${unidentifiedPeopleCount} unidentified detected (verifying persistence).`;
      }
    } else if (unidentifiedPeopleCount === 1) {
      if (personPresentFrames.current > framesToEscalate) {
        computedRisk = 'HIGH';
        computedReason = `HIGH RISK: Persistent unidentified target presence (> ${persistentPersonThreshold} seconds).`;
      } else {
        computedRisk = 'MEDIUM';
        computedReason = 'WARNING: Single unidentified target detected in security sector.';
      }
    } else if (friendlyPeopleCount > 0) {
      computedRisk = 'LOW';
      computedReason = `Secure scan. Authorized target(s) (${predictions.filter(p => p.isPersonIdentify && p.role === 'friendly').map(p => p.class.toUpperCase()).join(', ')}) monitored in sector.`;
    } else if (predictions.length > 0) {
      computedRisk = 'LOW';
      computedReason = `Secure scan. ${predictions.length} static object(s) monitored.`;
    } else {
      computedRisk = 'LOW';
      computedReason = 'Secure scan. No movement or targets identified.';
    }

    setRiskLevel(computedRisk);
    setRiskReason(computedReason);

    if (computedRisk !== lastRiskLevel.current) {
      triggerRiskChangeSound(computedRisk);
      const logType = computedRisk === 'HIGH' ? 'high' : computedRisk === 'MEDIUM' ? 'medium' : 'low';
      addEventLog(`PERIMETER UPDATE: Risk escalated to ${computedRisk} (${computedReason})`, logType);
      lastRiskLevel.current = computedRisk;
    }

    Object.keys(objectHistory.current).forEach(cls => {
      if (!activeClasses.includes(cls)) {
        objectHistory.current[cls] -= 1;
        if (objectHistory.current[cls] <= 0) {
          delete objectHistory.current[cls];
          addEventLog(`${cls.toUpperCase()} cleared from scan sector.`, 'info');
        }
      }
    });

    predictions.forEach(p => {
      const cls = p.class.toLowerCase();
      const scorePct = Math.round(p.score * 100);

      if (!objectHistory.current[cls]) {
        objectHistory.current[cls] = 12;
        const isWeapon = alertClasses.includes(cls) && cls !== 'person';
        const logType = cls === 'person' ? 'medium' : isWeapon ? 'high' : 'low';
        
        addEventLog(`OBJECT DETECTED: ${cls.toUpperCase()} (${scorePct}% match)`, logType, cls, p.score);
        triggerNewDetectionSound(isWeapon || cls === 'person');
      } else {
        objectHistory.current[cls] = 12;
      }
    });
  };

  // Handle canvas click to crop bounding box of clicked object
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isCameraActive) return;
    
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    
    const rect = canvas.getBoundingClientRect();
    const clickX = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const clickY = ((e.clientY - rect.top) / rect.height) * canvas.height;
    
    const candidates = allPredictionsRef.current.filter(p => {
      const [x, y, w, h] = p.bbox;
      return clickX >= x && clickX <= x + w && clickY >= y && clickY <= y + h;
    });
    
    let targetPrediction: any = null;
    
    if (candidates.length > 0) {
      candidates.sort((a, b) => (a.bbox[2] * a.bbox[3]) - (b.bbox[2] * b.bbox[3]));
      targetPrediction = candidates[0];
    } else {
      let minDistance = 120;
      allPredictionsRef.current.forEach(p => {
        const [x, y, w, h] = p.bbox;
        const centerX = x + w / 2;
        const centerY = y + h / 2;
        const dist = Math.sqrt(Math.pow(clickX - centerX, 2) + Math.pow(clickY - centerY, 2));
        if (dist < minDistance) {
          minDistance = dist;
          targetPrediction = p;
        }
      });
    }
    
    if (!targetPrediction) {
      const cropSize = 160;
      const x = Math.max(0, Math.min(clickX - cropSize / 2, canvas.width - cropSize));
      const y = Math.max(0, Math.min(clickY - cropSize / 2, canvas.height - cropSize));
      
      targetPrediction = {
        bbox: [x, y, cropSize, cropSize],
        class: 'custom target',
        score: 0.20
      };
    }
    
    openLabelingModal(targetPrediction);
  };

  // Open labeling modal
  const openLabelingModal = (prediction: any) => {
    const video = videoRef.current;
    if (!video) return;
    
    const [x, y, w, h] = prediction.bbox;
    
    const cropCanvas = document.createElement('canvas');
    cropCanvas.width = Math.max(w, 80);
    cropCanvas.height = Math.max(h, 80);
    const ctx = cropCanvas.getContext('2d');
    
    if (ctx) {
      ctx.drawImage(
        video,
        x, y, w, h,
        0, 0, cropCanvas.width, cropCanvas.height
      );
      
      const thumbnailUrl = cropCanvas.toDataURL('image/jpeg');
      setLabelingTarget({
        prediction,
        thumbnailUrl,
        cropCanvas
      });
      setCustomLabelInput(prediction.class);
      setIsLabelingModalOpen(true);
      playBeep(700, 0.1, 'sine');
    }
  };

  // Save the custom label embedding
  const handleSaveCustomLabel = async () => {
    if (!labelingTarget || !mobilenetModel || !customLabelInput.trim()) return;
    
    const label = customLabelInput.trim().toLowerCase();
    
    try {
      const { cropCanvas, thumbnailUrl } = labelingTarget;
      
      const imgTensor = tf.browser.fromPixels(cropCanvas);
      const activation = mobilenetModel.infer(imgTensor, true);
      const embeddingArray = Array.from(await activation.data()) as number[];
      
      imgTensor.dispose();
      activation.dispose();
      
      // If we are labeling a person, save to personnel registry instead of objects database
      if (labelingTarget.prediction.class.toLowerCase() === 'person') {
        // Improvement (I): if the name already exists, retrain (replace) its embedding
        // instead of creating a duplicate entry — lets you correct misidentifications.
        const existingIdx = registeredPeople.findIndex(p => p.name.toLowerCase() === label);
        let updatedPeople;
        if (existingIdx >= 0) {
          updatedPeople = registeredPeople.map((p, i) =>
            i === existingIdx
              ? { ...p, embedding: embeddingArray, thumbnail: thumbnailUrl, timestamp: new Date().toLocaleTimeString() }
              : p
          );
          addEventLog(`PERSONNEL REGISTRY: Retrained ${label.toUpperCase()} embedding via camera tag`, 'info');
        } else {
          const newPerson: RegisteredPerson = {
            id: Math.random().toString(36).substring(2, 9),
            name: label,
            role: 'friendly', // Default to friendly, can be adjusted in settings
            embedding: embeddingArray,
            thumbnail: thumbnailUrl,
            timestamp: new Date().toLocaleTimeString()
          };
          updatedPeople = [newPerson, ...registeredPeople];
          addEventLog(`PERSONNEL REGISTRY: Registered ${label.toUpperCase()} as FRIENDLY via camera tag`, 'info');
        }
        setRegisteredPeople(updatedPeople);
        localStorage.setItem('sentinel_registered_people', JSON.stringify(updatedPeople));
      } else {
        const newLearned: LearnedObject = {
          id: Math.random().toString(36).substring(2, 9),
          label,
          embedding: embeddingArray,
          timestamp: new Date().toLocaleTimeString(),
          thumbnail: thumbnailUrl
        };
        const updatedLearned = [newLearned, ...learnedObjects];
        setLearnedObjects(updatedLearned);
        localStorage.setItem('sentinel_learned_objects', JSON.stringify(updatedLearned));
        
        if (!alertClasses.includes(label)) {
          const updatedClasses = [...alertClasses, label];
          updateSetting('alertClasses', updatedClasses);
        }
        addEventLog(`MACHINE LEARNING SYNC: Registered custom class label "${label.toUpperCase()}"`, 'high');
      }
      
      playBeep(1200, 0.1, 'sine');
      setTimeout(() => playBeep(1500, 0.1, 'sine'), 80);
      
      setIsLabelingModalOpen(false);
      setLabelingTarget(null);
    } catch (e) {
      console.error('Failed to train custom object:', e);
    }
  };

  // Delete custom learned object
  const handleDeleteLearned = (id: string, label: string) => {
    const updated = learnedObjects.filter(obj => obj.id !== id);
    setLearnedObjects(updated);
    localStorage.setItem('sentinel_learned_objects', JSON.stringify(updated));
    addEventLog(`DATABASE SYNC: Removed custom class "${label.toUpperCase()}"`, 'info');
    playBeep(500, 0.15, 'sine');
  };

  // Upload photo of a person from disk and register
  const handleUploadPersonPhoto = async (
    e: React.ChangeEvent<HTMLInputElement>,
    name: string,
    role: 'friendly' | 'danger'
  ) => {
    if (!e.target.files || e.target.files.length === 0 || !mobilenetModel) return;
    
    const file = e.target.files[0];
    const reader = new FileReader();
    
    reader.onload = async (event) => {
      const img = new Image();
      img.onload = async () => {
        const canvas = document.createElement('canvas');
        // Improvement (G): use MobileNet's native 224x224 input instead of 128,
        // improves face/object embedding quality for the registry match.
        const TARGET = 224;
        canvas.width = TARGET;
        canvas.height = TARGET;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        
        const size = Math.min(img.width, img.height);
        const sx = (img.width - size) / 2;
        const sy = (img.height - size) / 2;
        
        ctx.drawImage(
          img,
          sx, sy, size, size,
          0, 0, TARGET, TARGET
        );
        
        const thumbnailUrl = canvas.toDataURL('image/jpeg');
        
        try {
          const imgTensor = tf.browser.fromPixels(canvas);
          const activation = mobilenetModel.infer(imgTensor, true);
          const embeddingArray = Array.from(await activation.data()) as number[];
          
          imgTensor.dispose();
          activation.dispose();
          
          const newPerson: RegisteredPerson = {
            id: Math.random().toString(36).substring(2, 9),
            name: name.trim().toLowerCase(),
            role,
            embedding: embeddingArray,
            thumbnail: thumbnailUrl,
            timestamp: new Date().toLocaleTimeString()
          };
          
          const updatedPeople = [newPerson, ...registeredPeople];
          setRegisteredPeople(updatedPeople);
          localStorage.setItem('sentinel_registered_people', JSON.stringify(updatedPeople));
          
          addEventLog(`PERSONNEL REGISTRY: Registered "${name.toUpperCase()}" as ${role.toUpperCase()}`, 'info');
          playBeep(1200, 0.1, 'sine');
          setTimeout(() => playBeep(1500, 0.1, 'sine'), 80);
        } catch (err) {
          console.error('Embedding extraction failed for uploaded photo:', err);
        }
      };
      img.src = event.target?.result as string;
    };
    reader.readAsDataURL(file);
  };

  // Delete registered person
  const handleDeletePerson = (id: string, name: string) => {
    const updated = registeredPeople.filter(p => p.id !== id);
    setRegisteredPeople(updated);
    localStorage.setItem('sentinel_registered_people', JSON.stringify(updated));
    addEventLog(`PERSONNEL REGISTRY: Removed "${name.toUpperCase()}" from registry`, 'info');
    playBeep(500, 0.15, 'sine');
  };

  const handleDeviceChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedDeviceId(e.target.value);
    if (isCameraActive) {
      setTimeout(() => startCamera(), 100);
    }
  };

  const handleAlertClassToggle = (cls: string) => {
    const updatedClasses = alertClasses.includes(cls)
      ? alertClasses.filter(c => c !== cls)
      : [...alertClasses, cls];
    updateSetting('alertClasses', updatedClasses);
  };

  // RENDER ROUTE: Full Settings Configuration Page
  if (currentRoute === '#settings') {
    return (
      <div className="settings-page-container">
        <header className="app-header">
          <div className="brand-section">
            <Activity className="brand-icon" size={24} />
            <div className="brand-title-group">
              <h1 className="app-title">SENTINEL</h1>
              <span className="app-subtitle">Surveillance Configuration Center</span>
            </div>
          </div>
          <div className="button-group">
            <button 
              className="back-btn" 
              onClick={() => window.location.hash = '#surveillance'}
            >
              ← Return to Surveillance
            </button>
            <button 
              className="btn btn-primary" 
              onClick={() => window.open('#settings', '_blank')}
              style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}
            >
              <ExternalLink size={14} />
              Open in New Tab
            </button>
          </div>
        </header>

        <div className="settings-grid">
          {/* Slider configurations card */}
          <section className="settings-page-card">
            <div className="settings-section-title" style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              <Sliders size={14} />
              Trigger Metrics & Latencies
            </div>
            
            <div className="settings-group">
              <label className="settings-label">
                <span>Min Camera Detection Confidence</span>
                <span className="settings-value">{Math.round(minConfidence * 100)}%</span>
              </label>
              <input 
                type="range" min="0.15" max="0.80" step="0.05"
                className="settings-slider"
                value={minConfidence}
                onChange={(e) => updateSetting('minConfidence', parseFloat(e.target.value))}
              />
            </div>

            <div className="settings-group">
              <label className="settings-label">
                <span>Surveillance Scan Latency (Frame Interval)</span>
                <span className="settings-value">{detectionInterval}ms</span>
              </label>
              <input 
                type="range" min="50" max="500" step="10"
                className="settings-slider"
                value={detectionInterval}
                onChange={(e) => updateSetting('detectionInterval', parseInt(e.target.value))}
              />
            </div>

            <div className="settings-group">
              <label className="settings-label">
                <span>Persistent Target Presence (High Danger Threshold)</span>
                <span className="settings-value">{persistentPersonThreshold} seconds</span>
              </label>
              <input 
                type="range" min="2" max="15" step="1"
                className="settings-slider"
                value={persistentPersonThreshold}
                onChange={(e) => updateSetting('persistentPersonThreshold', parseInt(e.target.value))}
              />
            </div>

            <div className="settings-group">
              <label className="settings-label">
                <span>Multiple Targets Limit (High Danger Trigger)</span>
                <span className="settings-value">{multiplePeopleCount} person(s)</span>
              </label>
              <input 
                type="range" min="1" max="5" step="1"
                className="settings-slider"
                value={multiplePeopleCount}
                onChange={(e) => updateSetting('multiplePeopleCount', parseInt(e.target.value))}
              />
            </div>

            <div className="settings-group">
              <label className="settings-label">
                <span>Custom Object Matching Similarity Threshold</span>
                <span className="settings-value">{Math.round(similarityThreshold * 100)}%</span>
              </label>
              <input 
                type="range" min="0.70" max="0.90" step="0.01"
                className="settings-slider"
                value={similarityThreshold}
                onChange={(e) => updateSetting('similarityThreshold', parseFloat(e.target.value))}
              />
            </div>
          </section>

          {/* Alarm categories checklist */}
          <section className="settings-page-card">
            <div className="settings-section-title">Critical Threat Categories</div>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>
              Select neural network classes that should immediately elevate danger levels when detected in secure zone:
            </p>
            <div className="checkbox-grid">
              {STANDARD_ALERT_OPTIONS.map(cls => {
                const checked = alertClasses.includes(cls);
                return (
                  <div 
                    key={cls} 
                    className={`checkbox-card ${checked ? 'checked' : ''}`}
                    onClick={() => handleAlertClassToggle(cls)}
                  >
                    <div className="checkbox-box">
                      {checked && <Check size={10} />}
                    </div>
                    <span>{cls}</span>
                  </div>
                );
              })}
            </div>
          </section>
        </div>

        <div className="settings-grid">
          {/* Personnel Access Registry (Face/Person ID) */}
          <section className="settings-page-card">
            <div className="settings-section-title" style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              <Brain size={14} />
              Personnel Access Registry (Face/Person ID)
            </div>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>
              Register people using photos to identify them and configure alert settings.
            </p>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginBottom: '1.25rem' }}>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <input 
                  type="text" 
                  className="input-cyber" 
                  placeholder="Person's Name (e.g. John)" 
                  value={uploadName} 
                  onChange={(e) => setUploadName(e.target.value)} 
                />
                <select 
                  className="camera-device-select"
                  style={{ minWidth: '120px' }}
                  value={uploadRole}
                  onChange={(e) => setUploadRole(e.target.value as 'friendly' | 'danger')}
                >
                  <option value="friendly">Friendly</option>
                  <option value="danger">Danger</option>
                </select>
              </div>
              
              <div style={{ position: 'relative' }}>
                <input 
                  type="file" 
                  accept="image/*"
                  id="person-photo-upload"
                  style={{ display: 'none' }}
                  onChange={(e) => {
                    if (uploadName.trim()) {
                      handleUploadPersonPhoto(e, uploadName, uploadRole);
                      setUploadName('');
                    } else {
                      alert('Please enter a name first.');
                    }
                  }}
                  disabled={!mobilenetModel}
                />
                <label 
                  htmlFor="person-photo-upload" 
                  className="btn btn-primary"
                  style={{ 
                    display: 'flex', 
                    justifyContent: 'center', 
                    cursor: uploadName.trim() ? 'pointer' : 'not-allowed',
                    opacity: uploadName.trim() ? 1 : 0.5
                  }}
                >
                  {!mobilenetModel ? 'Loading ML Engine...' : 'Select Photo & Register'}
                </label>
              </div>
            </div>

            <div className="settings-section-title" style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
              Registered Directory ({registeredPeople.length})
            </div>
            
            <div className="learned-list-container" style={{ maxHeight: '200px' }}>
              {registeredPeople.length === 0 ? (
                <div className="learned-empty">No personnel registered. Add name and photo above.</div>
              ) : (
                <div className="learned-db-grid" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(90px, 1fr))' }}>
                  {registeredPeople.map(person => (
                    <div className="learned-card" key={person.id} style={{ borderColor: person.role === 'danger' ? 'rgba(255, 255, 255, 0.6)' : 'rgba(212, 212, 212, 0.4)' }}>
                      <button 
                        className="learned-card-delete"
                        onClick={() => handleDeletePerson(person.id, person.name)}
                        title="Delete registry entry"
                      >
                        <X size={10} />
                      </button>
                      <img src={person.thumbnail} className="learned-card-img" alt={person.name} />
                      <span className="learned-card-label" title={person.name}>{person.name}</span>
                      <span 
                        style={{ 
                          fontSize: '0.65rem', 
                          fontWeight: 'bold', 
                          color: person.role === 'danger' ? 'var(--accent-red)' : 'var(--accent-green)',
                          textTransform: 'uppercase'
                        }}
                      >
                        {person.role}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </section>

          {/* Custom ML Database Manager */}
          <section className="settings-page-card">
            <div className="settings-section-title" style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              <Brain size={14} />
              Custom ML Objects Database ({learnedObjects.length})
            </div>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>
              View and manage custom objects taught directly from the surveillance feed canvas.
            </p>
            <div className="learned-list-container" style={{ maxHeight: '350px' }}>
              {learnedObjects.length === 0 ? (
                <div className="learned-empty">
                  No custom objects saved in database yet. Switch to the camera page to tag and teach Sentinel.
                </div>
              ) : (
                <div className="learned-db-grid" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(90px, 1fr))' }}>
                  {learnedObjects.map(obj => (
                    <div className="learned-card" key={obj.id}>
                      <button 
                        className="learned-card-delete"
                        onClick={() => handleDeleteLearned(obj.id, obj.label)}
                        title="Delete custom label"
                      >
                        <X size={10} />
                      </button>
                      <img src={obj.thumbnail} className="learned-card-img" alt={obj.label} />
                      <span className="learned-card-label" title={obj.label}>{obj.label}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </section>
        </div>

        <footer className="app-footer">
          <span>Sentinel Settings Page. Configured rules are auto-saved and synced across windows in real time.</span>
        </footer>
      </div>
    );
  }

  // RENDER ROUTE: Main Camera Surveillance Terminal
  return (
    <div className="app-container">
      {/* Main Terminal Header */}
      <header className="app-header">
        <div className="brand-section">
          <Activity className="brand-icon" size={24} />
          <div className="brand-title-group">
            <h1 className="app-title">SENTINEL</h1>
            <span className="app-subtitle">AI Surveillance Terminal v1.1.0</span>
          </div>
        </div>

        <div className="system-status">
          <div className="status-indicator">
            <span className={`status-dot ${isModelLoading ? 'loading' : model ? 'active' : ''}`} />
            <span>AI CORE: {isModelLoading ? 'LOADING' : model ? `READY (${modelType.toUpperCase()})` : 'OFFLINE'}</span>
          </div>
          
          <div className="status-indicator">
            <span className={`status-dot ${isCameraActive ? 'active' : ''}`} />
            <span>CAM: {isCameraActive ? 'ONLINE' : 'OFFLINE'}</span>
          </div>

          <button 
            className="btn btn-secondary" 
            onClick={() => setIsMuted(!isMuted)}
            title={isMuted ? "Unmute Alerts" : "Mute Alerts"}
            style={{ padding: '0.5rem' }}
          >
            {isMuted ? <VolumeX size={16} /> : <Volume2 size={16} />}
          </button>

          <button 
            className="btn btn-secondary" 
            onClick={() => window.open('#settings', '_blank')}
            title="Open Settings in New Tab"
            style={{ padding: '0.5rem', borderColor: 'rgba(255, 255, 255, 0.2)' }}
          >
            <Settings size={16} style={{ color: 'var(--accent-cyan)' }} />
          </button>
        </div>
      </header>

      {/* Dashboard Core */}
      <div className="dashboard-grid">
        {/* LEFT COLUMN: Camera Telemetry Feed */}
        <section className="camera-panel">
          <div className="panel-header">
            <h2 className="panel-title">
              <Camera size={18} />
              LIVE surveillance FEED
            </h2>
            
            {isCameraActive && (
              <div className="hud-rec-indicator">
                <span className="hud-rec-dot" />
                <span>SCANNER ENGAGED</span>
              </div>
            )}
          </div>

          {/* Viewport container */}
          <div className="camera-viewport-container">
            <div className="hud-corner hud-tl" />
            <div className="hud-corner hud-tr" />
            <div className="hud-corner hud-bl" />
            <div className="hud-corner hud-br" />

            <video
              ref={videoRef}
              className="video-feed"
              playsInline
              muted
              style={{ display: isCameraActive ? 'block' : 'none' }}
            />

            <canvas
              ref={canvasRef}
              className="canvas-overlay"
              onClick={handleCanvasClick}
              style={{ display: isCameraActive ? 'block' : 'none', cursor: 'crosshair' }}
            />

            {/* Spinner boot loadings */}
            {(isModelLoading) && (
              <div className="loading-overlay">
                <div className="spinner" />
                <span className="loading-text">BOOTING SYSTEM AI CORES...</span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Loading COCO-SSD & MobileNet V2 models</span>
              </div>
            )}

            {isCameraLoading && (
              <div className="loading-overlay">
                <div className="spinner" />
                <span className="loading-text">COMMUNICATING WITH WEBCAM...</span>
              </div>
            )}

            {!isCameraActive && !isCameraLoading && !isModelLoading && (
              <div className="loading-overlay" style={{ background: 'rgba(5, 5, 8, 0.85)' }}>
                <CameraOff size={48} className="text-muted" style={{ marginBottom: '0.5rem', opacity: 0.5 }} />
                <span className="loading-text" style={{ color: 'var(--text-secondary)' }}>
                  TERMINAL SYSTEM OFFLINE
                </span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                  Click "Start Surveillance" below to activate camera
                </span>
              </div>
            )}

            {cameraError && (
              <div className="loading-overlay">
                <div className="error-message">{cameraError}</div>
                <button className="btn btn-secondary" style={{ marginTop: '0.5rem' }} onClick={startCamera}>
                  Try Again
                </button>
              </div>
            )}

            {modelError && (
              <div className="loading-overlay">
                <div className="error-message">{modelError}</div>
              </div>
            )}

            {/* Overlaid stats summary bar */}
            {isCameraActive && (
              <div className="hud-stats">
                <div className="hud-stat-item">
                  <span>MODEL:</span>
                  <span className="hud-stat-value">{modelType.toUpperCase()}</span>
                </div>
                <div className="hud-stat-item">
                  <span>FPS:</span>
                  <span className="hud-stat-value">{fps}</span>
                </div>
                <div className="hud-stat-item">
                  <span>INFERENCE:</span>
                  <span className="hud-stat-value">{inferenceTime}ms</span>
                </div>
                <div className="hud-stat-item">
                  <span>RESOLUTION:</span>
                  <span className="hud-stat-value">
                    {videoRef.current ? `${videoRef.current.videoWidth}x${videoRef.current.videoHeight}` : '1280x720'}
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Action controls */}
          <div className="controls-bar">
            <div className="button-group">
              {!isCameraActive ? (
                <button
                  className="btn btn-primary"
                  onClick={startCamera}
                  disabled={!!isModelLoading || !!modelError}
                >
                  <Play size={16} />
                  Start Surveillance
                </button>
              ) : (
                <button
                  className="btn btn-danger"
                  onClick={stopCamera}
                >
                  <Square size={16} />
                  Stop Surveillance
                </button>
              )}
            </div>

            <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
              <select
                className="camera-device-select"
                onChange={(e) => setModelType(e.target.value as any)}
                value={modelType}
                disabled={isModelLoading}
                title="Select Detection Model"
              >
                <option value="mobilenet_v2">MobileNet V2 (Standard)</option>
                <option value="mobilenet_v1">MobileNet V1 (Legacy)</option>
                <option value="lite_mobilenet_v2">Lite MobileNet V2 (Fastest)</option>
              </select>

              {devices.length > 1 && (
                <select
                  className="camera-device-select"
                  onChange={handleDeviceChange}
                  value={selectedDeviceId}
                  disabled={isCameraLoading}
                >
                  {devices.map(device => (
                    <option key={device.deviceId} value={device.deviceId}>
                      {device.label || `Camera ${device.deviceId.slice(0, 5)}`}
                    </option>
                  ))}
                </select>
              )}
            </div>
          </div>

          {/* Objects in Frame List */}
          {isCameraActive && (
            <div className="detections-summary-card">
              <span className="summary-title" style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                <Brain size={14} style={{ color: 'var(--accent-cyan)' }} />
                Detections in Area (Click Bounding Box or Tag below to Teach)
              </span>
              {currentDetections.length === 0 ? (
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Scan area clear. No objects detected.</span>
              ) : (
                <div className="summary-list">
                  {Object.entries(detectionSummary).map(([cls, count]) => {
                    const match = currentDetections.find(p => p.class.toLowerCase() === cls);
                    const isCustom = match?.isCustom;
                    return (
                      <div className="summary-tag" key={cls} style={{ borderColor: isCustom ? 'var(--accent-cyan)' : undefined }}>
                        <span className="tag-label" style={{ color: isCustom ? '#ffffff' : undefined }}>
                          {isCustom ? `🧠 ${cls}` : cls}
                        </span>
                        <span className="tag-count" style={{ backgroundColor: isCustom ? 'rgba(255, 255, 255, 0.2)' : undefined, color: isCustom ? '#ffffff' : undefined }}>
                          x{count}
                        </span>
                        {match && (
                          <button
                            className="tag-rename-btn"
                            onClick={() => openLabelingModal(match)}
                            title={`Teach Sentinel another label for this ${cls}`}
                          >
                            <Brain size={11} />
                            Teach
                          </button>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}
        </section>

        {/* RIGHT COLUMN: Threat status & events */}
        <aside className="sidebar">
          {/* Threat Metric Card */}
          <div className={`risk-card risk-${riskLevel.toLowerCase()}`}>
            <div className="risk-header">
              <span className="risk-label">System Threat Level</span>
              <span 
                className={`risk-badge ${riskLevel.toLowerCase()}`}
                style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}
              >
                {riskLevel === 'LOW' && <ShieldCheck size={16} />}
                {riskLevel === 'MEDIUM' && <Shield size={16} />}
                {riskLevel === 'HIGH' && <ShieldAlert size={16} />}
                {riskLevel}
              </span>
            </div>

            <div className="risk-meter">
              <div className={`risk-meter-fill ${riskLevel.toLowerCase()}`} />
            </div>

            <p className="risk-description">
              {riskReason}
            </p>
          </div>

          {/* Event Log Panel */}
          <div className="log-card">
            <div className="log-header">
              <h2 className="log-title">
                <Info size={16} style={{ color: 'var(--accent-cyan)' }} />
                EVENT SYSTEM LOG
              </h2>
              {events.length > 0 && (
                <button className="clear-log-btn" onClick={() => setEvents([])}>
                  <Trash2 size={12} />
                  Clear Logs
                </button>
              )}
            </div>

            <div className="log-list">
              {events.length === 0 ? (
                <div className="log-empty">
                  <Activity size={24} style={{ opacity: 0.3, animation: 'pulse 2s infinite' }} />
                  <span>No perimeter events recorded.</span>
                </div>
              ) : (
                events.map(event => (
                  <div className="log-item" key={event.id}>
                    <div className="log-item-left">
                      <span className="log-timestamp">[{event.timestamp}]</span>
                      <span className="log-message">{event.message}</span>
                    </div>
                    {event.object && (
                      <span className={`log-badge-risk ${event.type}`}>
                        {event.type.toUpperCase()}
                      </span>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>
        </aside>
      </div>

      {/* --- TEACH OBJECT MODAL OVERLAY (Few-Shot Labeling Modal) --- */}
      {isLabelingModalOpen && labelingTarget && (
        <div className="modal-overlay" onClick={() => { setIsLabelingModalOpen(false); setLabelingTarget(null); }}>
          <div className="modal-container" style={{ maxWidth: '420px' }} onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3 className="modal-title">
                <Brain size={18} style={{ color: 'var(--accent)' }} />
                TEACH NEW OBJECT EMBEDDING
              </h3>
              <button className="close-btn" onClick={() => { setIsLabelingModalOpen(false); setLabelingTarget(null); }}>
                <X size={16} />
              </button>
            </div>

            <div className="crop-preview-container">
              <img src={labelingTarget.thumbnailUrl} className="crop-preview-img" alt="Object crop" />
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textAlign: 'center' }}>
                Sentinel captured this image patch. Teach the local AI to recognize this class.
              </p>
            </div>

            <div className="form-group">
              <label className="settings-label" style={{ marginBottom: '0.4rem' }}>Class Label (e.g. "my keys", "rufus")</label>
              <input 
                type="text" 
                className="input-cyber"
                value={customLabelInput}
                onChange={(e) => setCustomLabelInput(e.target.value)}
                placeholder="Enter custom object label"
                autoFocus
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleSaveCustomLabel();
                }}
              />
            </div>

            <div className="modal-actions">
              <button className="btn btn-secondary" onClick={() => { setIsLabelingModalOpen(false); setLabelingTarget(null); }}>
                Cancel
              </button>
              <button 
                className="btn btn-primary" 
                style={{ backgroundColor: 'var(--accent-cyan)', color: '#fff', boxShadow: '0 0 10px rgba(255, 255, 255, 0.2)' }}
                onClick={handleSaveCustomLabel}
                disabled={!customLabelInput.trim()}
              >
                Learn Object
              </button>
            </div>
          </div>
        </div>
      )}

      <footer className="app-footer">
        <span>Sentinel Local surveillance Monitor running on macOS. Powered by TensorFlow.js.</span>
      </footer>
    </div>
  );
}
