
'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { SignAnimator, Sign_Default } from '@/lib/SignAnimator';
import { useBrowserSpeechRecognition } from '@/hooks/useBrowserSpeechRecognition';
import { DeafAvatar } from '@/components/DeafAvatar';
import { Loader2, Settings, ChevronDown, ChevronUp, ZoomIn, ZoomOut, User, Ear, EarOff, Hand, Video, Mic, CircleDot, StopCircle } from 'lucide-react';
import CameraStream from './CameraStream';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle
} from '@/components/ui/card';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger
} from '@/components/ui/collapsible';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { useWebSocketContext } from '@/contexts/WebSocketContext';
import BrowserView from './BrowserView';
import VoiceActivityDetector from './VoiceActivityDetector';

interface TalkingHeadProps {
  className?: string;
}

const TalkingHead: React.FC<TalkingHeadProps> = ({
  className = ''
}) => {
  const headRef = useRef<any>(null);
  const avatarRef = useRef<HTMLDivElement>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const animatorRef = useRef<SignAnimator | null>(null);
  const audioQueueRef = useRef<any[]>([]);
  const isPlayingAudioRef = useRef(false);

  const [isLoading, setIsLoading] = useState(true);
  const [status, setStatus] = useState<{
    message: string;
    type: 'success' | 'error' | 'info';
  } | null>(null);

  const [selectedAvatar, setSelectedAvatar] = useState('F');
  const [selectedMood, setSelectedMood] = useState('neutral');
  const [scriptsLoaded, setScriptsLoaded] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);

  const [customUrl, setCustomUrl] = useState('');
  const [cameraView, setCameraView] = useState<'head' | 'full'>('full');
  const [debugLog, setDebugLog] = useState<string[]>([]); // New debug log

  // User Modes
  const [userMode, setUserMode] = useState<'normal' | 'mute' | 'deaf'>('normal');

  // Twilio Call State
  const [isCallConnected, setIsCallConnected] = useState(false);



  const addDebug = (msg: string) => setDebugLog(prev => [msg, ...prev].slice(0, 5));

  const avatarOptions = [
    { value: 'F', label: 'Default Female' },
    { value: 'M', label: 'Default Male' },
    { value: 'realistic_f', label: 'Realistic Female (Preset)' },
    { value: 'realistic_m', label: 'Realistic Male (Preset)' },
    { value: 'custom', label: 'Custom URL (Paste Link)' }
  ];

  const moodOptions = [
    { value: 'neutral', label: 'Neutral' },
    { value: 'happy', label: 'Happy' },
    { value: 'sad', label: 'Sad' },
    { value: 'angry', label: 'Angry' },
    { value: 'love', label: 'Love' }
  ];

  // Get WebSocket context
  const {
    isConnected,
    isConnecting, // Restored
    connect,
    disconnect,
    onAudioReceived,
    onInterrupt,
    onError,
    onStatusChange,
    browserImage, // Restored
    sendConfig, // Restored
    onAnimationReceived,
    onCaptionReceived,
    sendSignSequence
  } = useWebSocketContext();

  // Client-Side STT for Deaf Mode (Sign-Kit style)
  const { transcript, startListening, stopListening, resetTranscript } = useBrowserSpeechRecognition();
  const lastSignedLengthRef = useRef(0);

  // Mute Mode: Sign Language Capture
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const [isRecordingSign, setIsRecordingSign] = useState(false);
  const signIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const signFramesRef = useRef<string[]>([]);
  const hiddenVideoRef = useRef<HTMLVideoElement>(null);

  // Sync mode with backend
  useEffect(() => {
    if (isConnected) {
      sendConfig({ mode: userMode });
      addDebug(`Switched to ${userMode} mode`);
    }
  }, [userMode, isConnected, sendConfig]);

  // Model Switcher
  useEffect(() => {
    if (!headRef.current) return;
    loadAvatar(selectedAvatar);
  }, [userMode]);


  // Handle Animation Commands & Captions & Client STT
  /*
  // REMOVED: Using generic onCaptionReceived callback instead to avoid race conditions
  useEffect(() => {
    // Deaf Mode: Sign Captions (Backend AI)
    if (userMode === 'deaf' && lastMessage?.type === 'caption' && lastMessage.text) {
      animatorRef.current?.playText(lastMessage.text);
    }
  }, [lastMessage, userMode]);
  */

  useEffect(() => {
    // Deaf Mode: Sign User Speech (Client STT)
    // DISABLED: User wants Avatar to sign AI Response, not Mirror User
    /*
    if (userMode === 'deaf') {
      startListening();
    } else {
      stopListening();
      resetTranscript();
      lastSignedLengthRef.current = 0;
    }
    */
  }, [userMode]);

  useEffect(() => {
    // Reactively sign new user speech
    if (userMode === 'deaf' && transcript && animatorRef.current) {
      // Get only new part
      const fullLen = transcript.length;
      const prevLen = lastSignedLengthRef.current;
      if (fullLen > prevLen) {
        const newText = transcript.slice(prevLen).trim();
        if (newText) {
          console.log("DeafAvatar: Signing User Input:", newText);
          animatorRef.current.playText(newText);
        }
        lastSignedLengthRef.current = fullLen;
      }
    }
  }, [transcript, userMode]);

  useEffect(() => {
    // Register Animation Callback
    onAnimationReceived((gesture) => {
      const anim = gesture.toLowerCase();

      if (userMode === 'deaf') {
        // Play the gesture as text (e.g. "HOME" -> Sign Home)
        animatorRef.current?.playText(gesture);
      } else {
        // Standard Avatar Logic
        if (anim.includes('wave') || anim.includes('hello')) {
          if (headRef.current) {
            headRef.current.speak({ text: " ", avatarMood: 'happy' });
          }
        } else if (anim.includes('a') || anim.includes('home') || anim.includes('you')) {
          // Should we handle home/you for standard avatar too?
          // Standard avatar lacks bones for this via SignAnimator (unless compatible).
          // We decided fallback might fail or T-pose.
          showStatus(`Received Sign Command: ${gesture} `, 'info');
        }
      }
    });
  }, [onAnimationReceived, userMode]);

  const showStatus = (message: string, type: 'success' | 'error' | 'info') => {
    setStatus({ message, type });
    if (type === 'success' || type === 'info') {
      setTimeout(() => setStatus(null), 3000);
    }
  };



  // Initialize audio context
  const initAudioContext = useCallback(async () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext({ sampleRate: 22050 });
    }
    if (audioContextRef.current.state === 'suspended') {
      await audioContextRef.current.resume();
    }
  }, []);

  // Convert base64 to ArrayBuffer
  const base64ToArrayBuffer = useCallback((base64: string) => {
    const binaryString = atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
  }, []);

  // Convert Int16Array to Float32Array
  const int16ArrayToFloat32 = useCallback((int16Array: Int16Array) => {
    const float32Array = new Float32Array(int16Array.length);
    for (let i = 0; i < int16Array.length; i++) {
      float32Array[i] = int16Array[i] / 32768.0;
    }
    return float32Array;
  }, []);

  // Play next audio in queue
  const playNextAudio = useCallback(async () => {
    if (audioQueueRef.current.length === 0) {
      isPlayingAudioRef.current = false;
      setIsSpeaking(false);
      return;
    }

    isPlayingAudioRef.current = true;
    setIsSpeaking(true);

    const audioItem = audioQueueRef.current.shift();
    console.log('Playing audio item:', audioItem);

    try {
      if (
        headRef.current &&
        audioItem.timingData &&
        audioItem.timingData.words
      ) {
        // Use TalkingHead with native timing
        const speakData = {
          audio: audioItem.buffer,
          words: audioItem.timingData.words,
          wtimes: audioItem.timingData.word_times,
          wdurations: audioItem.timingData.word_durations
        };

        console.log('Using TalkingHead with timing data:', speakData);
        headRef.current.speakAudio(speakData);

        // Set timer for next audio
        setTimeout(() => {
          console.log('TalkingHead audio finished, playing next...');
          playNextAudio();
        }, audioItem.duration * 1000);
      } else if (headRef.current) {
        // Basic TalkingHead audio without timing
        console.log('Using basic TalkingHead audio');
        headRef.current.speakAudio({ audio: audioItem.buffer });

        setTimeout(() => {
          console.log('Basic TalkingHead audio finished, playing next...');
          playNextAudio();
        }, audioItem.duration * 1000);
      } else {
        // Fallback to Web Audio API
        console.log('Using Web Audio API fallback');
        await initAudioContext();
        const source = audioContextRef.current!.createBufferSource();
        source.buffer = audioItem.buffer;
        source.connect(audioContextRef.current!.destination);
        source.onended = () => {
          console.log('Web Audio finished, playing next...');
          playNextAudio();
        };
        source.start();
      }
    } catch (error) {
      console.error('Error playing audio:', error);
      // Continue to next audio on error
      setTimeout(() => playNextAudio(), 100);
    }
  }, [initAudioContext]);

  const handleCaptionReceived = useCallback((text: string) => {
    if (userMode === 'deaf' && animatorRef.current) {
      console.log("Triggering Animator for Caption:", text);
      showStatus(`Signing: "${text.substring(0, 20)}..."`, 'info');
      animatorRef.current.playText(text);
    }
  }, [userMode]);

  // Handle audio from WebSocket
  const handleAudioReceived = useCallback(
    async (
      base64Audio: string,
      timingData?: any,
      sampleRate = 24000,
      method = 'unknown'
    ) => {
      console.log('🎵 TALKINGHEAD handleAudioReceived CALLED!', {
        audioLength: base64Audio.length,
        timingData,
        sampleRate,
        method
      });

      try {
        await initAudioContext();

        // Convert base64 to audio buffer
        const arrayBuffer = base64ToArrayBuffer(base64Audio);
        const int16Array = new Int16Array(arrayBuffer);
        const float32Array = int16ArrayToFloat32(int16Array);

        console.log('Audio conversion successful:', {
          arrayBufferLength: arrayBuffer.byteLength,
          int16Length: int16Array.length,
          float32Length: float32Array.length
        });

        // Create AudioBuffer
        const audioBuffer = audioContextRef.current!.createBuffer(
          1,
          float32Array.length,
          sampleRate
        );
        audioBuffer.copyToChannel(float32Array, 0);

        console.log('AudioBuffer created:', {
          duration: audioBuffer.duration,
          sampleRate: audioBuffer.sampleRate,
          length: audioBuffer.length
        });

        // Add to queue
        audioQueueRef.current.push({
          buffer: audioBuffer,
          timingData: timingData,
          duration: audioBuffer.duration,
          method: method
        });

        console.log(
          'Audio added to queue. Queue length:',
          audioQueueRef.current.length
        );

        // Start playing if not already playing
        if (!isPlayingAudioRef.current) {
          console.log('Starting audio playback...');
          playNextAudio();
        } else {
          console.log('Audio already playing, added to queue');
        }

        const timingInfo = timingData
          ? ` with ${timingData.words?.length || 0} word timings`
          : ' (no timing)';
        console.log(
          `✅ Audio queued successfully: ${audioBuffer.duration.toFixed(2)}s${timingInfo} [${method}]`
        );
      } catch (error) {
        console.error(
          '❌ Error processing audio in handleAudioReceived:',
          error
        );
      }
    },
    [initAudioContext, base64ToArrayBuffer, int16ArrayToFloat32, playNextAudio]
  );

  // Handle interrupt from server
  const handleInterrupt = useCallback(() => {
    // Clear audio queue
    audioQueueRef.current = [];
    isPlayingAudioRef.current = false;
    setIsSpeaking(false);

    // Stop TalkingHead if speaking
    // if (headRef.current) {
    //   try {
    //     headRef.current.stop();
    //   } catch (error) {
    //     console.error('Error stopping TalkingHead:', error);
    //   }
    // }

    console.log('Audio interrupted and cleared');
  }, []);

  // Register WebSocket callbacks
  useEffect(() => {
    onAudioReceived(handleAudioReceived);
    onInterrupt(handleInterrupt);
    onError((error) => showStatus(`WebSocket error: ${error} `, 'error'));
    onStatusChange((status) => {
      if (status === 'connected')
        showStatus('Connected to voice assistant', 'success');
      else if (status === 'disconnected')
        showStatus('Disconnected from server', 'info');
      else
        showStatus(status, 'info');
    });

    // Listen to Twilio Voice Manager
    import('@/utils/TwilioVoiceManager').then(mod => {
      const manager = mod.default.getInstance();

      manager.on('connected', () => {
        console.log("UI: Call Connected");
        setIsCallConnected(true);
        showStatus("Phone Call Connected", "success");
      });

      manager.on('disconnected', () => {
        console.log("UI: Call Disconnected");
        setIsCallConnected(false);
        showStatus("Phone Call Ended", "info");
      });

      manager.on('error', (err: any) => {
        showStatus(`Call Error: ${err.message || err} `, "error");
      });
    });

    // Register Caption Listener
    onCaptionReceived((text) => {
      console.log("TalkingHead: Caption Received:", text);
      // We check userModeRef if needed, or rely on re-registration?
      // Let's use logic inside: if userMode is deaf.
      // But userMode is state.
      // We need to ensure this callback is fresh.
      handleCaptionReceived(text);
    });

  }, [
    onAudioReceived,
    onInterrupt,
    onError,
    onStatusChange,
    onCaptionReceived, // NEW
    handleAudioReceived,
    handleInterrupt,
    handleCaptionReceived // NEW
  ]);

  // Listen for TalkingHead library to load
  useEffect(() => {
    const handleTalkingHeadLoaded = () => {
      setScriptsLoaded(true);
    };

    const handleTalkingHeadError = () => {
      showStatus('Failed to load TalkingHead library', 'error');
    };

    if ((window as any).TalkingHead) {
      setScriptsLoaded(true);
      return;
    }

    window.addEventListener('talkinghead-loaded', handleTalkingHeadLoaded);
    window.addEventListener('talkinghead-error', handleTalkingHeadError);

    return () => {
      window.removeEventListener('talkinghead-loaded', handleTalkingHeadLoaded);
      window.removeEventListener('talkinghead-error', handleTalkingHeadError);
    };
  }, []);

  // Initialize TalkingHead
  useEffect(() => {
    if (!scriptsLoaded || !avatarRef.current) return;

    const initTalkingHead = async () => {
      try {
        setIsLoading(true);
        showStatus('Initializing avatar...', 'info');

        const TalkingHead = (window as any).TalkingHead;
        if (!TalkingHead) {
          throw new Error('TalkingHead library not loaded');
        }

        headRef.current = new TalkingHead(avatarRef.current, {
          ttsEndpoint: 'https://texttospeech.googleapis.com/v1/text:synthesize',
          jwtGet: () => Promise.resolve('dummy-jwt-token'),
          lipsyncModules: ['en'],
          lipsyncLang: 'en',
          modelFPS: 30,
          cameraView: cameraView, // Use state
          avatarMute: false,
          avatarMood: selectedMood
        });

        await loadAvatar(selectedAvatar);
        setIsLoading(false);
        showStatus('Avatar ready!', 'success');

        // Auto-connect removed - handled by Login page
        // connect();
      } catch (error: any) {
        setIsLoading(false);
        showStatus(`Failed to initialize: ${error.message} `, 'error');
      }
    };

    initTalkingHead();

    return () => {
      if (headRef.current) {
        try {
          headRef.current.stop();
        } catch (error) {
          console.error('Cleanup error:', error);
        }
      }
    };
  }, [scriptsLoaded, connect]);

  // Dynamic View Update
  useEffect(() => {
    if (headRef.current && headRef.current.setView) {
      try {
        console.log(`Setting camera view to: ${cameraView} `);
        headRef.current.setView(cameraView);
      } catch (error) {
        console.warn('Failed to set camera view:', error);
      }
    }
  }, [cameraView]);

  // Gesture/Mood Animation Loop
  useEffect(() => {
    if (!isSpeaking || !headRef.current) return;

    const interval = setInterval(() => {
      try {
        // Randomly shift mood/gesture to create "liveliness"
        const gestures = ['neutral', 'happy', 'neutral', 'happy'];
        const randomGesture = gestures[Math.floor(Math.random() * gestures.length)];

        // This is a safe way to trigger movement without knowing exact animation names
        // Changing mood triggers facial/head movement
        headRef.current.setMood(randomGesture);

        // If the library supports specific hand gestures, we'd call playGesture here
        // e.g. headRef.current.playAnimation('talking');
      } catch (e) {
        console.warn("Gesture error:", e);
      }
    }, 2000); // Change every 2 seconds while speaking

    return () => clearInterval(interval);
  }, [isSpeaking]);

  const loadAvatar = async (avatarValue: string = 'F') => {
    // 4K Textures & High Quality Settings
    // NOTE: Removed useDracoMeshCompression as it might require external WASM
    const qualityParams = '&textureSizeLimit=1024&textureFormat=png';
    const morphTargets = '?morphTargets=ARKit,Oculus+Visemes,mouthOpen,mouthSmile,eyesClosed,eyesLookUp,eyesLookDown';

    const avatarUrls: Record<string, string> = {
      F: `https://models.readyplayer.me/64bfa15f0e72c63d7c3934a6.glb${morphTargets}${qualityParams}`,
      // Use known valid Sample ID for Male
      M: `https://models.readyplayer.me/6185a4acfb622cf1cdc49348.glb${morphTargets}${qualityParams}`,
      // Map realistic to defaults to prevent 404s until user provides custom
      realistic_f: `https://models.readyplayer.me/6942c3c8fba15e26e148c0ba.glb${morphTargets}${qualityParams}`,
      realistic_m: `https://models.readyplayer.me/6185a4acfb622cf1cdc49348.glb${morphTargets}${qualityParams}`
    };

    // URL resolution moved to retry logic block below

    let url = avatarUrls[avatarValue];
    let isRetry = false;

    // Custom URL handling
    if (avatarValue === 'custom') {
      // ... existing custom logic
      url = customUrl.trim();
      if (!url) return;
    }

    // Helper to try loading
    const tryLoad = async (loadUrl: string, attemptName: string) => {
      try {
        addDebug(`Attempting (${attemptName}): ${loadUrl}`);

        try {
          await headRef.current?.showAvatar({
            url: loadUrl,
            avatarMood: selectedMood,
            lipsyncLang: 'en'
          });
        } catch (error: any) {
          console.warn("Avatar Load Error:", error);
          throw error;
        }

        // Initialize SignAnimator with the loaded avatar mesh
        if (headRef.current) {
          console.log("[DEBUG] TalkingHead keys:", Object.keys(headRef.current));
          const mesh = headRef.current.scene || headRef.current.armature || headRef.current.avatar?.scene || headRef.current.obj;

          if (mesh) {
            console.log("[DEBUG] Found avatar mesh:", mesh);
            animatorRef.current = new SignAnimator(mesh);
            showStatus('Sign Language Engine Ready', 'success');
          } else {
            console.warn("[DEBUG] Could not find avatar mesh in TalkingHead instance");
          }
        }

        showStatus(`Avatar loaded (${attemptName})`, 'success');
        addDebug(`Success: ${attemptName}`);
        return true;

      } catch (e: any) {
        console.error(`Load failed (${attemptName}):`, e);
        addDebug(`Error (${attemptName}): ${e.message}`);
        return false;
      }
    };

    // 1. Try High Quality / User URL
    if (await tryLoad(url, 'High Quality')) return;

    // 2. Fallback: Clean URL (No params)
    const cleanUrl = url.split('?')[0];
    if (cleanUrl !== url) {
      showStatus('Retrying with basic quality...', 'info');
      if (await tryLoad(cleanUrl, 'Basic Quality')) return;
    }

    showStatus('Failed to load avatar. See Debug Log.', 'error');
  };

  const toggleCameraView = () => {
    const newView = cameraView === 'full' ? 'head' : 'full';
    setCameraView(newView);
    // The useEffect above will handle the actual update
  };

  const handleCustomUrlSubmit = () => {
    if (selectedAvatar === 'custom') {
      loadAvatar('custom');
    }
  };

  const handleAvatarChange = (gender: string) => {
    setSelectedAvatar(gender);
    if (scriptsLoaded && headRef.current) {
      loadAvatar(gender);
    }
  };

  const handleMoodChange = (mood: string) => {
    setSelectedMood(mood);
    if (headRef.current) {
      headRef.current.setMood(mood);
    }
  };

  const toggleRecording = useCallback(() => {
    if (isRecordingSign) {
      // STOP Recording
      setIsRecordingSign(false);
      if (signIntervalRef.current) {
        clearInterval(signIntervalRef.current);
        signIntervalRef.current = null;
      }

      // Send Sequence
      if (signFramesRef.current.length > 0) {
        console.log(`Sending Sign Sequence: ${signFramesRef.current.length} frames`);
        sendSignSequence(signFramesRef.current);
        // Backend handles response
      }
    } else {
      // START Recording
      if (!cameraStream) {
        alert("No camera stream detected. Please enable camera.");
        return;
      }

      // Setup hidden video to play stream for capturing
      if (hiddenVideoRef.current && hiddenVideoRef.current.srcObject !== cameraStream) {
        hiddenVideoRef.current.srcObject = cameraStream;
        hiddenVideoRef.current.play();
      }

      signFramesRef.current = [];
      setIsRecordingSign(true);

      // Capture Loop
      signIntervalRef.current = setInterval(() => {
        if (hiddenVideoRef.current) {
          const canvas = document.createElement('canvas');
          canvas.width = 320; // Lower res for speed
          canvas.height = 240;
          const ctx = canvas.getContext('2d');
          if (ctx) {
            ctx.drawImage(hiddenVideoRef.current, 0, 0, canvas.width, canvas.height);
            // Get base64 (remove prefix)
            const dataUrl = canvas.toDataURL('image/jpeg', 0.6);
            const base64 = dataUrl.split(',')[1];
            signFramesRef.current.push(base64);
          }
        }
      }, 200); // 5 FPS
    }
  }, [isRecordingSign, cameraStream, sendSignSequence]);

  return (
    <Card className={`w-full ${className}`}>
      <CardHeader className="text-center">
        <CardTitle className="text-2xl font-bold">AI Avatar</CardTitle>
        <CardDescription>Voice-controlled 3D avatar</CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">


        {/* Mute Mode UI: PIP Camera & Control Bar */}
        {userMode === 'mute' && (
          <>
            {/* PIP Camera (Self View) */}
            <CameraStream
              onStreamChange={setCameraStream}
              onClose={() => setUserMode('normal')}
            />

            {/* Frozen / Recording Indicator Border on Camera? CameraStream handles "Live" indicator. */}

            {/* Bottom Control Bar */}
            <div className="fixed bottom-6 left-1/2 transform -translate-x-1/2 z-50 flex flex-col items-center gap-2 animate-in slide-in-from-bottom-10 fade-in duration-300">
              {/* Recording Feedback Bubble */}
              {isRecordingSign && (
                <div className="bg-red-600 text-white px-4 py-1 rounded-full text-xs font-bold animate-pulse shadow-lg mb-2">
                  Recording Signs...
                </div>
              )}

              <div className="bg-white/90 dark:bg-slate-900/90 backdrop-blur-md border border-slate-200 dark:border-slate-700 p-2 rounded-2xl shadow-2xl flex items-center gap-4">

                {/* Exit Button */}
                <Button
                  variant="ghost"
                  size="icon"
                  className="rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-500"
                  onClick={() => setUserMode('normal')}
                  title="Exit Mute Mode"
                >
                  <StopCircle size={20} />
                </Button>

                {/* Record Button */}
                <Button
                  onClick={toggleRecording}
                  size="lg"
                  className={`rounded-full w-16 h-16 shadow-xl border-4 transition-all duration-300 ${isRecordingSign
                    ? 'bg-white border-red-500 hover:bg-red-50 text-red-600 scale-105'
                    : 'bg-red-600 border-red-100 hover:bg-red-700 hover:scale-105 hover:shadow-red-500/25 ring-2 ring-red-100 ring-offset-2'
                    }`}
                >
                  {isRecordingSign
                    ? <div className="w-6 h-6 bg-red-600 rounded-sm" />
                    : <div className="w-6 h-6 bg-white rounded-full" />
                  }
                </Button>

                {/* Info / Help */}
                <div className="w-10 flex justify-center">
                  <div className="text-[10px] sm:text-xs font-medium text-slate-500 dark:text-slate-400 text-center leading-tight">
                    {isRecordingSign ? "Tap to\nSend" : "Hold/Tap\nto Sign"}
                  </div>
                </div>

              </div>
            </div>
          </>
        )}

        {/* Hidden Video for Capture */}
        <video ref={hiddenVideoRef} className="hidden" muted playsInline autoPlay />

        {/* Main Display Area - Grid Layout if browsing, else Single Column */}
        <div className={`grid gap-4 h-[500px] transition-all duration-300 ${browserImage ? 'grid-cols-1 lg:grid-cols-2' : 'grid-cols-1'}`}>

          {/* LEFT: Avatar Display (Full width if not browsing) */}
          <div className="relative overflow-hidden rounded-lg bg-gradient-to-br from-gray-100 to-gray-200 h-full border-2 border-slate-200 dark:border-slate-800 shadow-sm">
            <div className="absolute inset-0 flex items-center justify-center">
              <div ref={avatarRef} className={`h-full w-full ${userMode === 'deaf' ? 'invisible' : ''}`} />
            </div>

            {/* Deaf Avatar Overlay */}
            {userMode === 'deaf' && (
              <div className="absolute inset-0 z-20">
                <DeafAvatar onAnimatorReady={(anim) => {
                  console.log("Switched to YBot Animator");
                  animatorRef.current = anim;
                }} />
              </div>
            )}

            {/* Loading Overlay */}
            {(isLoading || !scriptsLoaded) && (
              <div className="bg-opacity-90 absolute inset-0 flex items-center justify-center bg-white z-20">
                <div className="text-center">
                  <Loader2 className="text-primary mx-auto mb-4 h-12 w-12 animate-spin" />
                  <p className="text-muted-foreground">
                    {!scriptsLoaded
                      ? 'Loading TalkingHead...'
                      : 'Loading avatar...'}
                  </p>
                </div>
              </div>
            )}

            {/* Status Badges */}
            {scriptsLoaded && !isLoading && (
              <div className="absolute top-4 left-4 space-y-2 z-10">
                <Badge variant={isConnected ? 'default' : 'secondary'}>
                  {isConnecting
                    ? 'Connecting...'
                    : isConnected
                      ? 'Connected'
                      : 'Disconnected'}
                </Badge>
                {isSpeaking && (
                  <Badge variant="destructive" className="block">
                    Speaking...
                  </Badge>
                )}
                {isRecordingSign && (
                  <Badge className="block bg-red-600 animate-pulse">
                    🎥 Recording...
                  </Badge>
                )}
              </div>
            )}

            {/* Controls Overlay */}
            <div className="absolute bottom-4 left-0 right-0 px-4 flex justify-between z-10 w-full">
              <div className="flex gap-2">
                <Button variant="ghost" size="icon" onClick={() => setIsSettingsOpen(!isSettingsOpen)}>
                  <Settings className="w-5 h-5" />
                </Button>
              </div>
            </div>
          </div>

          {/* RIGHT: Browser View (Only visible if browserImage exists) */}
          {browserImage && (
            <div className="h-full rounded-lg overflow-hidden border-2 border-slate-200 dark:border-slate-800 shadow-sm bg-slate-50 dark:bg-slate-900 transition-all duration-300 animate-in fade-in slide-in-from-right-10">
              <BrowserView image={browserImage} />
            </div>
          )}
        </div>

        {/* VAD Visualization */}
        <div className="mt-4">
          <VoiceActivityDetector
            cameraStream={cameraStream}
            autoStart={userMode !== 'mute'}
          />
        </div>

        {/* Connection Control */}
        <div className="flex justify-center space-x-4">
          <Button
            variant={isConnected ? 'destructive' : 'default'}
            onClick={() => (isConnected ? disconnect() : connect())}
            className="w-40 transition-all duration-300 transform hover:scale-105"
          >
            {isConnecting
              ? 'Connecting...'
              : isConnected
                ? 'Disconnect'
                : 'Connect'}
          </Button>

          {/* Zoom Toggle */}
          <Button
            onClick={toggleCameraView}
            disabled={!scriptsLoaded}
            variant="secondary"
            className="w-12 px-0"
            title={cameraView === 'full' ? "Zoom In (Face)" : "Zoom Out (Body)"}
          >
            {cameraView === 'full' ? <ZoomIn size={20} /> : <ZoomOut size={20} />}
          </Button>
        </div>

        {/* Mode Toggles for Mute/Deaf Accessibility */}
        <div className="flex gap-2 justify-center py-2 px-1">
          <Button
            variant={userMode === 'mute' ? "default" : "outline"}
            onClick={() => setUserMode(m => m === 'mute' ? 'normal' : 'mute')}
            className={`flex-1 h-8 text-xs ${userMode === 'mute' ? 'bg-blue-600 hover:bg-blue-700' : ''}`}
            title="Mute Mode: Use Sign Language Input"
          >
            <Hand className="mr-1 h-3 w-3" />
            Mute Users
          </Button>

          <Button
            variant={userMode === 'deaf' ? "default" : "outline"}
            onClick={() => setUserMode(m => m === 'deaf' ? 'normal' : 'deaf')}
            className={`flex-1 h-8 text-xs ${userMode === 'deaf' ? 'bg-purple-600 hover:bg-purple-700' : ''}`}
            title="Deaf Mode: Avatar signs back"
          >
            {userMode === 'deaf' ? <EarOff className="mr-1 h-3 w-3" /> : <Ear className="mr-1 h-3 w-3" />}
            Deaf Users
          </Button>
        </div>

        {userMode === 'deaf' && (
          <div className="text-center text-[10px] text-purple-600 font-bold animate-pulse">
            Avatar will respond with Sign Language videos
          </div>
        )}

        {/* Twilio Call Controls */}
        {isCallConnected && (
          <div className="flex gap-3 justify-center">
            <Button
              variant="destructive"
              className="w-full bg-red-600 hover:bg-red-700 animate-pulse"
              onClick={() => {
                import('@/utils/TwilioVoiceManager').then(m => m.default.getInstance().disconnect());
              }}
            >
              <span className="mr-2">📞</span> End Call
            </Button>
          </div>
        )}

        {/* Settings */}
        <Collapsible open={isSettingsOpen} onOpenChange={setIsSettingsOpen}>
          <CollapsibleTrigger asChild>
            <Button variant="outline" className="w-full">
              <Settings className="mr-2 h-4 w-4" />
              Avatar Settings
              {isSettingsOpen ? (
                <ChevronUp className="ml-2 h-4 w-4" />
              ) : (
                <ChevronDown className="ml-2 h-4 w-4" />
              )}
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent className="mt-4 space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2 col-span-2 md:col-span-1">
                <Label>Avatar Model</Label>
                <Select
                  value={selectedAvatar}
                  onValueChange={handleAvatarChange}
                  disabled={!scriptsLoaded}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {avatarOptions.map((avatar) => (
                      <SelectItem key={avatar.value} value={avatar.value}>
                        {avatar.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>

                {/* Custom URL Input */}
                {selectedAvatar === 'custom' && (
                  <div className="flex gap-2 mt-2">
                    <Input
                      placeholder="Paste ReadyPlayerMe GLB Link..."
                      value={customUrl}
                      onChange={(e) => setCustomUrl(e.target.value)}
                      className="text-xs"
                    />
                    <Button size="sm" onClick={handleCustomUrlSubmit}>Load</Button>
                  </div>
                )}
              </div>

              <div className="space-y-2">
                <Label>Mood</Label>
                <Select
                  value={selectedMood}
                  onValueChange={handleMoodChange}
                  disabled={!scriptsLoaded}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {moodOptions.map((mood) => (
                      <SelectItem key={mood.value} value={mood.value}>
                        {mood.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
          </CollapsibleContent>
        </Collapsible>

        {/* Status Display */}
        {status && (
          <Alert variant={status.type === 'error' ? 'destructive' : 'default'}>
            <AlertDescription>{status.message}</AlertDescription>
          </Alert>
        )}
      </CardContent>
    </Card>
  );
};

export default TalkingHead;
