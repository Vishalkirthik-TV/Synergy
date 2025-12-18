'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Loader2, Settings, ChevronDown, ChevronUp, ZoomIn, ZoomOut, User } from 'lucide-react';
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

interface TalkingHeadProps {
  className?: string;
  cameraStream?: MediaStream | null;
}

const TalkingHead: React.FC<TalkingHeadProps> = ({
  className = '',
  cameraStream
}) => {
  const avatarRef = useRef<HTMLDivElement>(null);
  const headRef = useRef<any>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioQueueRef = useRef<any[]>([]);
  const isPlayingAudioRef = useRef(false);
  const lastInterruptRef = useRef<number>(0); // Track last interrupt time for safety buffer

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
    isConnecting,
    connect,
    disconnect,
    onAudioReceived,
    onInterrupt,
    onError,
    onStatusChange
  } = useWebSocketContext();

  const showStatus = (message: string, type: 'success' | 'error' | 'info') => {
    setStatus({ message, type });
    if (type === 'success' || type === 'info') {
      setTimeout(() => setStatus(null), 3000);
    }
  };

  // Initialize audio context
  const initAudioContext = useCallback(async () => {
    if (!audioContextRef.current) {
      // Allow browser to choose default sample rate (usually 44.1k or 48k)
      // This prevents sync issues with 24k audio
      audioContextRef.current = new AudioContext();
    }
    if (audioContextRef.current.state === 'suspended') {
      await audioContextRef.current.resume();
    }
  }, []);

  // Helper to convert base64 to ArrayBuffer
  function base64ToArrayBuffer(base64: string) {
    const binaryString = window.atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
  }

  // Helper to play silence and wake up audio engine
  const playSilence = useCallback(async () => {
    if (!headRef.current || !audioContextRef.current) return;

    try {
      // Create a tiny silence buffer (0.1s)
      const ctx = audioContextRef.current;
      const buffer = ctx.createBuffer(1, ctx.sampleRate / 10, ctx.sampleRate);

      // Speak silence - this forces the internal audio context to resume/stay active
      headRef.current.speakAudio({ audio: buffer });
      console.log(`📢 Waking up audio engine with silence (${ctx.sampleRate}Hz)`);
    } catch (e) {
      console.error('Failed to play silence:', e);
    }
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

    // SAFETY BUFFER: If an interrupt happened recently (< 500ms), wait for it to clear
    // This prevents the "Stop" command from killing the new "Speak" command
    const timeSinceInterrupt = Date.now() - lastInterruptRef.current;
    if (timeSinceInterrupt < 500) {
      const waitTime = 500 - timeSinceInterrupt;
      console.log(`⏳ Safety Buffer: Waiting ${waitTime}ms for avatar to reset...`);
      setTimeout(() => playNextAudio(), waitTime);
      return;
    }

    // Ensure AudioContext is initialized and running
    if (!audioContextRef.current) {
      // Use interactive latency for snappier audio/interrupts
      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      audioContextRef.current = new AudioContextClass({ latencyHint: 'interactive', sampleRate: 48000 });
    }

    if (audioContextRef.current?.state === 'suspended') {
      console.log('AudioContext suspended, attempting to resume...');
      try {
        await audioContextRef.current.resume();
        console.log('AudioContext resumed successfully');

        // Also fire silence to wake up library
        playSilence();
      } catch (e) {
        console.error('Failed to resume AudioContext:', e);
      }
    }

    const audioItem = audioQueueRef.current.shift();
    console.log('Playing audio item:', audioItem, 'AudioContext State:', audioContextRef.current?.state);

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

        // Check if we actually have valid timing data (sometimes it's empty)
        if (speakData.words.length > 0) {
          console.log('Using TalkingHead with timing data:', speakData);
          headRef.current.speakAudio(speakData);
        } else {
          console.warn('Timing data empty. Playing audio directly to speakers (Bypassing Avatar).');
          // Manual Playback to ensure sound is heard
          try {
            const source = audioContextRef.current!.createBufferSource();
            source.buffer = audioItem.buffer;
            source.connect(audioContextRef.current!.destination);
            source.start(audioContextRef.current!.currentTime);
          } catch (e) {
            console.error('Manual fallback playback failed:', e);
          }
        }

        // Set timer for next audio
        setTimeout(() => {
          console.log('TalkingHead audio finished, playing next...');
          playNextAudio();
        }, audioItem.duration * 1000 + 50); // Add small buffer
      } else if (headRef.current) {
        // Basic TalkingHead audio without timing
        console.log('Using basic TalkingHead audio');
        headRef.current.speakAudio({ audio: audioItem.buffer });

        setTimeout(() => {
          console.log('Basic TalkingHead audio finished, playing next...');
          playNextAudio();
        }, audioItem.duration * 1000 + 50);
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
  }, [initAudioContext, playSilence]);

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

        // Convert base64 to ArrayBuffer (WAV file bytes)
        const arrayBuffer = base64ToArrayBuffer(base64Audio);

        // Decode WAV using browser's native decoder (Robust & Safe)
        const audioBuffer = await audioContextRef.current!.decodeAudioData(arrayBuffer);

        console.log('✅ WAV Audio Decoded:', {
          duration: audioBuffer.duration,
          sampleRate: audioBuffer.sampleRate,
          channels: audioBuffer.numberOfChannels,
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
          console.log('Starting audio playback (Queue not empty, not playing)...');
          playNextAudio();
        } else {
          console.log('Audio already playing, added to queue. Current Queue Size:', audioQueueRef.current.length);
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
    [initAudioContext, playNextAudio]
  );

  // Handle interrupt from server
  const handleInterrupt = useCallback(() => {
    // Clear audio queue
    audioQueueRef.current = [];
    isPlayingAudioRef.current = false;
    setIsSpeaking(false);
    lastInterruptRef.current = Date.now(); // Mark interrupt time

    // Stop TalkingHead if speaking
    if (headRef.current) {
      try {
        console.log('Stopping TalkingHead animation/audio...');
        // 🛑 Stop the avatar immediately
        headRef.current.stop();

        // Note: We previously used audioContext.suspend() here, but it caused 
        // "Resumed Ghost Audio" (Double Audio) issues when resuming for the next sentence.
        // We now rely on headRef.current.stop() and low-latency AudioContext.

        // "Wake up" the audio engine with silence immediately after stop
        setTimeout(() => {
          playSilence();
        }, 50);
      } catch (error) {
        console.error('Error stopping TalkingHead:', error);
      }
    }

    console.log('Audio interrupted, queue cleared, state reset.');
  }, [playSilence]);

  // Register WebSocket callbacks
  useEffect(() => {
    onAudioReceived(handleAudioReceived);
    onInterrupt(handleInterrupt);
    onError((error) => showStatus(`WebSocket error: ${error}`, 'error'));
    onStatusChange((status) => {
      if (status === 'connected')
        showStatus('Connected to voice assistant', 'success');
      if (status === 'disconnected')
        showStatus('Disconnected from server', 'info');
    });
  }, [
    onAudioReceived,
    onInterrupt,
    onError,
    onStatusChange,
    handleAudioReceived,
    handleInterrupt
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
        showStatus(`Failed to initialize: ${error.message}`, 'error');
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
        console.log(`Setting camera view to: ${cameraView}`);
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
        await headRef.current?.showAvatar({
          url: loadUrl,
          avatarMood: selectedMood,
          lipsyncLang: 'en'
        });
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

  return (
    <Card className={`w-full ${className}`}>
      <CardHeader className="text-center">
        <CardTitle className="text-2xl font-bold">AI Avatar</CardTitle>
        <CardDescription>Voice-controlled 3D avatar</CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Avatar Display */}
        <div
          className="relative overflow-hidden rounded-lg bg-gradient-to-br from-gray-100 to-gray-200"
          style={{ height: '500px' }}
        >
          <div ref={avatarRef} className="h-full w-full" />

          {/* Loading Overlay */}
          {(isLoading || !scriptsLoaded) && (
            <div className="bg-opacity-90 absolute inset-0 flex items-center justify-center bg-white">
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
            <div className="absolute top-4 left-4 space-y-2">
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
            </div>
          )}
        </div>

        {/* Connection Control */}
        <div className="flex gap-3">
          <Button
            onClick={isConnected ? disconnect : () => connect()}
            disabled={isConnecting || !scriptsLoaded}
            className="flex-1"
            variant={isConnected ? 'destructive' : 'default'}
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

        {/* Debug Log (Visible only if there are logs) */}
        {debugLog.length > 0 && (
          <div className="mt-4 p-2 bg-black/5 rounded text-[10px] font-mono text-gray-600 overflow-hidden">
            <p className="font-bold mb-1">Debug Log:</p>
            {debugLog.map((log, i) => (
              <div key={i} className="truncate">{log}</div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default TalkingHead;
