'use client';

import React, {
  createContext,
  useContext,
  useRef,
  useCallback,
  useState,
  ReactNode
} from 'react';

interface WordTiming {
  word: string;
  start_time: number;
  end_time: number;
}

interface WebSocketMessage {
  status?: string;
  client_id?: string;
  interrupt?: boolean;
  audio?: string;
  word_timings?: WordTiming[];
  sample_rate?: number;
  method?: string;
  audio_complete?: boolean;
  error?: string;
  type?: string;
  message?: string;
  image?: string;
  gesture?: string;
  text?: string;
}

interface WebSocketContextType {
  isConnected: boolean;
  isConnecting: boolean;
  connect: (userId?: string) => Promise<void>;
  disconnect: () => void;
  sendAudioSegment: (audioData: ArrayBuffer) => void;
  sendImage: (imageData: string) => void;
  sendAudioWithImage: (audioData: ArrayBuffer, imageData: string) => void;
  sendSignSequence: (frames: string[]) => void;
  sendConfig: (config: any) => void;
  onAudioReceived: (
    callback: (
      audioData: string,
      timingData?: any,
      sampleRate?: number,
      method?: string
    ) => void
  ) => void;
  onInterrupt: (callback: () => void) => void;
  onError: (callback: (error: string) => void) => void;
  onStatusChange: (
    callback: (status: 'connected' | 'disconnected' | 'connecting') => void
  ) => void;
  onAnimationReceived: (callback: (gesture: string) => void) => void;
  onCaptionReceived: (callback: (text: string) => void) => void;
  browserImage: string | null;
  lastMessage: WebSocketMessage | null;
}

const WebSocketContext = createContext<WebSocketContextType | null>(null);

export const useWebSocketContext = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error(
      'useWebSocketContext must be used within a WebSocketProvider'
    );
  }
  return context;
};

interface WebSocketProviderProps {
  children: ReactNode;
  serverUrl?: string;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({
  children,
  serverUrl = 'ws://localhost:8000/ws/'
}) => {
  const wsRef = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [browserImage, setBrowserImage] = useState<string | null>(null);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);

  // Callback refs
  const audioReceivedCallbackRef = useRef<
    | ((
      audioData: string,
      timingData?: any,
      sampleRate?: number,
      method?: string
    ) => void)
    | null
  >(null);
  const interruptCallbackRef = useRef<(() => void) | null>(null);
  const errorCallbackRef = useRef<((error: string) => void) | null>(null);
  const statusChangeCallbackRef = useRef<
    ((status: 'connected' | 'disconnected' | 'connecting') => void) | null
  >(null);
  const animationReceivedCallbackRef = useRef<((gesture: string) => void) | null>(null);
  const captionReceivedCallbackRef = useRef<((text: string) => void) | null>(null);

  const connect = useCallback(async (userId: string = 'test-client') => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      setIsConnecting(true);
      statusChangeCallbackRef.current?.('connecting');

      // Append userId to the base URL (ensure no trailing slash on userId)
      const cleanUserId = userId?.replace(/\/$/, '') || 'user';
      wsRef.current = new WebSocket(`${serverUrl}${cleanUserId}`);

      wsRef.current.onopen = () => {
        setIsConnected(true);
        setIsConnecting(false);
        statusChangeCallbackRef.current?.('connected');
        console.log('WebSocket connected');
      };

      wsRef.current.onmessage = (event) => {
        try {
          const data: WebSocketMessage = JSON.parse(event.data);
          console.log('WebSocket message received:', data);
          setLastMessage(data); // Set the last received message

          if (data.status === 'connected') {
            console.log(
              `Server confirmed connection. Client ID: ${data.client_id}`
            );
          } else if (data.interrupt) {
            console.log('Received interrupt signal');
            interruptCallbackRef.current?.();
          } else if (data.audio) {
            // Handle audio with native timing
            let timingData = null;

            if (data.word_timings) {
              // Convert to TalkingHead format
              timingData = {
                words: data.word_timings.map((wt) => wt.word),
                word_times: data.word_timings.map((wt) => wt.start_time),
                word_durations: data.word_timings.map(
                  (wt) => wt.end_time - wt.start_time
                )
              };
              console.log('Converted timing data:', timingData);
            }

            console.log('Calling audioReceivedCallback with:', {
              audioLength: data.audio.length,
              timingData,
              sampleRate: data.sample_rate || 24000,
              method: data.method || 'unknown'
            });

            audioReceivedCallbackRef.current?.(
              data.audio,
              timingData,
              data.sample_rate || 24000,
              data.method || 'unknown'
            );
          } else if (data.audio_complete) {
            console.log('Audio processing complete');
          } else if (data.type === 'status') {
            console.log('Status update:', data.message);
            // Trigger Twilio Call if message indicates calling
            // "Calling +123..."
            if (data.message && data.message.startsWith("Calling")) {
              const match = data.message.match(/Calling\s+([+\d\s-]+)/);
              if (match) {
                const number = match[1];
                console.log("📞 Triggering Browser Call to:", number);
                import('../utils/TwilioVoiceManager').then(({ default: TwilioVoiceManager }) => {
                  TwilioVoiceManager.getInstance().makeCall(number);
                });
              }
            }
          } else if (data.type === 'browser_update') {
            console.log('🌐 Browser Update Received');
            setBrowserImage(data.image || null);
          } else if (data.type === 'animate' && data.gesture) {
            console.log('💃 Received animate command:', data.gesture);
            animationReceivedCallbackRef.current?.(data.gesture);
          } else if (data.type === 'caption' && data.text) {
            console.log('📝 Received caption:', data.text);
            captionReceivedCallbackRef.current?.(data.text);
          } else if (data.type === 'ping') {
            // Keepalive ping - no action needed
          }
        } catch (e) {
          console.log('Non-JSON message:', event.data);
        }
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        errorCallbackRef.current?.('WebSocket connection error');
      };

      wsRef.current.onclose = () => {
        setIsConnected(false);
        setIsConnecting(false);
        statusChangeCallbackRef.current?.('disconnected');
        console.log('WebSocket disconnected');
      };
    } catch (error) {
      setIsConnecting(false);
      errorCallbackRef.current?.('Failed to connect to WebSocket server');
    }
  }, [serverUrl]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const sendAudioSegment = useCallback((audioData: ArrayBuffer) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      // Convert ArrayBuffer to base64
      const bytes = new Uint8Array(audioData);
      let binary = '';
      for (let i = 0; i < bytes.byteLength; i++) {
        binary += String.fromCharCode(bytes[i]);
      }
      const base64Audio = btoa(binary);

      const message = {
        audio_segment: base64Audio
      };

      wsRef.current.send(JSON.stringify(message));
      console.log(`Sent audio segment: ${audioData.byteLength} bytes`);
    }
  }, []);

  const sendImage = useCallback((imageData: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const message = {
        image: imageData
      };

      wsRef.current.send(JSON.stringify(message));
      console.log('Sent image to server');
    }
  }, []);

  const sendAudioWithImage = useCallback((audioData: ArrayBuffer, imageData: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const bytes = new Uint8Array(audioData);
      let binary = '';
      for (let i = 0; i < bytes.byteLength; i++) {
        binary += String.fromCharCode(bytes[i]);
      }
      const base64Audio = btoa(binary);

      wsRef.current.send(
        JSON.stringify({
          audio_segment: base64Audio,
          image: imageData
        })
      );
    }
  }, []);

  const sendSignSequence = useCallback((frames: string[]) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          type: 'sign_sequence',
          frames: frames
        })
      );
    }
  }, []);

  const sendConfig = useCallback((config: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          type: 'config',
          ...config
        })
      );
      console.log('Sent config to server:', config);
    }
  }, []);

  const onAudioReceived = useCallback(
    (
      callback: (
        data: string,
        timing?: any,
        rate?: number,
        method?: string
      ) => void
    ) => {
      audioReceivedCallbackRef.current = callback;
    },
    []
  );

  const onInterrupt = useCallback((callback: () => void) => {
    interruptCallbackRef.current = callback;
  }, []);

  const onError = useCallback((callback: (error: string) => void) => {
    errorCallbackRef.current = callback;
  }, []);

  const onAnimationReceived = useCallback((callback: (gesture: string) => void) => {
    animationReceivedCallbackRef.current = callback;
  }, []);

  const onCaptionReceived = useCallback((callback: (text: string) => void) => {
    captionReceivedCallbackRef.current = callback;
  }, []);


  const onStatusChange = useCallback(
    (
      callback: (status: 'connected' | 'disconnected' | 'connecting') => void
    ) => {
      statusChangeCallbackRef.current = callback;
    },
    []
  );

  const value: WebSocketContextType = {
    isConnected,
    isConnecting,
    connect,
    disconnect,
    sendAudioSegment,
    sendImage,
    sendAudioWithImage,
    sendSignSequence,
    sendConfig,
    onAudioReceived,
    onInterrupt,
    onError,
    onStatusChange,
    onAnimationReceived,
    onCaptionReceived,
    browserImage,
    lastMessage
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
};
