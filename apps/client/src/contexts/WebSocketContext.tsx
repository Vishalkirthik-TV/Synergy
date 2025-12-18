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
}

interface WebSocketContextType {
  isConnected: boolean;
  isConnecting: boolean;
  connect: (userId?: string) => Promise<void>;
  disconnect: () => void;
  sendAudioSegment: (audioData: ArrayBuffer) => void;
  sendImage: (imageData: string) => void;
  sendAudioWithImage: (audioData: ArrayBuffer, imageData: string) => void;
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
  triggerInterrupt: () => void;
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

  // Sequence State
  const currentSeqRef = useRef<number>(0);

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

  const connect = useCallback(async (userId: string = 'test-client') => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      setIsConnecting(true);
      statusChangeCallbackRef.current?.('connecting');

      // Append userId to the base URL
      wsRef.current = new WebSocket(`${serverUrl}${userId}`);

      wsRef.current.onopen = () => {
        setIsConnected(true);
        setIsConnecting(false);
        statusChangeCallbackRef.current?.('connected');
        console.log('WebSocket connected');
      };

      wsRef.current.onmessage = (event) => {
        try {
          const data: WebSocketMessage = JSON.parse(event.data);

          // --- SEQUENCING LOGIC START ---
          if (data.seq !== undefined && data.seq !== null) {
            const incomingSeq = data.seq;
            // If incoming sequence is OLDER than current, ignore it
            if (incomingSeq < currentSeqRef.current) {
              console.log(`🗑️ Ignored stale message (seq=${incomingSeq} < current=${currentSeqRef.current})`, data.type || 'audio');
              return;
            }
            // If NEWER sequence, update our current pointer
            if (incomingSeq > currentSeqRef.current) {
              console.log(`⏩ New sequence detected (seq=${incomingSeq}), updating from ${currentSeqRef.current}`);
              currentSeqRef.current = incomingSeq;
            }
          } else if (currentSeqRef.current > 0 && (data.type === 'interrupt' || (data as any).interrupt)) {
            // STRICT MODE: If we are tracking sequences, IGNORE interrupts without sequence ID
            // This prevents "Ghost Interrupts" from killing valid new audio
            console.warn('🛡️ Ignored ghost interrupt (no seq) because sequence tracking is active.');
            return;
          }
          // --- SEQUENCING LOGIC END ---

          console.log('WebSocket message received:', { type: data.type || (data.audio ? 'audio' : 'unknown'), seq: data.seq });

          if (data.status === 'connected') {
            console.log(
              `Server confirmed connection. Client ID: ${data.client_id}`
            );
          } else if (data.type === 'interrupt' || (data as any).interrupt) {
            console.log(`🛑 Received confirmed interrupt (seq=${data.seq})`);
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
            }

            audioReceivedCallbackRef.current?.(
              data.audio,
              timingData,
              data.sample_rate || 24000,
              data.method || 'unknown'
            );
          } else if (data.type === 'ping') {
            // Send pong? No, just ignore or log
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
          errorCallbackRef.current?.('Failed to parse message from server');
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

  const sendAudioWithImage = useCallback(
    (audioData: ArrayBuffer, imageData: string) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        // Convert ArrayBuffer to base64
        const bytes = new Uint8Array(audioData);
        let binary = '';
        for (let i = 0; i < bytes.byteLength; i++) {
          binary += String.fromCharCode(bytes[i]);
        }
        const base64Audio = btoa(binary);

        const message = {
          audio_segment: base64Audio,
          image: imageData
        };

        wsRef.current.send(JSON.stringify(message));
        console.log(`Sent audio + image: ${audioData.byteLength} bytes audio`);
      }
    },
    []
  );

  // Callback registration methods
  const onAudioReceived = useCallback(
    (
      callback: (
        audioData: string,
        timingData?: any,
        sampleRate?: number,
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

  const onStatusChange = useCallback(
    (
      callback: (status: 'connected' | 'disconnected' | 'connecting') => void
    ) => {
      statusChangeCallbackRef.current = callback;
    },
    []
  );

  const triggerInterrupt = useCallback(() => {
    // 1. Local interrupt (stop audio)
    interruptCallbackRef.current?.();

    // 2. Server interrupt
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'interrupt' }));
    }
  }, []);

  return (
    <WebSocketContext.Provider
      value={{
        isConnected,
        isConnecting,
        connect,
        disconnect,
        sendAudioSegment,
        sendImage,
        sendAudioWithImage,
        onAudioReceived,
        onInterrupt,
        onError,
        onStatusChange,
        triggerInterrupt
      }}
    >
      {children}
    </WebSocketContext.Provider>
  );
};
