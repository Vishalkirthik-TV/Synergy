import { useState, useEffect, useRef } from 'react';

export function useBrowserSpeechRecognition() {
    const [transcript, setTranscript] = useState('');
    const [listening, setListening] = useState(false);
    const recognitionRef = useRef<any>(null); // Use any to avoid type issues with webkitSpeechRecognition

    useEffect(() => {
        if (typeof window !== 'undefined') {
            const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
            if (SpeechRecognition) {
                const recognition = new SpeechRecognition();
                recognition.continuous = true;
                recognition.interimResults = false; // We use final results to avoid jittery signing
                recognition.lang = 'en-US';

                recognition.onresult = (event: any) => {
                    let finalTranscript = '';
                    for (let i = event.resultIndex; i < event.results.length; ++i) {
                        if (event.results[i].isFinal) {
                            finalTranscript += event.results[i][0].transcript;
                        }
                    }

                    if (finalTranscript) {
                        setTranscript(prev => (prev ? prev + ' ' : '') + finalTranscript.trim());
                    }
                };

                recognition.onstart = () => setListening(true);
                recognition.onend = () => setListening(false);
                recognition.onerror = (event: any) => {
                    console.error("Speech Recog Error:", event.error);
                    setListening(false);
                };

                recognitionRef.current = recognition;
            }
        }
    }, []);

    const startListening = () => {
        try {
            recognitionRef.current?.start();
        } catch (e) {
            // Already started
        }
    };
    const stopListening = () => recognitionRef.current?.stop();
    const resetTranscript = () => setTranscript('');

    return { transcript, listening, startListening, stopListening, resetTranscript };
}
