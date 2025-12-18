'use client';

import { useState } from 'react';
import VoiceActivityDetector from '@/components/VoiceActivityDetector';
import TalkingHead from '@/components/TalkingHead';
import { CameraToggleButton } from '@/components/CameraStream';
import { useWebSocketContext } from '@/contexts/WebSocketContext';

export default function Home() {
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const [username, setUsername] = useState('');
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const { connect, isConnected } = useWebSocketContext();

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    if (username.trim()) {
      setIsLoggedIn(true);
      connect(username);
    }
  };

  if (!isLoggedIn) {
    return (
      <main className="flex min-h-screen items-center justify-center bg-gradient-to-br from-gray-50 to-gray-100">
        <div className="w-full max-w-md rounded-lg bg-white p-8 shadow-xl">
          <div className="mb-8 text-center">
            <h1 className="mb-2 text-4xl font-bold text-gray-900">TalkMateAI</h1>
            <p className="text-gray-600">Your personalized AI companion</p>
          </div>
          <form onSubmit={handleLogin} className="space-y-6">
            <div>
              <label htmlFor="username" className="mb-2 block text-sm font-medium text-gray-700">
                What's your name?
              </label>
              <input
                type="text"
                id="username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="w-full rounded-md border border-gray-300 px-4 py-2 text-gray-900 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Ex. Saran"
                required
              />
            </div>
            <button
              type="submit"
              className="w-full rounded-md bg-blue-600 px-4 py-2 font-medium text-white transition-colors hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
            >
              Start Chatting
            </button>
          </form>
        </div>
      </main>
    );
  }

  return (
    <main className="relative min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8 text-center">
          <h1 className="mb-2 text-4xl font-bold text-gray-900">TalkMateAI</h1>
          <p className="text-lg text-gray-600">
            Chatting with <span className="font-semibold text-blue-600">{username}</span>
          </p>
        </div>

        {/* Main Content Layout */}
        <div className="mb-8 grid grid-cols-1 gap-8 xl:grid-cols-2">
          {/* TalkingHead Component */}
          <div className="order-1">
            <div className="rounded-lg bg-white p-6 shadow-lg">
              <TalkingHead cameraStream={cameraStream} />
            </div>
          </div>

          {/* Voice Activity Detector */}
          <div className="order-2">
            <VoiceActivityDetector cameraStream={cameraStream} />
          </div>
        </div>
      </div>

      {/* Floating Camera Component */}
      <CameraToggleButton onStreamChange={setCameraStream} />
    </main>
  );
}
