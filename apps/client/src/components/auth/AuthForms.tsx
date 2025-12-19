'use client';

import { useState } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { Loader2 } from 'lucide-react';

export default function AuthForms() {
  const [isLoginMode, setIsLoginMode] = useState(true);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const { login, signup } = useAuth();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      if (isLoginMode) {
        await login(username, password);
      } else {
        await signup(username, password);
      }
    } catch (err: any) {
      setError(err.message || 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="w-full max-w-md rounded-2xl border border-white/10 bg-zinc-900/60 p-8 shadow-2xl backdrop-blur-xl">
      <div className="mb-8 text-center">
        <h1 className="mb-2 text-4xl font-bold tracking-tight text-white">Nova</h1>
        <p className="text-zinc-400">Your intelligent multilingual companion</p>
      </div>

      <div className="mb-8 flex justify-center space-x-4 border-b border-white/5 pb-4">
        <button
          className={`pb-2 text-lg font-medium transition-all ${isLoginMode
              ? 'border-b-2 border-indigo-500 text-indigo-400'
              : 'text-zinc-500 hover:text-zinc-300'
            }`}
          onClick={() => {
            setIsLoginMode(true);
            setError('');
          }}
        >
          Login
        </button>
        <button
          className={`pb-2 text-lg font-medium transition-all ${!isLoginMode
              ? 'border-b-2 border-indigo-500 text-indigo-400'
              : 'text-zinc-500 hover:text-zinc-300'
            }`}
          onClick={() => {
            setIsLoginMode(false);
            setError('');
          }}
        >
          Sign Up
        </button>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <label className="mb-2 block text-sm font-medium text-zinc-300">
            Username
          </label>
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            className="w-full rounded-lg border border-zinc-800 bg-zinc-950/50 px-4 py-3 text-white placeholder-zinc-500 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/20 focus:outline-none transition-all"
            placeholder="Enter your username"
            required
          />
        </div>

        <div>
          <label className="mb-2 block text-sm font-medium text-zinc-300">
            Password
          </label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full rounded-lg border border-zinc-800 bg-zinc-950/50 px-4 py-3 text-white placeholder-zinc-500 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/20 focus:outline-none transition-all"
            placeholder="••••••••"
            required
            minLength={4}
          />
        </div>

        {error && (
          <div className="rounded-lg border border-red-500/20 bg-red-500/10 p-3 text-sm text-red-400">
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={isLoading}
          className="flex w-full items-center justify-center rounded-lg bg-indigo-600 px-4 py-3 font-semibold text-white transition-all hover:bg-indigo-500 focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 focus:ring-offset-zinc-900 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-indigo-500/20"
        >
          {isLoading ? (
            <Loader2 className="mr-2 h-5 w-5 animate-spin" />
          ) : isLoginMode ? (
            'Login to Nova'
          ) : (
            'Create Account'
          )}
        </button>
      </form>
    </div>
  );
}
