'use client';

import { useState, useEffect } from 'react';
import { useAuth } from '@/contexts/AuthContext';

interface Conversation {
    id: string;
    timestamp: string;
    summary: string;
}

interface HistoryDrawerProps {
    isOpen: boolean;
    onClose: () => void;
}

export default function HistoryDrawer({ isOpen, onClose }: HistoryDrawerProps) {
    const { token } = useAuth();
    const [conversations, setConversations] = useState<Conversation[]>([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        if (isOpen && token) {
            fetchHistory();
        }
    }, [isOpen, token]);

    const fetchHistory = async () => {
        setLoading(true);
        try {
            const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
            const res = await fetch(`${API_URL}/conversations`, {
                headers: {
                    Authorization: `Bearer ${token}`,
                    'ngrok-skip-browser-warning': 'true'
                }
            });
            if (res.ok) {
                const data = await res.json();
                setConversations(data.conversations);
            }
        } catch (error) {
            console.error("Failed to fetch history", error);
        } finally {
            setLoading(false);
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-[100] flex justify-end">
            {/* Backdrop */}
            <div
                className="absolute inset-0 bg-black/50 backdrop-blur-sm transition-opacity"
                onClick={onClose}
            />

            {/* Drawer Content */}
            <div className="relative h-full w-full max-w-sm bg-zinc-950 border-l border-white/10 shadow-2xl transform transition-transform duration-300 ease-in-out p-6 overflow-y-auto">
                <div className="flex items-center justify-between mb-6">
                    <h2 className="text-xl font-bold text-white">History</h2>
                    <button
                        onClick={onClose}
                        className="p-2 text-zinc-400 hover:text-white rounded-full hover:bg-white/5"
                    >
                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                </div>

                {loading ? (
                    <div className="flex justify-center py-8">
                        <div className="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
                    </div>
                ) : conversations.length === 0 ? (
                    <div className="text-center py-8 text-zinc-500">
                        No conversations yet.
                    </div>
                ) : (
                    <div className="space-y-4">
                        {conversations.map((chat) => (
                            <div key={chat.id} className="p-4 rounded-xl bg-white/5 border border-white/5 hover:border-indigo-500/30 transition-colors">
                                <div className="text-xs text-indigo-400 font-medium mb-1">
                                    {new Date(chat.timestamp).toLocaleDateString()} • {new Date(chat.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                </div>
                                <p className="text-sm text-zinc-300 leading-relaxed">
                                    {chat.summary}
                                </p>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
