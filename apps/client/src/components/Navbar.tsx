'use client';

import Link from 'next/link';
import { useAuth } from '@/contexts/AuthContext';

import { useState } from 'react';
import HistoryDrawer from './HistoryDrawer';

export default function Navbar() {
    const { user, isAuthenticated, logout } = useAuth();
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
    const [isHistoryOpen, setIsHistoryOpen] = useState(false);

    return (
        <>
            <nav className="sticky top-0 z-50 border-b border-white/5 bg-zinc-950/80 backdrop-blur-md">
                <div className="container mx-auto flex items-center justify-between px-4 py-3">
                    {/* Brand */}
                    <a href="/" className="flex items-center gap-2 group">
                        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 font-bold text-white shadow-lg shadow-indigo-500/20 ring-1 ring-white/10 transition-transform group-hover:scale-105">
                            N
                        </div>
                        <span className="text-xl font-bold text-white tracking-tight">Nova</span>
                    </a>

                    {/* Right Side Actions (Desktop) */}
                    <div className="hidden sm:flex items-center gap-4">
                        {isAuthenticated ? (
                            <div className="flex items-center gap-4 border-l border-white/10 pl-4">
                                <button
                                    onClick={() => setIsHistoryOpen(true)}
                                    className="text-sm font-medium text-zinc-400 transition-colors hover:text-white"
                                >
                                    History
                                </button>
                                <Link
                                    href="/reminders"
                                    className="text-sm font-medium text-zinc-400 transition-colors hover:text-white"
                                >
                                    Reminders
                                </Link>
                                <span className="text-sm text-zinc-400">
                                    Hi, <span className="font-semibold text-zinc-200">{user?.username}</span>
                                </span>
                                <button
                                    onClick={logout}
                                    className="rounded-full border border-white/5 bg-zinc-900 px-4 py-1.5 text-sm font-medium text-zinc-300 transition-all hover:bg-zinc-800 hover:text-white hover:border-white/10"
                                >
                                    Logout
                                </button>
                            </div>
                        ) : null}
                    </div>

                    {/* Mobile Menu Button */}
                    <div className="flex sm:hidden">
                        {isAuthenticated && (
                            <button
                                onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                                className="text-zinc-400 hover:text-white focus:outline-none"
                            >
                                <svg
                                    className="h-6 w-6"
                                    fill="none"
                                    viewBox="0 0 24 24"
                                    stroke="currentColor"
                                >
                                    {isMobileMenuOpen ? (
                                        <path
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                            strokeWidth={2}
                                            d="M6 18L18 6M6 6l12 12"
                                        />
                                    ) : (
                                        <path
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                            strokeWidth={2}
                                            d="M4 6h16M4 12h16M4 18h16"
                                        />
                                    )}
                                </svg>
                            </button>
                        )}
                    </div>
                </div>

                {/* Mobile Menu Dropdown */}
                {isAuthenticated && isMobileMenuOpen && (
                    <div className="sm:hidden border-t border-white/5 bg-zinc-950 px-4 py-4 shadow-xl">
                        <div className="flex flex-col space-y-4">
                            <span className="text-sm text-zinc-400">
                                Hi, <span className="font-semibold text-zinc-200">{user?.username}</span>
                            </span>
                            <button
                                onClick={() => {
                                    setIsMobileMenuOpen(false);
                                    setIsHistoryOpen(true);
                                }}
                                className="text-base font-medium text-zinc-300 hover:text-white text-left"
                            >
                                History
                            </button>
                            <Link
                                href="/reminders"
                                className="text-base font-medium text-zinc-300 hover:text-white"
                                onClick={() => setIsMobileMenuOpen(false)}
                            >
                                Reminders
                            </Link>
                            <button
                                onClick={() => {
                                    setIsMobileMenuOpen(false);
                                    logout();
                                }}
                                className="w-full text-left rounded-lg bg-zinc-900 px-4 py-2 text-sm font-medium text-zinc-300 hover:bg-zinc-800 hover:text-white"
                            >
                                Logout
                            </button>
                        </div>
                    </div>
                )}
            </nav>
            <HistoryDrawer isOpen={isHistoryOpen} onClose={() => setIsHistoryOpen(false)} />
        </>
    );
}
