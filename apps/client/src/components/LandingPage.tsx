'use client';

import { ArrowRight, Globe, Mic, Video, Zap, MessageSquare, Shield, CheckCircle2 } from 'lucide-react';
import { useState } from 'react';

interface LandingPageProps {
    onGetStarted: () => void;
}

export default function LandingPage({ onGetStarted }: LandingPageProps) {
    return (
        <div className="min-h-screen w-full bg-zinc-950 text-white selection:bg-indigo-500/30">

            {/* Background Effects */}
            <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none">
                <div className="absolute top-0 left-1/4 h-[500px] w-[500px] rounded-full bg-indigo-500/10 blur-[120px]" />
                <div className="absolute bottom-0 right-1/4 h-[500px] w-[500px] rounded-full bg-violet-500/10 blur-[120px]" />
            </div>

            <div className="relative z-10 mx-auto max-w-7xl px-6 pt-20 pb-32 lg:px-8">

                {/* Hero */}
                <div className="flex flex-col items-center text-center">
                    <div className="inline-flex items-center gap-2 rounded-full border border-zinc-800 bg-zinc-900/50 px-3 py-1 text-xs font-medium text-zinc-400 backdrop-blur-md">
                        <span className="relative flex h-2 w-2">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-indigo-500"></span>
                        </span>
                        Nova AI v1.0 is now live
                    </div>

                    <h1 className="mt-8 max-w-4xl text-5xl font-extrabold tracking-tight text-white sm:text-7xl lg:text-8xl">
                        The Future of <br className="hidden sm:block" />
                        <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-violet-400">
                            Connection
                        </span>
                    </h1>

                    <p className="mt-6 max-w-2xl text-lg text-zinc-400 sm:text-xl">
                        Break language barriers instantly. Nova combines real-time voice translation, vision analysis, and empathetic AI into a single, elegant interface.
                    </p>

                    <div className="mt-10 flex flex-col sm:flex-row gap-4">
                        <button
                            onClick={onGetStarted}
                            className="group relative inline-flex items-center justify-center gap-2 rounded-full bg-white px-8 py-3.5 text-base font-semibold text-zinc-950 transition-all hover:bg-zinc-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 focus:ring-offset-zinc-950"
                        >
                            Get Started Free
                            <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
                        </button>
                        <button
                            onClick={onGetStarted}
                            className="inline-flex items-center justify-center gap-2 rounded-full border border-zinc-800 bg-zinc-900/50 px-8 py-3.5 text-base font-semibold text-white transition-all hover:bg-zinc-800 hover:border-zinc-700 backdrop-blur-sm"
                        >
                            Log In
                        </button>
                    </div>
                </div>

                {/* Floating UI Elements / Image Replacement */}
                <div className="mt-20 relative isolate">
                    <div className="rounded-2xl border border-white/10 bg-white/5 p-2 backdrop-blur-xl ring-1 ring-black/5">
                        <div className="rounded-xl overflow-hidden bg-zinc-900 border border-white/5 aspect-[16/9] flex items-center justify-center relative shadow-2xl">
                            {/* Abstract Representation of UI */}
                            <div className="absolute inset-0 bg-gradient-to-tr from-indigo-500/10 via-transparent to-violet-500/10 pointer-events-none" />
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 p-12 w-full max-w-5xl">
                                {/* Fake UI Card 1 */}
                                <div className="bg-zinc-950/80 border border-white/10 rounded-2xl p-6 flex flex-col items-center text-center">
                                    <div className="h-12 w-12 rounded-xl bg-indigo-500/20 flex items-center justify-center mb-4 text-indigo-400">
                                        <Mic className="h-6 w-6" />
                                    </div>
                                    <h3 className="text-lg font-semibold text-white">Voice First</h3>
                                    <p className="text-sm text-zinc-500 mt-2">Just speak. Nova listens, translates, and responds in real-time.</p>
                                </div>
                                {/* Fake UI Card 2 (Center) */}
                                <div className="bg-gradient-to-b from-zinc-800 to-zinc-950 border border-white/10 rounded-2xl p-6 flex flex-col items-center text-center scale-110 shadow-xl shadow-indigo-500/10 z-10">
                                    <div className="h-16 w-16 rounded-full bg-indigo-600 flex items-center justify-center mb-4 text-white shadow-lg shadow-indigo-500/30">
                                        <Zap className="h-8 w-8" />
                                    </div>
                                    <h3 className="text-xl font-bold text-white">Instant AI</h3>
                                    <p className="text-sm text-zinc-400 mt-2">Powered by Groq & Gemini Flash for sub-second latency.</p>
                                </div>
                                {/* Fake UI Card 3 */}
                                <div className="bg-zinc-950/80 border border-white/10 rounded-2xl p-6 flex flex-col items-center text-center">
                                    <div className="h-12 w-12 rounded-xl bg-violet-500/20 flex items-center justify-center mb-4 text-violet-400">
                                        <Globe className="h-6 w-6" />
                                    </div>
                                    <h3 className="text-lg font-semibold text-white">Global Reach</h3>
                                    <p className="text-sm text-zinc-500 mt-2">Seamless translation across 30+ languages.</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>


                {/* Features Grid */}
                <div className="mt-32">
                    <div className="text-center mb-16">
                        <h2 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">Engineered for Excellence</h2>
                        <p className="mt-4 text-lg text-zinc-400">Modern tools for modern communication.</p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                        {[
                            {
                                title: "Multilingual Intelligence",
                                desc: "Detects and speaks 10+ languages fluently with native accents.",
                                icon: Globe,
                                color: "text-blue-400"
                            },
                            {
                                title: "Visual Understanding",
                                desc: "Show Nova anything via camera. She sees what you see.",
                                icon: Video,
                                color: "text-purple-400"
                            },
                            {
                                title: "Private & Secure",
                                desc: "Your conversations are encrypted and private by default.",
                                icon: Shield,
                                color: "text-emerald-400"
                            }
                        ].map((item, i) => (
                            <div key={i} className="group relative rounded-2xl border border-zinc-800 bg-zinc-900/50 p-8 hover:bg-zinc-800/50 transition-colors">
                                <div className={`mb-4 inline-flex rounded-lg bg-zinc-950 p-3 ring-1 ring-white/10 ${item.color}`}>
                                    <item.icon className="h-6 w-6" />
                                </div>
                                <h3 className="text-lg font-semibold text-white">{item.title}</h3>
                                <p className="mt-2 text-zinc-400 leading-relaxed">{item.desc}</p>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Footer */}
                <div className="mt-32 border-t border-zinc-800 pt-8 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <div className="flex h-6 w-6 items-center justify-center rounded bg-indigo-600/20 text-indigo-400 font-bold text-xs ring-1 ring-indigo-500/30">N</div>
                        <span className="text-sm font-semibold text-zinc-300">Nova AI</span>
                    </div>
                    <p className="text-sm text-zinc-600">© 2024 Nova Inc.</p>
                </div>

            </div>
        </div>
    );
}
