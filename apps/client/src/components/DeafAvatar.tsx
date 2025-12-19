'use client';

import React, { useEffect, useRef, Suspense } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { useGLTF, Environment, OrbitControls, Html } from '@react-three/drei';
import { SignAnimator, Sign_Default } from '@/lib/SignAnimator';

interface DeafAvatarProps {
    onAnimatorReady: (animator: SignAnimator) => void;
}

function YBotModel({ onAnimatorReady }: { onAnimatorReady: (animator: SignAnimator) => void }) {
    const { scene } = useGLTF('/models/ybot.glb');
    const animatorRef = useRef<SignAnimator | null>(null);

    useEffect(() => {
        if (scene && !animatorRef.current) {
            console.log("[DeafAvatar] YBot Loaded. Initializing SignAnimator.");

            // Initialize Animator
            const animator = new SignAnimator(scene);
            animatorRef.current = animator;
            onAnimatorReady(animatorRef.current);

            // Load Animation Library from JSON
            animatorRef.current.loadLibrary().then(() => {
                console.log("DeafAvatar: Sign Library Loaded");
            });

            // Fix Orientation: Correcting Z-up to Y-up often requires -90 deg rotation.
            scene.rotation.set(0, 0, 0);

            // Trigger Startup Animation (Default Pose from Sign-Kit)
            console.log("DeafAvatar: Triggering Default Pose");
            if (animatorRef.current) {
                Sign_Default(animatorRef.current);
            }
        }
    }, [scene, onAnimatorReady]);

    // Adjust rotation (X: -90 degrees)
    // Raising position Y from -1.0 to -0.5 to fix "bottom" issue
    return <primitive object={scene} position={[0, -0.6, 0]} scale={[1.2, 1.2, 1.2]} rotation={[-Math.PI / 2, 0, 0]} />;
}

export function DeafAvatar({ onAnimatorReady }: DeafAvatarProps) {
    return (
        <div className="w-full h-full bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-indigo-950/80 via-slate-900 to-black rounded-xl overflow-hidden shadow-2xl border border-white/5">
            <Canvas camera={{ position: [0, 1.5, 2.5], fov: 50 }}>
                <ambientLight intensity={1.5} />
                <directionalLight position={[0, 2, 5]} intensity={2} />
                <pointLight position={[-2, 1, 3]} intensity={1} color="white" />
                <pointLight position={[2, 1, 3]} intensity={1} color="white" />

                <Suspense fallback={<Html center><div className="text-white">Loading YBot...</div></Html>}>
                    <YBotModel onAnimatorReady={onAnimatorReady} />
                </Suspense>

                <Environment preset="city" />
                <OrbitControls enableZoom={true} enablePan={false} target={[0, 1.4, 0]} minPolarAngle={Math.PI / 3} maxPolarAngle={Math.PI / 2} />
                {/* Removed Debug Helpers for cleaner UI */}
            </Canvas>
            <div className="absolute bottom-4 right-4 bg-black/40 text-white/80 text-xs px-3 py-1 rounded-full backdrop-blur-sm border border-white/10">
                Deaf Mode Active
            </div>
        </div>
    );
}
