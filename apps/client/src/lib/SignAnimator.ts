
import * as THREE from 'three';

export class SignAnimator {
    avatar: THREE.Object3D;
    animations: any[][] = []; // Queue of frames
    pending: boolean = false;
    currentFrameIndex: number = 0;

    library: Record<string, any[][]> = {};

    // Animation Settings
    speed: number = 0.1;
    pause: number = 800; // ms between signs

    constructor(avatar: THREE.Object3D) {
        this.avatar = avatar;
    }

    async loadLibrary(url: string = '/animations.json') {
        try {
            const response = await fetch(url);
            const data = await response.json();
            this.library = data;
            console.log(`[SignAnimator] Loaded ${Object.keys(this.data).length} animations.`);
        } catch (e) {
            console.error("[SignAnimator] Failed to load library:", e);
        }
    }

    get data() { return this.library; }

    // Text to Sign Logic
    playText(text: string) {
        console.log(`[SignAnimator] Signing: "${text}"`);
        const words = text.toUpperCase().replace(/[^\w\s]/g, '').split(/\s+/);

        for (const word of words) {
            if (!word) continue;

            if (this.library[word]) {
                // Word exists
                this.enqueueSequence(this.library[word]);
            } else {
                // Fingerspell
                for (const char of word) {
                    if (this.library[char]) {
                        this.enqueueSequence(this.library[char]);
                    }
                }
            }
            // Add pause between words (How? Push empty frame or delay?)
            // We can handle pause in animation loop.
        }

        if (!this.pending) {
            this.pending = true;
            this.animate();
        }
    }

    enqueueSequence(frames: any[][]) {
        // Frames is a list of lists of moves.
        for (const frame of frames) {
            this.animations.push(frame); // Push each frame to queue
        }
    }

    animate = () => {
        if (this.animations.length === 0) {
            this.pending = false;
            return;
        }

        requestAnimationFrame(this.animate);

        const currentFrame = this.animations[0];
        let frameComplete = true;

        if (currentFrame.length > 0) {
            // Check if it's a special delay frame? No, based on Sign-Kit logic:
            // It tries to move bones to target.

            // Iterate moves in current frame
            // [boneName, prop, axis, limit, sign]
            // We need to track if ALL bones reached target.

            let movesActive = false;

            for (let i = 0; i < currentFrame.length; i++) {
                const [boneName, prop, axis, limit, sign] = currentFrame[i];
                // Find bone
                let bone = this.findBone(boneName);
                if (!bone) continue; // Skip missing bones

                // Apply rotation
                // Sign-Kit uses Euler rotation.
                // axis is "x", "y", "z".
                // limit is target value.
                // sign is "+" (increment) or "-" (decrement).

                const currentVal = bone.rotation[axis as 'x' | 'y' | 'z'];
                let reached = false;

                if (sign === "+") {
                    if (currentVal < limit) {
                        bone.rotation[axis as 'x' | 'y' | 'z'] += this.speed;
                        // Clamp
                        if (bone.rotation[axis as 'x' | 'y' | 'z'] > limit) {
                            bone.rotation[axis as 'x' | 'y' | 'z'] = limit;
                        }
                        movesActive = true;
                    } else {
                        reached = true;
                    }
                } else if (sign === "-") {
                    if (currentVal > limit) {
                        bone.rotation[axis as 'x' | 'y' | 'z'] -= this.speed;
                        // Clamp
                        if (bone.rotation[axis as 'x' | 'y' | 'z'] < limit) {
                            bone.rotation[axis as 'x' | 'y' | 'z'] = limit;
                        }
                        movesActive = true;
                    } else {
                        reached = true;
                    }
                }

                // Note: We don't remove completed moves from array because we need to check them next frame?
                // Sign-Kit logic splices them out!
                // "ref.animations[0].splice(i, 1);"
                // If we splice, index handling is tricky.
                // I'll just check conditions.
            }

            if (!movesActive) {
                // All moves done for this frame.
                // Move to next frame.
                this.animations.shift();

                // Add pause if needed? Sign-Kit uses "flag" and setTimeout for pause.
                // For now, minimal pause.
            }
        } else {
            // Empty frame (sometimes used for delays?)
            this.animations.shift();
        }
    }

    findBone(name: string): THREE.Bone | null {
        let bone = this.avatar.getObjectByName(name) as THREE.Bone;
        if (!bone) {
            // Try removing mixamorig prefix
            const shortName = name.replace("mixamorig", "");
            bone = this.avatar.getObjectByName(shortName) as THREE.Bone;
        }
        return bone;
    }
}

// Default Pose (Hardcoded for startup)
export const Sign_Default = (animator: SignAnimator) => {
    // Manually push default pose frame
    // We can simulate a "frame" for loop, or just set strictly.
    // Let's set strictly for startup to be instant.
    const bones = [
        ["mixamorigNeck", "rotation", "x", Math.PI / 12],
        ["mixamorigLeftArm", "rotation", "z", -Math.PI / 3],
        ["mixamorigLeftForeArm", "rotation", "y", -Math.PI / 1.5],
        ["mixamorigRightArm", "rotation", "z", Math.PI / 3],
        ["mixamorigRightForeArm", "rotation", "y", Math.PI / 1.5]
    ];

    bones.forEach(([name, prop, axis, val]) => {
        const bone = animator.findBone(name as string);
        if (bone) {
            bone.rotation[axis as 'x' | 'y' | 'z'] = val as number;
        }
    });
}
