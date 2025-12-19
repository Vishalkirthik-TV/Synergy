import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';

export async function loadYBotToScene(scene: THREE.Scene, url: string): Promise<THREE.Group> {
    return new Promise((resolve, reject) => {
        const loader = new GLTFLoader();
        loader.load(url, (gltf) => {
            const model = gltf.scene;

            // Adjust scale/position to match TalkingHead standards
            model.scale.set(1, 1, 1);
            model.position.set(0, -1.5, 0); // Approx position

            // Enable shadows
            model.traverse((child: any) => {
                if (child.isMesh) {
                    child.castShadow = true;
                    child.receiveShadow = true;
                }
            });

            // Remove existing avatar from scene if any?
            // Caller handles that.

            scene.add(model);
            console.log("YBot loaded manually to scene", model);
            resolve(model);
        }, undefined, (error) => {
            reject(error);
        });
    });
}
