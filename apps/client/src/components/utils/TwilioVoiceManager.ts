import { Device, Call } from '@twilio/voice-sdk';

class TwilioVoiceManager {
    private device: Device | null = null;
    private currentCall: Call | null = null;
    private tokenEndpoint = 'http://localhost:8000/api/twilio-token'; // Adjust if needed

    private static instance: TwilioVoiceManager;

    private constructor() { }

    public static getInstance(): TwilioVoiceManager {
        if (!TwilioVoiceManager.instance) {
            TwilioVoiceManager.instance = new TwilioVoiceManager();
        }
        return TwilioVoiceManager.instance;
    }

    private listeners: { [key: string]: Function[] } = {};

    public on(event: string, callback: Function) {
        if (!this.listeners[event]) {
            this.listeners[event] = [];
        }
        this.listeners[event].push(callback);
    }

    public off(event: string, callback: Function) {
        if (!this.listeners[event]) return;
        this.listeners[event] = this.listeners[event].filter(cb => cb !== callback);
    }

    private emit(event: string, data?: any) {
        if (!this.listeners[event]) return;
        this.listeners[event].forEach(cb => cb(data));
    }

    async initialize() {
        try {
            console.log('Fetching Twilio Token...');
            const response = await fetch(this.tokenEndpoint);
            if (!response.ok) throw new Error('Failed to fetch token');

            const data = await response.json();
            const token = data.token;

            console.log('Initializing Twilio Device...');
            this.device = new Device(token, {
                logLevel: 1,
            });

            this.device.on('registered', () => {
                console.log('✅ Twilio Device Registered');
            });

            this.device.on('error', (error) => {
                console.error('❌ Twilio Device Error:', error);
                this.emit('error', error);
            });

            this.device.on('incoming', (call) => {
                console.log("📞 Incoming Call:", call);
                call.accept();
                this.currentCall = call;
                this.emit('connected', call);

                call.on('disconnect', () => {
                    console.log('Incoming call disconnected');
                    this.currentCall = null;
                    this.emit('disconnected');
                });
            });

            await this.device.register();
        } catch (error) {
            console.error('Error initializing Twilio Voice:', error);
            this.emit('error', error);
        }
    }

    async makeCall(targetNumber: string) {
        if (!this.device) {
            await this.initialize();
        }

        if (!this.device) {
            console.error("Cannot make call, device not initialized");
            return;
        }

        try {
            console.log(`📞 Calling ${targetNumber}...`);
            const params = { To: targetNumber };

            const call = await this.device.connect({ params });
            this.currentCall = call;

            call.on('accept', () => {
                console.log('✅ Call Accepted');
                this.emit('connected', call);
            });

            call.on('disconnect', () => {
                console.log('Call Disconnected');
                this.currentCall = null;
                this.emit('disconnected');
            });

            call.on('error', (err) => {
                console.error('Call Error:', err);
                this.emit('error', err);
            });

        } catch (error) {
            console.error("Error making call:", error);
            this.emit('error', error);
        }
    }

    disconnect() {
        if (this.currentCall) {
            console.log("Emptying current call...");
            this.currentCall.disconnect();
            this.currentCall = null;
            this.emit('disconnected'); // Ensure UI updates immediately
        }
    }
}

export default TwilioVoiceManager;
