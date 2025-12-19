'use client';

import { useState } from 'react';
import { useWebSocketContext } from '@/contexts/WebSocketContext';

const SUPPORTED_LANGUAGES = [
    { code: 'en', name: 'English', flag: '🇬🇧' },
    { code: 'hi', name: 'हिन्दी (Hindi)', flag: '🇮🇳' },
    { code: 'es', name: 'Español (Spanish)', flag: '🇪🇸' },
    { code: 'fr', name: 'Français (French)', flag: '🇫🇷' },
    { code: 'de', name: 'Deutsch (German)', flag: '🇩🇪' },
    { code: 'zh', name: '中文 (Chinese)', flag: '🇨🇳' },
    { code: 'ja', name: '日本語 (Japanese)', flag: '🇯🇵' },
    { code: 'ko', name: '한국어 (Korean)', flag: '🇰🇷' },
];

export default function LanguageSelector() {
    const [selectedLanguage, setSelectedLanguage] = useState('en');
    const [isOpen, setIsOpen] = useState(false);
    const { sendConfig } = useWebSocketContext();

    const handleLanguageChange = (code: string) => {
        setSelectedLanguage(code);
        setIsOpen(false);

        // Send language config to server
        if (sendConfig) {
            sendConfig({ language: code });
        }
    };

    const selectedLang = SUPPORTED_LANGUAGES.find(lang => lang.code === selectedLanguage) || SUPPORTED_LANGUAGES[0];

    return (
        <div className="relative">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="flex items-center gap-2 rounded-lg bg-white px-4 py-2 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
                aria-label="Select language"
            >
                <span className="text-xl">{selectedLang.flag}</span>
                <span>{selectedLang.name}</span>
                <svg
                    className={`h-4 w-4 transition-transform ${isOpen ? 'rotate-180' : ''}`}
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
            </button>

            {isOpen && (
                <div className="absolute right-0 z-10 mt-2 w-56 origin-top-right rounded-md bg-white shadow-lg ring-1 ring-black ring-opacity-5">
                    <div className="py-1" role="menu">
                        {SUPPORTED_LANGUAGES.map((lang) => (
                            <button
                                key={lang.code}
                                onClick={() => handleLanguageChange(lang.code)}
                                className={`flex w-full items-center gap-3 px-4 py-2 text-left text-sm hover:bg-gray-100 ${selectedLanguage === lang.code ? 'bg-blue-50 text-blue-700' : 'text-gray-700'
                                    }`}
                                role="menuitem"
                            >
                                <span className="text-xl">{lang.flag}</span>
                                <span>{lang.name}</span>
                                {selectedLanguage === lang.code && (
                                    <svg className="ml-auto h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
                                        <path
                                            fillRule="evenodd"
                                            d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                                            clipRule="evenodd"
                                        />
                                    </svg>
                                )}
                            </button>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}