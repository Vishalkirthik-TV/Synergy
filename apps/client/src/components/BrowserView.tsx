'use strict';
import React from 'react';
import { Card } from '@/components/ui/card';
import { Loader2 } from 'lucide-react';

interface BrowserViewProps {
    image: string | null;
}

const BrowserView: React.FC<BrowserViewProps> = ({ image }) => {
    if (!image) return null;

    return (
        <Card className="absolute top-4 right-4 w-96 z-50 overflow-hidden shadow-2xl border-2 border-primary/20 bg-black/80 backdrop-blur">
            <div className="relative aspect-video w-full bg-black flex items-center justify-center">
                {image && image !== 'LOADING' ? (
                    <img
                        src={`data:image/jpeg;base64,${image}`}
                        alt="Agent Browser View"
                        className="w-full h-full object-contain"
                    />
                ) : (
                    <div className="flex flex-col items-center gap-2 text-muted-foreground">
                        <Loader2 className="h-8 w-8 animate-spin" />
                        <span className="text-xs">Connecting to Browser...</span>
                    </div>
                )}

                <div className="absolute bottom-2 right-2 px-2 py-1 bg-black/50 text-white text-[10px] rounded backdrop-blur-sm">
                    Agent View
                </div>
            </div>
        </Card>
    );
};

export default BrowserView;
