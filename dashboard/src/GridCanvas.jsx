import React, { useRef, useEffect, useState } from 'react';

// Grid hazard coords imported from env definition
const HAZARDS = [
    [5, 3], [5, 4], [5, 5], [5, 6], [8, 6], [9, 6], [7, 9], [8, 9],
    [12, 13], [13, 13], [13, 12], [10, 10], [11, 10], [12, 10], [3, 11]
];
const GOAL = [14, 14];
const GRID_SIZE = 16;
const TILE_SIZE = 24;

export default function GridCanvas({ algo }) {
    const canvasRef = useRef(null);
    const [replay, setReplay] = useState(null);
    const [step, setStep] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);

    useEffect(() => {
        if (!algo) return;
        setStep(0);
        setIsPlaying(false);
        fetch(`/results/replays/${algo}.json`)
            .then(r => r.json())
            .then(data => setReplay(data))
            .catch(e => console.error(e));
    }, [algo]);

    useEffect(() => {
        if (isPlaying && replay && step < replay.length - 1) {
            const timer = setTimeout(() => setStep(s => s + 1), 60);
            return () => clearTimeout(timer);
        } else if (step >= (replay?.length || 0) - 1) {
            setIsPlaying(false);
        }
    }, [isPlaying, step, replay]);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        // Clear
        ctx.fillStyle = '#1e1e24';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw Grid lines
        ctx.strokeStyle = '#33333e';
        for (let x = 0; x <= GRID_SIZE; x++) {
            ctx.beginPath();
            ctx.moveTo(x * TILE_SIZE, 0);
            ctx.lineTo(x * TILE_SIZE, GRID_SIZE * TILE_SIZE);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(0, x * TILE_SIZE);
            ctx.lineTo(GRID_SIZE * TILE_SIZE, x * TILE_SIZE);
            ctx.stroke();
        }

        // Draw Hazards
        ctx.fillStyle = 'rgba(255, 60, 60, 0.4)';
        HAZARDS.forEach(h => {
            ctx.fillRect(h[0] * TILE_SIZE, h[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE);
        });

        // Draw Goal
        ctx.fillStyle = 'rgba(60, 255, 60, 0.6)';
        ctx.fillRect(GOAL[0] * TILE_SIZE, GOAL[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE);

        // Draw Agent Trail
        if (replay && step > 0) {
            ctx.beginPath();
            ctx.moveTo(replay[0].x * TILE_SIZE + TILE_SIZE / 2, replay[0].y * TILE_SIZE + TILE_SIZE / 2);
            ctx.strokeStyle = 'rgba(100, 200, 255, 0.3)';
            ctx.lineWidth = 2;
            for (let i = 1; i <= step; i++) {
                ctx.lineTo(replay[i].x * TILE_SIZE + TILE_SIZE / 2, replay[i].y * TILE_SIZE + TILE_SIZE / 2);
            }
            ctx.stroke();
        }

        // Draw Current Agent
        if (replay && replay.length > 0) {
            const current = replay[step];
            ctx.fillStyle = '#4facfe';
            ctx.beginPath();
            ctx.arc(current.x * TILE_SIZE + TILE_SIZE / 2, current.y * TILE_SIZE + TILE_SIZE / 2, TILE_SIZE * 0.4, 0, 2 * Math.PI);
            ctx.fill();

            // Direction Indicator
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(current.x * TILE_SIZE + TILE_SIZE / 2, current.y * TILE_SIZE + TILE_SIZE / 2);
            const angles = [0, Math.PI / 2, Math.PI, 3 * Math.PI / 2];
            const dirAngle = angles[current.dir] || 0;
            ctx.lineTo(current.x * TILE_SIZE + TILE_SIZE / 2 + Math.cos(dirAngle) * TILE_SIZE * 0.4,
                current.y * TILE_SIZE + TILE_SIZE / 2 + Math.sin(dirAngle) * TILE_SIZE * 0.4);
            ctx.stroke();
        }
    }, [replay, step]);

    return (
        <div className="canvas-container">
            <canvas ref={canvasRef} width={GRID_SIZE * TILE_SIZE} height={GRID_SIZE * TILE_SIZE} />
            <div className="controls">
                <button onClick={() => setIsPlaying(!isPlaying)}>
                    {isPlaying ? "Pause" : "Play Replay"}
                </button>
                <button onClick={() => setStep(0)}>Reset</button>
                <span className="step-counter">
                    Step: {step} / {replay ? replay.length - 1 : 0} |
                    Violation: {replay && replay[step]?.violation ? "YES" : "NO"} |
                    Reward: {replay && replay[step]?.reward.toFixed(2)}
                </span>
            </div>
        </div>
    );
}
