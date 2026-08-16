import React, { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import WaveSurfer from 'wavesurfer.js';

const Result = ({ file, score, verdict }) => {
  const waveformRef = useRef(null);
  const wavesurfer = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    if (file && waveformRef.current) {
      if (wavesurfer.current) {
        wavesurfer.current.destroy();
      }

      wavesurfer.current = WaveSurfer.create({
        container: waveformRef.current,
        waveColor: 'rgba(255, 255, 255, 0.03)',
        progressColor: verdict === 'GENUINE' ? '#22C55E' : '#EF4444',
        cursorColor: '#6366F1',
        barWidth: 2,
        barRadius: 3,
        responsive: true,
        height: 80,
        gradient: true,
      });

      const url = URL.createObjectURL(file);
      wavesurfer.current.load(url);

      wavesurfer.current.on('play', () => setIsPlaying(true));
      wavesurfer.current.on('pause', () => setIsPlaying(false));
      wavesurfer.current.on('finish', () => setIsPlaying(false));

      return () => {
        if (wavesurfer.current) {
          wavesurfer.current.destroy();
        }
        URL.revokeObjectURL(url);
      };
    }
  }, [file, verdict]);

  const togglePlay = () => {
    if (wavesurfer.current) {
      wavesurfer.current.playPause();
    }
  };

  const isGenuine = verdict === 'GENUINE';

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.15,
        delayChildren: 0.2
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: { 
      y: 0, 
      opacity: 1,
      transition: { duration: 0.8, ease: [0.16, 1, 0.3, 1] }
    }
  };

  return (
    <motion.div 
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="w-full max-w-4xl mx-auto space-y-8 pb-12"
    >
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        
        {/* Score Ring */}
        <motion.div 
            variants={itemVariants}
            className="glass p-10 flex flex-col items-center justify-center space-y-6 rounded-[2.5rem] relative overflow-hidden"
        >
          <div className="absolute inset-0 bg-gradient-to-b from-white/[0.02] to-transparent pointer-events-none" />
          <div className="relative w-44 h-44">
            <svg className="w-full h-full transform -rotate-90">
              <circle
                cx="88"
                cy="88"
                r="78"
                stroke="currentColor"
                strokeWidth="10"
                fill="transparent"
                className="text-white/5"
              />
              <motion.circle
                cx="88"
                cy="88"
                r="78"
                stroke="currentColor"
                strokeWidth="10"
                fill="transparent"
                strokeDasharray="490"
                initial={{ strokeDashoffset: 490 }}
                animate={{ strokeDashoffset: 490 - (490 * score) / 100 }}
                transition={{ duration: 2.5, ease: [0.16, 1, 0.3, 1], delay: 0.5 }}
                className={isGenuine ? "text-neon-green drop-shadow-neon-green" : "text-neon-red drop-shadow-neon-red"}
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <motion.span 
                initial={{ opacity: 0, scale: 0.5 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 1.2, duration: 0.5 }}
                className="text-5xl font-black font-mono tracking-tighter"
              >
                {Math.round(score)}%
              </motion.span>
              <span className="text-[10px] uppercase tracking-[0.4em] text-slate-600 font-black mt-1">Accuracy</span>
            </div>
          </div>
          <p className="text-[10px] uppercase tracking-[0.5em] font-black text-slate-500 text-center relative z-10">Neural Confidence</p>
        </motion.div>

        <motion.div 
          variants={itemVariants}
          whileHover={{ rotateX: 2, rotateY: -2, scale: 1.005 }}
          className={`md:col-span-2 glass p-10 flex flex-col items-center justify-center border-2 rounded-[2.5rem] relative overflow-hidden ${isGenuine ? 'border-neon-green/20' : 'border-neon-red/20'}`}
        >
          {/* Animated Background Glow */}
          <motion.div 
            animate={{ 
                scale: [1, 1.2, 1],
                opacity: [0.1, 0.2, 0.1] 
            }}
            transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
            className={`absolute -top-24 -right-24 w-80 h-80 blur-[100px] rounded-full ${isGenuine ? 'bg-neon-green' : 'bg-neon-red'}`} 
          />

          <motion.div
            animate={{ y: [0, -8, 0] }}
            transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
            className={`w-24 h-24 rounded-3xl flex items-center justify-center mb-8 shadow-2xl border relative z-10 ${isGenuine ? 'bg-neon-green/10 text-neon-green border-neon-green/30' : 'bg-neon-red/10 text-neon-red border-neon-red/30'}`}
          >
            {isGenuine ? (
              <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            )}
          </motion.div>
          
          <h3 className={`text-4xl md:text-6xl font-black italic uppercase tracking-[-0.05em] mb-4 relative z-10 ${isGenuine ? 'text-neon-green drop-shadow-neon-green' : 'text-neon-red drop-shadow-neon-red'}`}>
            {isGenuine ? 'Genuine Voice' : 'Fraud Detected'}
          </h3>
          <p className="text-slate-500 font-light tracking-[0.2em] text-[10px] md:text-xs uppercase text-center max-w-md leading-relaxed relative z-10">
            {isGenuine 
              ? 'Biometric consistency verified. No synthetic signatures or neural artifacts identified.' 
              : 'Critical neural artifacts detected. Inconsistencies matched with deepfake generation patterns.'}
          </p>

          <div className="mt-12 pt-8 border-t border-white/5 w-full relative z-10">
            <div className="flex justify-between items-center mb-4">
               <span className="text-[10px] font-black uppercase tracking-[0.3em] text-slate-500">Forensic Probability</span>
               <span className={`text-xs font-mono font-black ${isGenuine ? 'text-neon-green' : 'text-neon-red'}`}>
                {isGenuine ? (100 - score).toFixed(2) : score.toFixed(2)}%
               </span>
            </div>
            <div className="w-full h-3 bg-white/5 rounded-full overflow-hidden p-0.5 border border-white/5">
                <motion.div 
                    initial={{ width: 0 }}
                    animate={{ width: `${isGenuine ? (100 - score) : score}%` }}
                    transition={{ duration: 2, delay: 1, ease: [0.16, 1, 0.3, 1] }}
                    className={`h-full rounded-full ${isGenuine ? 'bg-neon-green shadow-[0_0_15px_#22C55E]' : 'bg-neon-red shadow-[0_0_15px_#EF4444]'}`} 
                />
            </div>
          </div>
        </motion.div>
      </div>

      {/* Waveform Player */}
      <motion.div 
        variants={itemVariants}
        className="glass p-12 rounded-[2.5rem] space-y-8 border-white/10 shadow-2xl relative"
      >
        <div className="flex flex-col md:flex-row items-center gap-10">
            <button 
                onClick={togglePlay}
                className={`w-20 h-20 rounded-3xl border flex items-center justify-center transition-all duration-500 shadow-2xl relative overflow-hidden group shrink-0 ${
                    isGenuine 
                    ? 'bg-neon-green/5 border-neon-green/20 text-neon-green hover:bg-neon-green/10 shadow-neon-green/5' 
                    : 'bg-neon-red/5 border-neon-red/20 text-neon-red hover:bg-neon-red/10 shadow-neon-red/5'
                }`}
            >
                <div className="absolute inset-0 bg-white/5 translate-y-full group-hover:translate-y-0 transition-transform duration-500" />
                {isPlaying ? (
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 relative z-10" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M6 4h4v16H6V4zm8 0h4v16h4V4z" />
                    </svg>
                ) : (
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 ml-1 relative z-10" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M8 5v14l11-7z" />
                    </svg>
                )}
            </button>
            <div className="flex-1 w-full relative">
                <div className="absolute inset-x-0 h-px top-1/2 bg-white/5 pointer-events-none" />
                <div ref={waveformRef} className="w-full relative z-10 opacity-60 hover:opacity-100 transition-opacity duration-300" />
            </div>
        </div>
        
        <div className="flex flex-col md:flex-row justify-between items-center gap-6 pt-4 border-t border-white/5">
            <div className="flex items-center space-x-3 text-[10px] font-black uppercase tracking-[0.4em] text-slate-600">
                <div className={`w-2 h-2 rounded-full animate-pulse ${isGenuine ? 'bg-neon-green drop-shadow-neon-green' : 'bg-neon-red drop-shadow-neon-red'}`} />
                <span>Neural Payload // Secure Forensic Trace Active</span>
            </div>
            <motion.button 
                onClick={() => window.location.reload()}
                whileHover={{ scale: 1.05, y: -2, backgroundColor: "rgba(255,255,255,0.08)" }}
                whileTap={{ scale: 0.95 }}
                className="px-12 py-4 rounded-2xl bg-white/5 border border-white/10 text-[10px] font-black uppercase tracking-[0.3em] text-slate-300 hover:bg-white/10 hover:border-white/20 transition-all duration-300 shadow-xl"
            >
                Re-Initialize Sweep
            </motion.button>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default Result;
