import React from 'react';
import { motion } from 'framer-motion';

const Loading = () => {
  return (
    <div className="flex flex-col items-center justify-center p-12 space-y-12">
      <div className="relative flex items-center justify-center">
        {/* Concentric pulsing circles */}
        {[1, 1.5, 2].map((scale, i) => (
          <motion.div
            key={i}
            initial={{ scale: 1, opacity: 0.5 }}
            animate={{ 
              scale: [1, scale + 0.5], 
              opacity: [0.5, 0],
              borderWidth: ["1px", "0px"]
            }}
            transition={{ 
              duration: 2.5, 
              repeat: Infinity, 
              delay: i * 0.6,
              ease: "easeOut" 
            }}
            className="absolute w-24 h-24 border border-neon-indigo/40 rounded-full pointer-events-none"
          />
        ))}

        {/* Futuristic circular scan lines */}
        <motion.div 
          animate={{ rotate: 360 }}
          transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
          className="absolute w-32 h-32 border-t-2 border-r-2 border-neon-indigo/20 rounded-full"
        />

        {/* Sound wave bars - AI Visualizer */}
        <div className="flex items-center justify-center space-x-1.5 h-20 w-40 relative z-10">
          {[...Array(15)].map((_, i) => (
            <motion.div
              key={i}
              animate={{ 
                height: [
                  Math.random() * 20 + 5, 
                  Math.random() * 50 + 20, 
                  Math.random() * 20 + 5
                ],
                opacity: [0.4, 1, 0.4]
              }}
              transition={{ 
                duration: 0.6 + Math.random() * 0.4, 
                repeat: Infinity, 
                delay: i * 0.03,
                ease: "easeInOut"
              }}
              className={`w-1 rounded-full ${
                i % 3 === 0 ? 'bg-neon-indigo' : i % 3 === 1 ? 'bg-purple-500' : 'bg-neon-green'
              } shadow-[0_0_10px_currentColor]`}
            />
          ))}
        </div>
      </div>

      <div className="space-y-4 text-center">
        <motion.p
          animate={{ opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
          className="text-[10px] font-black uppercase tracking-[0.6em] text-neon-indigo drop-shadow-[0_0_8px_rgba(99,102,241,0.5)]"
        >
          Analyzing Voice Signature
        </motion.p>
        
        <div className="flex items-center justify-center space-x-1">
          {[0, 1, 2].map((i) => (
            <motion.div
              key={i}
              animate={{ opacity: [0, 1, 0] }}
              transition={{ duration: 1, repeat: Infinity, delay: i * 0.2 }}
              className="w-1.5 h-1.5 rounded-full bg-neon-indigo"
            />
          ))}
        </div>

        <p className="text-[8px] font-bold text-slate-600 uppercase tracking-[0.3em] font-mono">
          Neural-Net Trace // ID: {Math.random().toString(16).substring(2, 8).toUpperCase()}
        </p>
      </div>
    </div>
  );
};

export default Loading;
