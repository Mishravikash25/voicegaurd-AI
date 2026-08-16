import React, { useState } from 'react';
import { motion } from 'framer-motion';
import Hero from './components/Hero';
import Upload from './components/Upload';
import AudioComparison from './components/AudioComparison';
import ProfileManager from './components/ProfileManager';
import BackgroundEffects from './effects/BackgroundEffects';

function App() {
  const [analysisMode, setAnalysisMode] = useState('single'); // 'single', 'compare', 'profiles'

  return (
    <div className="min-h-screen bg-slate-950 relative overflow-hidden selection:bg-neon-indigo/30 selection:text-white">
      {/* Advanced Global Visuals */}
      <BackgroundEffects />

      {/* Main Content */}
      <main className="relative z-10 w-full scroll-smooth">
        {/* Sticky Utility Header */}
        <header className="fixed top-0 left-0 w-full p-8 z-50 flex justify-between items-center pointer-events-none">
          <div className="text-xl font-black tracking-[0.3em] uppercase pointer-events-auto mix-blend-difference">
            <span className="text-neon-indigo">Voice</span>Guard
          </div>
          <div className="text-[10px] font-bold tracking-[0.5em] uppercase text-slate-500 pointer-events-auto hidden md:block">
            Secure Forensic Protocol // AR-2041
          </div>
        </header>

        <Hero />
        
        {/* Analysis Section */}
        <section id="analysis" className="min-h-screen py-32 flex flex-col items-center justify-center px-4 relative">
          <motion.div
            initial={{ opacity: 0, y: 60 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 1, ease: "easeOut" }}
            className="w-full z-10 max-w-5xl mx-auto"
          >
            <div className="text-center mb-12">
              <h2 className="text-4xl md:text-6xl font-black mb-6 font-display uppercase tracking-[0.2em]">
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-white via-neon-indigo to-neon-indigo bg-[length:200%_auto] animate-[textGradient_10s_linear_infinite]">
                  Neural Analysis Core
                </span>
              </h2>
              <div className="w-24 h-1 bg-gradient-to-r from-transparent via-neon-indigo to-transparent mx-auto mb-8 opacity-50 shadow-neon-indigo" />
              
              {/* Mode Switcher Buttons */}
              <div className="flex flex-wrap justify-center gap-4 max-w-3xl mx-auto p-2 bg-black/40 rounded-2xl border border-white/10 glass">
                <button
                  onClick={() => setAnalysisMode('single')}
                  className={`px-6 py-3 rounded-xl text-xs font-black uppercase tracking-wider transition-all ${
                    analysisMode === 'single'
                      ? 'bg-neon-indigo text-white shadow-neon-indigo'
                      : 'text-slate-400 hover:text-white hover:bg-white/5'
                  }`}
                >
                  ⚡ Single File Forensic Scan
                </button>

                <button
                  onClick={() => setAnalysisMode('compare')}
                  className={`px-6 py-3 rounded-xl text-xs font-black uppercase tracking-wider transition-all ${
                    analysisMode === 'compare'
                      ? 'bg-gradient-to-r from-neon-indigo to-purple-600 text-white shadow-purple-500/20'
                      : 'text-slate-400 hover:text-white hover:bg-white/5'
                  }`}
                >
                  ⚔️ 1-to-1 Audio Comparison
                </button>

                <button
                  onClick={() => setAnalysisMode('profiles')}
                  className={`px-6 py-3 rounded-xl text-xs font-black uppercase tracking-wider transition-all ${
                    analysisMode === 'profiles'
                      ? 'bg-emerald-600 text-white shadow-emerald-500/20'
                      : 'text-slate-400 hover:text-white hover:bg-white/5'
                  }`}
                >
                  👤 Target Person & Datasets
                </button>
              </div>
            </div>
            
            {/* Mode Views */}
            {analysisMode === 'single' && <Upload />}
            {analysisMode === 'compare' && <AudioComparison />}
            {analysisMode === 'profiles' && <ProfileManager />}
          </motion.div>
        </section>

        {/* Features Section */}
        <section id="features" className="min-h-[50vh] py-32 flex flex-col items-center justify-center px-4 border-t border-white/5 bg-black/40 backdrop-blur-3xl">
          <div className="max-w-6xl w-full">
            <h3 className="text-3xl md:text-4xl font-bold mb-16 text-center tracking-[0.3em] uppercase opacity-80">
                System Capabilities
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              {[
                { title: "Harmonic Vectoring", desc: "Analysis of voice timbre and frequency fluctuations beyond human perception." },
                { title: "Synthetic Signature", desc: "Detection of AI-generated artifacts and digital stitching patterns." },
                { title: "Biometric Matching", desc: "Cross-referencing audio traits against verified neural identity fingerprints." }
              ].map((feat, i) => (
                <motion.div 
                    key={i} 
                    initial={{ opacity: 0, y: 30 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true, margin: "-50px" }}
                    whileHover={{ 
                        rotateX: 5, 
                        rotateY: -5,
                        y: -10,
                        transition: { duration: 0.4, ease: "easeOut" }
                    }}
                    transition={{ delay: i * 0.15, duration: 0.8 }}
                    className="glass p-10 hover:border-neon-indigo/40 group transition-all duration-700 rounded-[2.5rem] relative"
                >
                  <div className="absolute inset-0 bg-gradient-to-br from-white/[0.03] to-transparent pointer-events-none rounded-[2.5rem]" />
                  <div className="w-14 h-14 rounded-2xl bg-neon-indigo/5 flex items-center justify-center mb-8 border border-white/5 group-hover:border-neon-indigo/30 group-hover:drop-shadow-neon-indigo transition-all duration-500">
                    <div className="w-3 h-3 rounded-full bg-neon-indigo group-hover:animate-ping" />
                  </div>
                  <h4 className="text-lg font-black mb-4 uppercase tracking-widest">{feat.title}</h4>
                  <p className="text-slate-500 font-light text-sm leading-relaxed">
                    {feat.desc}
                  </p>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="py-20 text-center border-t border-white/5 opacity-30">
            <div className="text-[8px] uppercase tracking-[1em] font-black">
                VoiceGuard AI // Global Security Infrastructure
            </div>
        </footer>
      </main>
    </div>
  );
}

export default App;
