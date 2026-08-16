import React from 'react';
import { motion } from 'framer-motion';

const Hero = () => {
  const scrollToFeatures = () => {
    const featuresSection = document.getElementById('features');
    if (featuresSection) {
      featuresSection.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <section className="relative h-screen flex flex-col items-center justify-center text-center px-4 overflow-hidden bg-transparent">
      {/* Background Glows */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-neon-indigo/10 blur-[120px] rounded-full pointer-events-none" />

      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        whileInView={{ opacity: 1, scale: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 1.2, ease: [0.16, 1, 0.3, 1] }}
        className="z-10 flex flex-col items-center max-w-5xl"
      >
        <motion.div
           initial={{ opacity: 0, y: 30 }}
           animate={{ opacity: 1, y: 0 }}
           transition={{ duration: 0.8, delay: 0.2 }}
        >
          <h1 
            className="text-6xl md:text-8xl font-bold tracking-tighter mb-6 leading-[1.1] font-display"
          >
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-neon-indigo via-purple-400 via-pink-500 to-neon-indigo bg-[length:200%_auto] animate-[textGradient_8s_linear_infinite] drop-shadow-[0_0_15px_rgba(99,102,241,0.3)]">
              Trust Voices.<br />
              Detect Deception.
            </span>
          </h1>
        </motion.div>

        <motion.p 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="text-xl md:text-2xl text-slate-400 mb-12 max-w-3xl font-light tracking-wide px-4"
        >
          Protecting digital identities with advanced <span className="text-neon-indigo font-medium">neural scaling</span> and <span className="text-neon-green font-medium">real-time de-noising</span> technology.
        </motion.p>

        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="relative group"
        >
          <div className="absolute -inset-1.5 bg-gradient-to-r from-neon-indigo to-purple-600 rounded-2xl blur opacity-30 group-hover:opacity-100 transition duration-700 group-hover:duration-200 animate-pulse-slow"></div>
          <motion.button 
            whileHover={{ 
              scale: 1.05, 
              y: -4,
              boxShadow: "0 20px 40px rgba(99, 102, 241, 0.4)"
            }}
            whileTap={{ scale: 0.95 }}
            className="relative px-12 py-5 rounded-2xl bg-black border border-white/10 text-white font-bold tracking-[0.2em] uppercase text-xs flex items-center gap-4 transition-all duration-300 shadow-2xl glass overflow-hidden"
          >
            <span className="relative z-10">Start Analysis</span>
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-neon-indigo relative z-10" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z" clipRule="evenodd" />
            </svg>
            <div className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/5 to-white/0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000"></div>
          </motion.button>
        </motion.div>
      </motion.div>

      {/* Functional Scroll Down Indicator */}
      <motion.button 
        onClick={scrollToFeatures}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.5, duration: 1 }}
        whileHover={{ 
            y: 5,
            scale: 1.1,
            transition: { duration: 0.2 }
        }}
        className="absolute bottom-12 left-1/2 -translate-x-1/2 flex flex-col items-center gap-3 text-slate-500 hover:text-white transition-colors cursor-pointer group z-20"
      >
        <span className="text-[10px] uppercase tracking-[0.4em] font-black opacity-80 group-hover:opacity-100">Explore System</span>
        <motion.div 
          animate={{ y: [0, 10, 0], opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut" }}
          className="w-px h-16 bg-gradient-to-b from-neon-indigo via-neon-indigo/50 to-transparent rounded-full shadow-[0_0_8px_rgba(99,102,241,0.5)]"
        />
      </motion.button>

      {/* Decorative ambient flare */}
      <div className="absolute top-0 right-0 w-1/2 h-1/2 bg-neon-indigo/[0.02] blur-[150px] pointer-events-none" />
    </section>
  );
};

export default Hero;
