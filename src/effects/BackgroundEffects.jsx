import React, { useMemo } from 'react';
import { motion, useSpring, useTransform } from 'framer-motion';
import { useMousePosition } from '../hooks/useMousePosition';
import ParticleField from './ParticleField';

const FloatingBlob = ({ color, size, initialX, initialY, duration, delay }) => {
  return (
    <motion.div
      initial={{ x: initialX, y: initialY }}
      animate={{ 
        x: [initialX, initialX + 30, initialX - 15, initialX],
        y: [initialY, initialY + 40, initialY - 20, initialY],
      }}
      transition={{ 
        duration, 
        repeat: Infinity, 
        delay, 
        ease: "easeInOut" 
      }}
      className={`absolute rounded-full blur-[120px] opacity-20 pointer-events-none z-0`}
      style={{ 
        backgroundColor: color,
        width: size,
        height: size,
        willChange: 'transform',
        left: initialX,
        top: initialY
      }}
    />
  );
};

const BackgroundEffects = () => {
  const { x, y } = useMousePosition();
  
  // Parallax calculations with smooth springs
  const springConfig = { damping: 40, stiffness: 200 };
  const mouseX = useSpring(x, springConfig);
  const mouseY = useSpring(y, springConfig);

  // Parallax layers (blobs move more, grid moves less)
  const blobX = useTransform(mouseX, [0, 1920], [-40, 40]);
  const blobY = useTransform(mouseY, [0, 1080], [-40, 40]);
  
  const gridX = useTransform(mouseX, [0, 1920], [-10, 10]);
  const gridY = useTransform(mouseY, [0, 1080], [-10, 10]);

  const blobs = useMemo(() => [
    { color: '#6366F1', size: '450px', initialX: '5%', initialY: '5%', duration: 18, delay: 0 },
    { color: '#8B5CF6', size: '550px', initialX: '65%', initialY: '45%', duration: 25, delay: 3 },
    { color: '#3B82F6', size: '350px', initialX: '85%', initialY: '5%', duration: 20, delay: 5 },
    { color: '#10B981', size: '400px', initialX: '15%', initialY: '75%', duration: 28, delay: 2 },
  ], []);

  return (
    <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
      {/* Animated Gradient Background Base */}
      <div className="absolute inset-0 bg-slate-950/90 z-0" />
      
      {/* Three.js Particle Field */}
      <ParticleField />
      
      {/* Floating Parallax Blobs */}
      <motion.div 
        style={{ x: blobX, y: blobY }}
        className="absolute inset-0 z-10 pointer-events-none"
      >
        {blobs.map((blob, i) => (
          <FloatingBlob key={i} {...blob} />
        ))}
      </motion.div>

      {/* Parallax Grid Overlay */}
      <motion.div 
        style={{ x: gridX, y: gridY }}
        className="absolute inset-0 opacity-[0.04] z-20 pointer-events-none"
      >
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#ffffff_1px,transparent_1px),linear-gradient(to_bottom,#ffffff_1px,transparent_1px)] bg-[size:60px_60px]" />
      </motion.div>

      {/* Global Grain/Noise Texture */}
      <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-[0.08] pointer-events-none z-30 mix-blend-overlay" />

      {/* Interactive Cursor Glow */}
      <motion.div
        className="fixed w-[700px] h-[700px] rounded-full blur-[150px] bg-neon-indigo/[0.06] pointer-events-none z-40"
        style={{
          x: mouseX,
          y: mouseY,
          translateX: '-50%',
          translateY: '-50%',
          willChange: 'transform'
        }}
      />
      
      {/* Vignette Overlay */}
      <div className="absolute inset-0 shadow-[inset_0_0_200px_rgba(0,0,0,0.9)] z-50 pointer-events-none" />
    </div>
  );
};

export default BackgroundEffects;
