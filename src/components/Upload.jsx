import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import WaveSurfer from 'wavesurfer.js';
import axios from 'axios';
import Loading from './Loading';
import Result from './Result';

const Upload = () => {
  const [file, setFile] = useState(null);
  const [isDragActive, setIsDragActive] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showResult, setShowResult] = useState(false);
  const [error, setError] = useState(null);
  const [analysisData, setAnalysisData] = useState({ score: 0, verdict: 'GENUINE' });
  
  const waveformRef = useRef(null);
  const wavesurfer = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true);
    } else {
      setIsDragActive(false);
    }
  };

  const processFile = (selectedFile) => {
    if (selectedFile && (selectedFile.type === "audio/wav" || selectedFile.type === "audio/mpeg" || selectedFile.name.endsWith('.mp3'))) {
      setFile(selectedFile);
      setError(null);
    } else {
      alert("Please upload a valid .wav or .mp3 file.");
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
    const selectedFile = e.dataTransfer.files[0];
    processFile(selectedFile);
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    processFile(selectedFile);
  };

  const startAnalysis = async () => {
    setIsAnalyzing(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('file', file);

      // Connect to Forensic Backend
      const response = await axios.post('http://localhost:8000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      const { similarity_score, fraud_probability, verdict } = response.data;
      
      // Logic: If similarity_score > 60 -> green UI (GENUINE), else red UI (FRAUD)
      const interpretedVerdict = similarity_score > 60 ? 'GENUINE' : 'FRAUD';
      
      setAnalysisData({ 
          score: similarity_score, 
          verdict: interpretedVerdict 
      });
      
      setIsAnalyzing(false);
      setShowResult(true);
    } catch (err) {
      console.error("Forensic analysis failed:", err);
      const errorMessage = err.response?.data?.detail || "Forensic Connection Failed. Ensure backend is active at http://localhost:8000/predict.";
      setError(errorMessage);
      setIsAnalyzing(false);
    }
  };

  useEffect(() => {
    if (file && waveformRef.current && !isAnalyzing && !showResult && !error) {
      if (wavesurfer.current) {
        wavesurfer.current.destroy();
      }

      wavesurfer.current = WaveSurfer.create({
        container: waveformRef.current,
        waveColor: 'rgba(99, 102, 241, 0.2)',
        progressColor: '#6366F1',
        cursorColor: '#6366F1',
        barWidth: 2,
        barRadius: 3,
        responsive: true,
        height: 80,
        gradient: true,
      });

      const url = URL.createObjectURL(file);
      wavesurfer.current.load(url);

      return () => {
        if (wavesurfer.current) {
          wavesurfer.current.destroy();
        }
        URL.revokeObjectURL(url);
      };
    }
  }, [file, isAnalyzing, showResult, error]);

  if (error) {
    return (
        <motion.div 
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="glass p-12 rounded-[2.5rem] border-neon-red/20 text-center space-y-8 max-w-2xl mx-auto shadow-neon-red/10"
        >
          <div className="w-24 h-24 rounded-3xl bg-neon-red/10 border border-neon-red/30 flex items-center justify-center mx-auto text-neon-red shadow-neon-red/20">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
          </div>
          <div className="space-y-3">
              <h3 className="text-3xl font-black text-white uppercase tracking-wider">Analysis Failed</h3>
              <p className="text-slate-500 font-light tracking-widest leading-relaxed uppercase text-[10px]">
                  {error}
              </p>
          </div>
          <button 
              onClick={() => { setFile(null); setError(null); }}
              className="px-12 py-5 bg-neon-red/10 border border-neon-red/40 rounded-2xl text-[10px] font-black uppercase tracking-[0.4em] text-neon-red hover:bg-neon-red/20 transition-all shadow-lg"
          >
              Re-Initialize System
          </button>
        </motion.div>
    )
  }

  return (
    <div className="w-full max-w-4xl mx-auto space-y-6 min-h-[400px]">
      <AnimatePresence mode="wait">
        {isAnalyzing ? (
          <motion.div
            key="loading-state"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 1.1 }}
            transition={{ duration: 0.5 }}
          >
            <Loading />
          </motion.div>
        ) : showResult ? (
          <motion.div
            key="result-state"
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
          >
            <Result file={file} score={analysisData.score} verdict={analysisData.verdict} />
          </motion.div>
        ) : !file ? (
          <motion.div
            key="upload-dropzone"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: -20 }}
            whileHover={{ 
                rotateX: 2, 
                rotateY: -2, 
                scale: 1.01,
                transition: { duration: 0.3 }
            }}
            className={`relative group h-72 border-2 border-dashed ${
              isDragActive ? 'border-neon-indigo shadow-neon-indigo scale-[1.02]' : 'border-white/10'
            } rounded-[2.5rem] flex flex-col items-center justify-center cursor-pointer transition-all duration-500 hover:border-neon-indigo/50 hover:shadow-neon-indigo/20 overflow-hidden glass`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => document.getElementById('fileInput').click()}
          >
            <div className={`absolute inset-0 bg-neon-indigo/[0.03] transition-opacity duration-300 ${isDragActive ? 'opacity-100' : 'opacity-0'}`} />
            
            <input
              id="fileInput"
              type="file"
              className="hidden"
              accept=".wav,.mp3"
              onChange={handleFileChange}
            />

            <motion.div
              animate={isDragActive ? { y: -10, scale: 1.1 } : { y: 0, scale: 1 }}
              transition={{ type: "spring", stiffness: 300, damping: 20 }}
              className="mb-6 relative"
            >
              <div className="absolute -inset-6 bg-neon-indigo/10 blur-2xl rounded-full animate-pulse z-0" />
              <div className="w-20 h-20 rounded-2xl bg-white/5 border border-white/5 flex items-center justify-center relative z-10 group-hover:border-neon-indigo/30 group-hover:shadow-neon-indigo transition-all duration-500">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 text-neon-indigo" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </div>
            </motion.div>

            <h3 className="text-2xl font-bold mb-2 tracking-tight uppercase">Supply Neural Input</h3>
            <p className="text-slate-500 text-[10px] font-black tracking-[0.5em] uppercase">
              Drag & Drop <span className="text-neon-indigo">.wav</span> or <span className="text-neon-green">.mp3</span>
            </p>

            <div className="absolute bottom-6 left-1/2 -translate-x-1/2 text-[8px] uppercase tracking-[0.6em] font-black text-slate-700">
              Analysis Core v4.0 // Secure Forensic Protocol
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="upload-preview"
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            whileHover={{ 
                rotateX: 1, 
                rotateY: -1,
                transition: { duration: 0.3 }
            }}
            className="glass p-10 rounded-[2.5rem] space-y-8 relative overflow-hidden border-white/10"
          >
            <div className="absolute top-0 right-0 p-4">
                <button 
                onClick={() => setFile(null)}
                className="w-10 h-10 rounded-full bg-white/5 hover:bg-neon-red/10 border border-white/5 hover:border-neon-red/30 flex items-center justify-center transition-all group"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-slate-500 group-hover:text-neon-red" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                </button>
            </div>

            <div className="flex items-center space-x-6">
                <div className="w-16 h-16 rounded-2xl bg-neon-indigo/5 flex items-center justify-center border border-neon-indigo/20 shadow-neon-indigo/10">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-neon-indigo" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                    </svg>
                </div>
                <div>
                  <h4 className="text-xl font-bold truncate max-w-[350px] tracking-tight">{file.name}</h4>
                  <p className="text-[10px] text-slate-500 font-black uppercase tracking-[0.3em] mt-2">
                    Size: {(file.size / (1024 * 1024)).toFixed(2)} MB • Format: {file.name.split('.').pop().toUpperCase()}
                  </p>
                </div>
            </div>

            <div className="relative group p-4 bg-black/20 rounded-3xl border border-white/5">
               <div ref={waveformRef} className="w-full opacity-90 relative z-10" />
            </div>

            <div className="flex gap-6">
              <motion.button 
                onClick={startAnalysis}
                whileHover={{ scale: 1.02, y: -2 }}
                whileTap={{ scale: 0.98 }}
                className="flex-1 py-5 bg-neon-indigo/10 border border-neon-indigo/50 text-neon-indigo rounded-2xl font-black uppercase tracking-[0.3em] text-[10px] hover:bg-neon-indigo/20 shadow-neon-indigo transition-all duration-300"
              >
                Initialize Deepfake Detection Core
              </motion.button>
            </div>
            
            <div className="flex justify-center">
                 <div className="flex items-center space-x-3 text-[8px] uppercase tracking-[0.4em] text-slate-600 font-black">
                    <div className="w-2 h-2 rounded-full bg-neon-green animate-pulse" />
                    <span>Neural Payload Ready for Logic Processing</span>
                </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Upload;
