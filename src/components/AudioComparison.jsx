import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';

const AudioComparison = () => {
  const [fileA, setFileA] = useState(null);
  const [fileB, setFileB] = useState(null);
  const [isComparing, setIsComparing] = useState(false);
  const [comparisonResult, setComparisonResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileA = (e) => {
    const file = e.target.files[0];
    if (file) setFileA(file);
  };

  const handleFileB = (e) => {
    const file = e.target.files[0];
    if (file) setFileB(file);
  };

  const runComparison = async () => {
    if (!fileA || !fileB) {
      alert("Please upload both Audio Sample A and Audio Sample B.");
      return;
    }

    setIsComparing(true);
    setError(null);
    setComparisonResult(null);

    try {
      const formData = new FormData();
      formData.append('file_a', fileA);
      formData.append('file_b', fileB);

      const response = await axios.post('http://localhost:8000/compare', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setComparisonResult(response.data);
    } catch (err) {
      console.error("1-to-1 Comparison failed:", err);
      const msg = err.response?.data?.detail || "Failed to compare audio files. Make sure backend is active at http://localhost:8000";
      setError(msg);
    } finally {
      setIsComparing(false);
    }
  };

  const resetComparison = () => {
    setFileA(null);
    setFileB(null);
    setComparisonResult(null);
    setError(null);
  };

  return (
    <div className="w-full max-w-4xl mx-auto space-y-8">
      {/* Header Info */}
      <div className="text-center space-y-2">
        <h3 className="text-2xl md:text-3xl font-black uppercase tracking-widest text-white">
          1-to-1 Voice & Spectral Comparison
        </h3>
        <p className="text-slate-400 text-xs tracking-widest uppercase font-light">
          Upload two audio samples to compare their 130-D acoustic feature vectors & verify speaker match.
        </p>
      </div>

      {/* Main Dual Upload Area */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Audio File A Card */}
        <div className="glass p-8 rounded-[2rem] border border-white/10 flex flex-col justify-between space-y-6 relative group hover:border-neon-indigo/40 transition-all">
          <div className="flex justify-between items-center">
            <span className="px-4 py-1 rounded-full bg-neon-indigo/10 border border-neon-indigo/30 text-neon-indigo text-[10px] font-black uppercase tracking-wider">
              Audio Sample A
            </span>
            {fileA && (
              <button 
                onClick={() => setFileA(null)} 
                className="text-xs text-slate-500 hover:text-red-400 uppercase font-mono"
              >
                Clear
              </button>
            )}
          </div>

          {!fileA ? (
            <label className="h-44 border-2 border-dashed border-white/10 rounded-2xl flex flex-col items-center justify-center cursor-pointer hover:border-neon-indigo/50 hover:bg-white/[0.02] transition-all">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 text-slate-500 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 100-6 3 3 0 000 6z" />
              </svg>
              <span className="text-xs font-bold uppercase tracking-wider text-slate-400">Select Audio A</span>
              <span className="text-[9px] text-slate-600 uppercase tracking-widest mt-1">.wav or .mp3</span>
              <input type="file" accept=".wav,.mp3" onChange={handleFileA} className="hidden" />
            </label>
          ) : (
            <div className="p-4 rounded-2xl bg-black/40 border border-white/5 space-y-2">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 rounded-xl bg-neon-indigo/20 flex items-center justify-center text-neon-indigo font-bold text-sm">
                  A
                </div>
                <div className="overflow-hidden">
                  <p className="text-sm font-bold truncate text-white">{fileA.name}</p>
                  <p className="text-[10px] text-slate-500 font-mono">{(fileA.size / (1024 * 1024)).toFixed(2)} MB</p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Audio File B Card */}
        <div className="glass p-8 rounded-[2rem] border border-white/10 flex flex-col justify-between space-y-6 relative group hover:border-neon-purple/40 transition-all">
          <div className="flex justify-between items-center">
            <span className="px-4 py-1 rounded-full bg-purple-500/10 border border-purple-500/30 text-purple-400 text-[10px] font-black uppercase tracking-wider">
              Audio Sample B
            </span>
            {fileB && (
              <button 
                onClick={() => setFileB(null)} 
                className="text-xs text-slate-500 hover:text-red-400 uppercase font-mono"
              >
                Clear
              </button>
            )}
          </div>

          {!fileB ? (
            <label className="h-44 border-2 border-dashed border-white/10 rounded-2xl flex flex-col items-center justify-center cursor-pointer hover:border-purple-500/50 hover:bg-white/[0.02] transition-all">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 text-slate-500 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 100-6 3 3 0 000 6z" />
              </svg>
              <span className="text-xs font-bold uppercase tracking-wider text-slate-400">Select Audio B</span>
              <span className="text-[9px] text-slate-600 uppercase tracking-widest mt-1">.wav or .mp3</span>
              <input type="file" accept=".wav,.mp3" onChange={handleFileB} className="hidden" />
            </label>
          ) : (
            <div className="p-4 rounded-2xl bg-black/40 border border-white/5 space-y-2">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 rounded-xl bg-purple-500/20 flex items-center justify-center text-purple-400 font-bold text-sm">
                  B
                </div>
                <div className="overflow-hidden">
                  <p className="text-sm font-bold truncate text-white">{fileB.name}</p>
                  <p className="text-[10px] text-slate-500 font-mono">{(fileB.size / (1024 * 1024)).toFixed(2)} MB</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Action Button */}
      <div className="flex justify-center pt-2">
        <motion.button
          onClick={runComparison}
          disabled={!fileA || !fileB || isComparing}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          className={`px-12 py-5 rounded-2xl font-black uppercase tracking-[0.3em] text-[10px] transition-all duration-300 shadow-xl ${
            !fileA || !fileB
              ? 'bg-slate-800 text-slate-600 border border-slate-700 cursor-not-allowed'
              : 'bg-gradient-to-r from-neon-indigo to-purple-600 text-white border border-neon-indigo/50 hover:shadow-neon-indigo/40'
          }`}
        >
          {isComparing ? "Analyzing 1-to-1 Vectors..." : "Compare Audio Samples"}
        </motion.button>
      </div>

      {/* Error View */}
      {error && (
        <div className="p-6 glass rounded-2xl border border-red-500/30 text-center space-y-2">
          <p className="text-red-400 font-bold text-xs uppercase tracking-wider">{error}</p>
        </div>
      )}

      {/* Results View */}
      {comparisonResult && (
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass p-8 rounded-[2.5rem] border border-white/10 space-y-8 relative overflow-hidden"
        >
          <div className="flex justify-between items-center">
            <h4 className="text-lg font-black uppercase tracking-wider text-white">
              1-to-1 Forensic Comparison Report
            </h4>
            <button 
              onClick={resetComparison} 
              className="px-4 py-2 bg-white/5 rounded-xl text-[10px] font-bold uppercase tracking-widest text-slate-400 hover:text-white hover:bg-white/10 transition-all"
            >
              Reset Test
            </button>
          </div>

          {/* Similarity Metric Gauge */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-center">
            <div className="p-6 rounded-2xl bg-black/40 border border-white/5 space-y-2">
              <span className="text-[10px] uppercase font-bold text-slate-400 tracking-widest">Pair Acoustic Similarity</span>
              <p className="text-4xl font-black text-neon-indigo">{comparisonResult.pair_acoustic_similarity}%</p>
              <span className="text-[9px] text-slate-500 font-mono">130-D Cosine Distance</span>
            </div>

            <div className="p-6 rounded-2xl bg-black/40 border border-white/5 space-y-2">
              <span className="text-[10px] uppercase font-bold text-slate-400 tracking-widest">Speaker Verification</span>
              <p className={`text-xl font-black tracking-wider uppercase ${
                comparisonResult.speaker_verdict === "MATCH_SAME_SPEAKER"
                  ? "text-emerald-400"
                  : comparisonResult.speaker_verdict.includes("DEEPFAKE")
                  ? "text-rose-400"
                  : "text-amber-400"
              }`}>
                {comparisonResult.speaker_verdict.replaceAll('_', ' ')}
              </p>
            </div>

            <div className="p-6 rounded-2xl bg-black/40 border border-white/5 space-y-2">
              <span className="text-[10px] uppercase font-bold text-slate-400 tracking-widest">Deepfake Integrity</span>
              <p className="text-2xl font-black text-white">
                {comparisonResult.file_a.verdict === "GENUINE" && comparisonResult.file_b.verdict === "GENUINE"
                  ? "AUTHENTIC BOTH"
                  : "SYNTHETIC ALERT"}
              </p>
            </div>
          </div>

          {/* Breakdown for File A and File B */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4">
            {/* File A Stats */}
            <div className="p-6 rounded-2xl bg-black/20 border border-neon-indigo/20 space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-xs font-bold text-neon-indigo uppercase">File A: {comparisonResult.file_a.filename}</span>
                <span className={`px-3 py-1 rounded-full text-[9px] font-black uppercase ${
                  comparisonResult.file_a.verdict === "GENUINE" ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/40" : "bg-rose-500/20 text-rose-400 border border-rose-500/40"
                }`}>
                  {comparisonResult.file_a.verdict}
                </span>
              </div>
              <div className="space-y-2 text-xs font-mono text-slate-300">
                <div className="flex justify-between">
                  <span>Fraud Risk:</span>
                  <span className="font-bold text-white">{comparisonResult.file_a.fraud_probability}%</span>
                </div>
                <div className="flex justify-between">
                  <span>Authenticity Score:</span>
                  <span className="font-bold text-white">{comparisonResult.file_a.similarity_score}%</span>
                </div>
              </div>
            </div>

            {/* File B Stats */}
            <div className="p-6 rounded-2xl bg-black/20 border border-purple-500/20 space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-xs font-bold text-purple-400 uppercase">File B: {comparisonResult.file_b.filename}</span>
                <span className={`px-3 py-1 rounded-full text-[9px] font-black uppercase ${
                  comparisonResult.file_b.verdict === "GENUINE" ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/40" : "bg-rose-500/20 text-rose-400 border border-rose-500/40"
                }`}>
                  {comparisonResult.file_b.verdict}
                </span>
              </div>
              <div className="space-y-2 text-xs font-mono text-slate-300">
                <div className="flex justify-between">
                  <span>Fraud Risk:</span>
                  <span className="font-bold text-white">{comparisonResult.file_b.fraud_probability}%</span>
                </div>
                <div className="flex justify-between">
                  <span>Authenticity Score:</span>
                  <span className="font-bold text-white">{comparisonResult.file_b.similarity_score}%</span>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default AudioComparison;
