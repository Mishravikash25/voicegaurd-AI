import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import axios from 'axios';

const ProfileManager = () => {
  const [activeTab, setActiveTab] = useState('compare'); // 'compare', 'enroll', 'dataset'
  const [profiles, setProfiles] = useState(['Joe Biden', 'Barack Obama', 'Donald Trump', 'Elon Musk', 'Linus Torvalds', 'Margot Robbie']);
  
  // Profile Match State
  const [selectedTarget, setSelectedTarget] = useState('Joe Biden');
  const [matchFile, setMatchFile] = useState(null);
  const [isComparing, setIsComparing] = useState(false);
  const [matchResult, setMatchResult] = useState(null);

  // Enrollment State (Multiple Files Support)
  const [enrollName, setEnrollName] = useState('');
  const [enrollDesc, setEnrollDesc] = useState('');
  const [enrollFiles, setEnrollFiles] = useState([]);
  const [isEnrolling, setIsEnrolling] = useState(false);
  const [enrollSuccess, setEnrollSuccess] = useState(null);

  // Dataset State (Multiple Files Support)
  const [datasetCategory, setDatasetCategory] = useState('REAL');
  const [datasetFiles, setDatasetFiles] = useState([]);
  const [isUploadingDataset, setIsUploadingDataset] = useState(false);
  const [datasetSuccess, setDatasetSuccess] = useState(null);

  const [error, setError] = useState(null);

  const fetchProfiles = async () => {
    try {
      const res = await axios.get('http://localhost:8000/profiles');
      if (res.data && res.data.profiles) {
        setProfiles(res.data.profiles);
        if (!selectedTarget && res.data.profiles.length > 0) {
          setSelectedTarget(res.data.profiles[0]);
        }
      }
    } catch (err) {
      console.warn("Could not load dynamic profiles, using defaults.");
    }
  };

  useEffect(() => {
    fetchProfiles();
  }, []);

  // Run Profile Match
  const handleProfileCompare = async () => {
    if (!matchFile || !selectedTarget) {
      alert("Please select a target person and upload an audio file.");
      return;
    }

    setIsComparing(true);
    setError(null);
    setMatchResult(null);

    try {
      const formData = new FormData();
      formData.append('target_speaker', selectedTarget);
      formData.append('file', matchFile);

      const res = await axios.post('http://localhost:8000/profiles/compare', formData);
      setMatchResult(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || "Profile comparison failed.");
    } finally {
      setIsComparing(false);
    }
  };

  // Run Profile Enrollment with Multiple Audio Files
  const handleEnroll = async (e) => {
    e.preventDefault();
    if (!enrollName.trim() || enrollFiles.length === 0) {
      alert("Please provide speaker name and select at least one reference audio file.");
      return;
    }

    setIsEnrolling(true);
    setError(null);
    setEnrollSuccess(null);

    try {
      const formData = new FormData();
      formData.append('speaker_name', enrollName.trim());
      formData.append('description', enrollDesc);

      // Append all selected files for multi-file composite profile creation!
      Array.from(enrollFiles).forEach((file) => {
        formData.append('files', file);
      });

      const res = await axios.post('http://localhost:8000/profiles/enroll', formData);
      setEnrollSuccess(res.data.message);
      setEnrollName('');
      setEnrollDesc('');
      setEnrollFiles([]);
      fetchProfiles();
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to enroll voice profile.");
    } finally {
      setIsEnrolling(false);
    }
  };

  // Run Dataset Import with Multiple Files
  const handleDatasetUpload = async (e) => {
    e.preventDefault();
    if (datasetFiles.length === 0) {
      alert("Please choose audio files or a ZIP archive to upload.");
      return;
    }

    setIsUploadingDataset(true);
    setError(null);
    setDatasetSuccess(null);

    try {
      const formData = new FormData();
      formData.append('category', datasetCategory);

      Array.from(datasetFiles).forEach((file) => {
        formData.append('files', file);
      });

      const res = await axios.post('http://localhost:8000/dataset/upload', formData);
      setDatasetSuccess(res.data.message);
      setDatasetFiles([]);
    } catch (err) {
      setError(err.response?.data?.detail || "Dataset upload failed.");
    } finally {
      setIsUploadingDataset(false);
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto space-y-8">
      {/* Sub-Navigation Tabs */}
      <div className="flex justify-center border-b border-white/10 pb-4 gap-4">
        {[
          { id: 'compare', label: '👤 Target Voice Match' },
          { id: 'enroll', label: '➕ Enroll New Profile (Multi-Sample)' },
          { id: 'dataset', label: '📂 Insert Custom Dataset' }
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => { setActiveTab(tab.id); setError(null); }}
            className={`px-6 py-3 rounded-xl text-xs font-black uppercase tracking-wider transition-all ${
              activeTab === tab.id
                ? 'bg-neon-indigo/20 text-neon-indigo border border-neon-indigo/40 shadow-neon-indigo/20'
                : 'text-slate-400 hover:text-white hover:bg-white/5'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {error && (
        <div className="p-4 rounded-xl glass border border-red-500/30 text-red-400 text-xs font-bold text-center">
          {error}
        </div>
      )}

      {/* TAB 1: Target Voice Match */}
      {activeTab === 'compare' && (
        <div className="glass p-8 rounded-[2.5rem] border border-white/10 space-y-6">
          <div className="text-center space-y-2">
            <h4 className="text-xl font-bold uppercase tracking-wider text-white">Compare Audio Against Target Profile</h4>
            <p className="text-xs text-slate-400 font-light">Select an enrolled person's voice print & test if incoming audio matches their voice profile.</p>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">
                Select Target Person Profile
              </label>
              <select
                value={selectedTarget}
                onChange={(e) => setSelectedTarget(e.target.value)}
                className="w-full p-4 rounded-xl bg-black/50 border border-white/10 text-white font-bold text-sm focus:border-neon-indigo focus:outline-none"
              >
                {profiles.map((p) => (
                  <option key={p} value={p} className="bg-slate-900 text-white">{p}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">
                Upload Test Audio File
              </label>
              <input
                type="file"
                accept=".wav,.mp3,.flac,.ogg"
                onChange={(e) => setMatchFile(e.target.files[0])}
                className="w-full p-4 rounded-xl bg-black/30 border border-white/10 text-slate-300 text-xs file:mr-4 file:py-2 file:px-4 file:rounded-xl file:border-0 file:text-xs file:font-bold file:bg-neon-indigo/20 file:text-neon-indigo hover:file:bg-neon-indigo/30"
              />
            </div>

            <div className="pt-4 flex justify-center">
              <button
                onClick={handleProfileCompare}
                disabled={isComparing || !matchFile}
                className={`px-10 py-4 rounded-xl font-black uppercase tracking-widest text-xs transition-all ${
                  !matchFile || isComparing
                    ? 'bg-slate-800 text-slate-600 cursor-not-allowed'
                    : 'bg-neon-indigo text-white hover:shadow-neon-indigo'
                }`}
              >
                {isComparing ? 'Comparing Profile...' : `Match Against ${selectedTarget}`}
              </button>
            </div>
          </div>

          {/* Match Result Display */}
          {matchResult && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="p-6 rounded-2xl bg-black/40 border border-white/10 space-y-4"
            >
              <div className="flex justify-between items-center border-b border-white/10 pb-3">
                <span className="text-xs font-bold uppercase text-slate-400">Target Speaker: {matchResult.target_speaker}</span>
                <span className={`px-3 py-1 rounded-full text-[9px] font-black uppercase ${
                  matchResult.is_speaker_match ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40' : 'bg-rose-500/20 text-rose-400 border border-rose-500/40'
                }`}>
                  {matchResult.is_speaker_match ? 'MATCH CONFIRMED' : 'NO MATCH'}
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 rounded-xl bg-white/[0.02] border border-white/5 space-y-1">
                  <span className="text-[10px] text-slate-500 uppercase font-bold">Profile Similarity</span>
                  <p className="text-2xl font-black text-neon-indigo">{matchResult.target_profile_similarity}%</p>
                </div>
                <div className="p-4 rounded-xl bg-white/[0.02] border border-white/5 space-y-1">
                  <span className="text-[10px] text-slate-500 uppercase font-bold">Deepfake Fraud Score</span>
                  <p className="text-2xl font-black text-white">{matchResult.deepfake_analysis.fraud_probability}%</p>
                </div>
              </div>

              <div className="p-4 rounded-xl bg-black/30 border border-white/5">
                <p className="text-xs font-bold text-slate-200">{matchResult.verdict}</p>
              </div>
            </motion.div>
          )}
        </div>
      )}

      {/* TAB 2: Enroll New Profile (Multi-Sample Support) */}
      {activeTab === 'enroll' && (
        <form onSubmit={handleEnroll} className="glass p-8 rounded-[2.5rem] border border-white/10 space-y-6">
          <div className="text-center space-y-2">
            <h4 className="text-xl font-bold uppercase tracking-wider text-white">Enroll New Voice Profile (Multi-Sample)</h4>
            <p className="text-xs text-slate-400 font-light">Add a target person by selecting one or MULTIPLE audio files at once to build a composite voice embedding profile.</p>
          </div>

          {enrollSuccess && (
            <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-bold text-center">
              {enrollSuccess}
            </div>
          )}

          <div className="space-y-4">
            <div>
              <label className="block text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">
                Speaker Full Name *
              </label>
              <input
                type="text"
                required
                placeholder="e.g. Shrey (Custom User)"
                value={enrollName}
                onChange={(e) => setEnrollName(e.target.value)}
                className="w-full p-4 rounded-xl bg-black/50 border border-white/10 text-white font-bold text-sm focus:border-neon-indigo focus:outline-none"
              />
            </div>

            <div>
              <label className="block text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">
                Description / Notes
              </label>
              <input
                type="text"
                placeholder="e.g. Official Multi-Sample Reference Dataset"
                value={enrollDesc}
                onChange={(e) => setEnrollDesc(e.target.value)}
                className="w-full p-4 rounded-xl bg-black/50 border border-white/10 text-slate-300 text-sm focus:border-neon-indigo focus:outline-none"
              />
            </div>

            <div>
              <div className="flex justify-between items-center mb-2">
                <label className="block text-xs font-bold uppercase tracking-wider text-slate-400">
                  Reference Audio Samples (Multiple Files Allowed) *
                </label>
                {enrollFiles.length > 0 && (
                  <span className="text-[10px] font-bold text-emerald-400 bg-emerald-500/10 px-2.5 py-1 rounded-full border border-emerald-500/30">
                    Selected {enrollFiles.length} File(s)
                  </span>
                )}
              </div>
              <input
                type="file"
                required
                multiple
                accept=".wav,.mp3,.flac,.ogg,.zip"
                onChange={(e) => setEnrollFiles(e.target.files)}
                className="w-full p-4 rounded-xl bg-black/30 border border-white/10 text-slate-300 text-xs file:mr-4 file:py-2 file:px-4 file:rounded-xl file:border-0 file:text-xs file:font-bold file:bg-neon-indigo/20 file:text-neon-indigo"
              />
              {enrollFiles.length > 0 && (
                <div className="mt-2 p-3 rounded-xl bg-black/20 border border-white/5 max-h-32 overflow-y-auto">
                  <p className="text-[10px] font-bold text-slate-400 uppercase mb-1">Selected Files:</p>
                  <ul className="text-xs text-slate-300 space-y-1">
                    {Array.from(enrollFiles).map((f, i) => (
                      <li key={i} className="truncate">🎵 {f.name} ({(f.size / 1024).toFixed(1)} KB)</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            <div className="pt-4 flex justify-center">
              <button
                type="submit"
                disabled={isEnrolling || enrollFiles.length === 0}
                className={`px-10 py-4 rounded-xl font-black uppercase tracking-widest text-xs transition-all ${
                  isEnrolling || enrollFiles.length === 0
                    ? 'bg-slate-800 text-slate-600 cursor-not-allowed'
                    : 'bg-neon-indigo text-white hover:shadow-neon-indigo'
                }`}
              >
                {isEnrolling ? `Averaging ${enrollFiles.length} Voice Samples...` : `Save & Enroll Profile (${enrollFiles.length} Samples)`}
              </button>
            </div>
          </div>
        </form>
      )}

      {/* TAB 3: Insert Custom Dataset (Multi-File Support) */}
      {activeTab === 'dataset' && (
        <form onSubmit={handleDatasetUpload} className="glass p-8 rounded-[2.5rem] border border-white/10 space-y-6">
          <div className="text-center space-y-2">
            <h4 className="text-xl font-bold uppercase tracking-wider text-white">Insert Custom Audio Dataset (Multi-File)</h4>
            <p className="text-xs text-slate-400 font-light">Upload multiple `.wav` / `.mp3` files or `.zip` dataset archives at once to expand your local training dataset.</p>
          </div>

          {datasetSuccess && (
            <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-bold text-center">
              {datasetSuccess}
            </div>
          )}

          <div className="space-y-4">
            <div>
              <label className="block text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">
                Dataset Category
              </label>
              <select
                value={datasetCategory}
                onChange={(e) => setDatasetCategory(e.target.value)}
                className="w-full p-4 rounded-xl bg-black/50 border border-white/10 text-white font-bold text-sm focus:border-neon-indigo focus:outline-none"
              >
                <option value="REAL" className="bg-slate-900 text-white">REAL (Authentic Voice Samples)</option>
                <option value="FAKE" className="bg-slate-900 text-white">FAKE (Deepfake / Synthetic Samples)</option>
              </select>
            </div>

            <div>
              <div className="flex justify-between items-center mb-2">
                <label className="block text-xs font-bold uppercase tracking-wider text-slate-400">
                  Choose Audio Files or ZIP Dataset (Multiple Files Allowed) *
                </label>
                {datasetFiles.length > 0 && (
                  <span className="text-[10px] font-bold text-emerald-400 bg-emerald-500/10 px-2.5 py-1 rounded-full border border-emerald-500/30">
                    Selected {datasetFiles.length} File(s)
                  </span>
                )}
              </div>
              <input
                type="file"
                required
                multiple
                accept=".wav,.mp3,.flac,.ogg,.zip"
                onChange={(e) => setDatasetFiles(e.target.files)}
                className="w-full p-4 rounded-xl bg-black/30 border border-white/10 text-slate-300 text-xs file:mr-4 file:py-2 file:px-4 file:rounded-xl file:border-0 file:text-xs file:font-bold file:bg-neon-indigo/20 file:text-neon-indigo"
              />
              {datasetFiles.length > 0 && (
                <div className="mt-2 p-3 rounded-xl bg-black/20 border border-white/5 max-h-32 overflow-y-auto">
                  <p className="text-[10px] font-bold text-slate-400 uppercase mb-1">Files to Import:</p>
                  <ul className="text-xs text-slate-300 space-y-1">
                    {Array.from(datasetFiles).map((f, i) => (
                      <li key={i} className="truncate">📦 {f.name} ({(f.size / 1024).toFixed(1)} KB)</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            <div className="pt-4 flex justify-center">
              <button
                type="submit"
                disabled={isUploadingDataset || datasetFiles.length === 0}
                className={`px-10 py-4 rounded-xl font-black uppercase tracking-widest text-xs transition-all ${
                  isUploadingDataset || datasetFiles.length === 0
                    ? 'bg-slate-800 text-slate-600 cursor-not-allowed'
                    : 'bg-neon-indigo text-white hover:shadow-neon-indigo'
                }`}
              >
                {isUploadingDataset ? `Importing ${datasetFiles.length} Files...` : `Import ${datasetFiles.length} Dataset File(s)`}
              </button>
            </div>
          </div>
        </form>
      )}
    </div>
  );
};

export default ProfileManager;
