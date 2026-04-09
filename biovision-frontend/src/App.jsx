import React, { useState } from 'react';

export default function BioVisionApp() {
  const [selectedModel, setSelectedModel] = useState('pneumonia');
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const models = {
    pneumonia: { name: 'Pneumonia Detection', endpoint: 'pneumonia' },
    brain_tumor: { name: 'Brain Tumor Analysis', endpoint: 'brain_tumor' },
    dr: { name: 'Diabetic Retinopathy', endpoint: 'dr' },
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      setImagePreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handlePredict = async () => {
    if (!imageFile) {
      setError("Please select an image first.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', imageFile);

    try {
      const response = await fetch(`http://127.0.0.1:8000/predict/${models[selectedModel].endpoint}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.statusText}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError("Failed to connect to the BioVision API. Ensure the backend is running.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 font-sans selection:bg-teal-500 selection:text-white">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-md sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <h1 className="text-2xl font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-blue-500">
            BioVision AI
          </h1>
          <div className="text-sm font-medium text-slate-400">Diagnostic Interface v1.0</div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-12 grid grid-cols-1 lg:grid-cols-2 gap-12">
        {/* Left Column: Input Panel */}
        <div className="space-y-8">
          
          {/* Model Selector */}
          <div className="bg-slate-800 rounded-2xl p-6 border border-slate-700 shadow-xl">
            <h2 className="text-lg font-semibold text-slate-200 mb-4">1. Select Diagnostic Model</h2>
            <div className="flex flex-wrap gap-3">
              {Object.entries(models).map(([key, model]) => (
                <button
                  key={key}
                  onClick={() => setSelectedModel(key)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                    selectedModel === key
                      ? 'bg-teal-500 text-white shadow-[0_0_15px_rgba(20,184,166,0.4)]'
                      : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                  }`}
                >
                  {model.name}
                </button>
              ))}
            </div>
          </div>

          {/* Image Upload */}
          <div className="bg-slate-800 rounded-2xl p-6 border border-slate-700 shadow-xl">
            <h2 className="text-lg font-semibold text-slate-200 mb-4">2. Upload Medical Scan</h2>
            <div className="relative border-2 border-dashed border-slate-600 rounded-xl p-8 text-center hover:border-teal-500 transition-colors duration-200 bg-slate-800/50">
              <input
                type="file"
                accept="image/*"
                onChange={handleImageChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              {imagePreview ? (
                <img src={imagePreview} alt="Scan preview" className="mx-auto max-h-64 rounded-lg shadow-md object-contain" />
              ) : (
                <div className="space-y-4">
                  <div className="w-16 h-16 rounded-full bg-slate-700 flex items-center justify-center mx-auto text-teal-400">
                    <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"></path></svg>
                  </div>
                  <p className="text-slate-300 font-medium">Drag & drop your scan here</p>
                  <p className="text-xs text-slate-500">Supports JPG, PNG, DICOM (converted)</p>
                </div>
              )}
            </div>

            <button
              onClick={handlePredict}
              disabled={loading || !imageFile}
              className={`w-full mt-6 py-3 rounded-xl font-bold text-lg transition-all duration-300 ${
                loading || !imageFile
                  ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
                  : 'bg-gradient-to-r from-teal-500 to-blue-600 text-white hover:shadow-[0_0_20px_rgba(20,184,166,0.5)] transform hover:-translate-y-0.5'
              }`}
            >
              {loading ? 'Running AI Analysis...' : 'Run Diagnostics'}
            </button>
            
            {error && <p className="mt-4 text-red-400 text-sm text-center bg-red-900/20 py-2 rounded-lg">{error}</p>}
          </div>
        </div>

        {/* Right Column: Results Panel */}
        <div className="bg-slate-800 rounded-2xl p-6 border border-slate-700 shadow-xl flex flex-col">
          <h2 className="text-lg font-semibold text-slate-200 mb-6 border-b border-slate-700 pb-4">Diagnostic Results</h2>
          
          {loading ? (
            <div className="flex-1 flex flex-col items-center justify-center space-y-4">
              <div className="w-12 h-12 border-4 border-slate-600 border-t-teal-500 rounded-full animate-spin"></div>
              <p className="text-teal-400 animate-pulse font-medium">Processing scan via {models[selectedModel].name} model...</p>
            </div>
          ) : result ? (
            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
              {/* Primary Result */}
              <div className="bg-slate-900 rounded-xl p-6 border border-slate-700">
                <p className="text-sm text-slate-400 uppercase tracking-wider mb-1">Primary Finding</p>
                <h3 className="text-3xl font-bold text-white mb-2">{result.class}</h3>
                <div className="flex items-center space-x-4">
                  <div className="flex-1 h-3 bg-slate-800 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-teal-500 rounded-full transition-all duration-1000"
                      style={{ width: `${(result.confidence * 100).toFixed(1)}%` }}
                    ></div>
                  </div>
                  <span className="text-teal-400 font-bold">{(result.confidence * 100).toFixed(2)}% Confidence</span>
                </div>
              </div>

              {/* Detailed Probabilities */}
              <div>
                <h4 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">Detailed Probability Breakdown</h4>
                <div className="space-y-3">
                  {Object.entries(result.all_probabilities)
                    .sort(([,a], [,b]) => b - a)
                    .map(([className, prob]) => (
                    <div key={className} className="flex items-center justify-between text-sm bg-slate-800/50 p-3 rounded-lg border border-slate-700/50">
                      <span className="text-slate-300">{className}</span>
                      <span className="font-mono text-slate-400">{(prob * 100).toFixed(2)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center text-slate-500 space-y-4">
              <svg className="w-16 h-16 opacity-20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
              <p>Upload a scan and run diagnostics to see results.</p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
