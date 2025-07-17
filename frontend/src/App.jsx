// frontend/src/App.jsx
import { useState } from 'react';

export default function App() {
  const [prompt, setPrompt] = useState('');
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [videoFile, setVideoFile] = useState(null);
  const [inputType, setInputType] = useState('url'); // 'url' or 'upload'
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [gifs, setGifs] = useState([]);

  const handleFileChange = (e) => {
    setVideoFile(e.target.files[0]);
    setYoutubeUrl(''); // Clear the other input
  };

  const handleUrlChange = (e) => {
    setYoutubeUrl(e.target.value);
    setVideoFile(null); // Clear the other input
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setGifs([]);

    try {
      let response;
      if (inputType === 'url') {
        response = await fetch('http://127.0.0.1:5000/api/generate-gifs', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt: prompt,
            youtube_url: youtubeUrl,
          }),
        });
      } else {
        const formData = new FormData();
        formData.append('prompt', prompt);
        formData.append('video', videoFile);

        response = await fetch('http://127.0.0.1:5000/api/generate-gifs-from-upload', {
          method: 'POST',
          body: formData,
        });
      }

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Something went wrong on the server.');
      }
      
      setGifs(data.gifs);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownload = async (gifUrl, filename) => {
    try {
      const response = await fetch(gifUrl);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Download failed:', error);
      window.open(gifUrl, '_blank');
    }
  };

  return (
    <div className="app-container">
      
      <header className="header">
        <div className="header-content">
          <h1 className="main-title gradient-text">
            AI Video to GIF
          </h1>
          <p className="subtitle">
            Transform videos into perfect GIFs with AI powered captions ✨
          </p>
          <div className="features">
            <span className="feature-item">
              <div className="status-dot status-green"></div>
              AI Powered
            </span>
            <span className="feature-item">
              <div className="status-dot status-blue"></div>
              Smart Captions
            </span>
            <span className="feature-item">
              <div className="status-dot status-purple"></div>
              Instant Results
            </span>
          </div>
        </div>
      </header>

      <main className="main-content">
        
        <div className="form-container glass">
          <form onSubmit={handleSubmit} className="form">
            
            <div className="input-group">
              <label className="input-label">
                🎯 What moments or text do you want?
              </label>
              <input
                type="text"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                className="input-field"
                placeholder="funny moments, epic fails, quotes..."
                required
              />
            </div>

            <div className="input-group">
              <div className="tab-switcher">
                <button 
                  type="button" 
                  className={`tab-button ${inputType === 'url' ? 'active' : ''}`}
                  onClick={() => setInputType('url')}
                >
                  🎬 YouTube URL
                </button>
                <button 
                  type="button" 
                  className={`tab-button ${inputType === 'upload' ? 'active' : ''}`}
                  onClick={() => setInputType('upload')}
                >
                  💻 Upload Video
                </button>
              </div>

              {inputType === 'url' ? (
                <input
                  type="url"
                  value={youtubeUrl}
                  onChange={handleUrlChange}
                  className="input-field"
                  placeholder="http://youtube.com/watch?v=..."
                  required={inputType === 'url'}
                />
              ) : (
                <div className="file-input-container">
                  <input
                    type="file"
                    id="video-upload"
                    className="file-input"
                    onChange={handleFileChange}
                    accept="video/mp4,video/mov,video/webm"
                    required={inputType === 'upload'}
                  />
                  <label htmlFor="video-upload" className="file-input-label">
                    {videoFile ? `Selected: ${videoFile.name}` : '📂 Choose a video file...'}
                  </label>
                </div>
              )}
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="submit-button"
            >
              {isLoading ? (
                <div className="button-loading">
                  <div className="spinner"></div>
                  Creating your GIFs...
                </div>
              ) : (
                '🚀 Create GIFs'
              )}
            </button>
          </form>
        </div>

        {error && (
          <div className="error-container">
            <div className="error-box">
              <div className="error-content">
                <span>⚠️</span>
                <span>{error}</span>
              </div>
            </div>
          </div>
        )}

        {isLoading && (
          <div className="loading-section">
            <div className="loading-card glass-card">
              <div className="loading-spinner"></div>
              <h3 className="loading-title">Creating Magic...</h3>
              <p className="loading-text">Analyzing video and generating GIFs</p>
            </div>
          </div>
        )}

        {gifs.length > 0 && (
          <div className="results-section animate-fade-up">
            <div className="results-header">
              <h2 className="results-title gradient-text-green">
              Your GIFs are Ready!
              </h2>
              <p className="results-subtitle">
                {gifs.length} amazing GIFs created just for you
              </p>
            </div>
            
            <div className="gif-grid">
              {gifs.map((gif, index) => (
                <div 
                  key={index} 
                  className="gif-card glass-card"
                >
                  <div className="gif-image-container">
                    {/* --- FIX: Add a unique timestamp to the URL to force reload --- */}
                    <img 
                      src={`${gif}?t=${new Date().getTime()}`} 
                      alt={`GIF ${index + 1}`} 
                      className="gif-image"
                    />
                    {/* ----------------------------------------------------------------- */}
                    <div className="gif-number">
                      #{index + 1}
                    </div>
                  </div>
                  
                  <div className="gif-content">
                    <h3 className="gif-title">AI Generated GIF</h3>
                    <p className="gif-description">Perfect moment with smart captions</p>
                    <button
                      onClick={() => handleDownload(gif, `gif-${index + 1}.gif`)}
                      className="download-button"
                    >
                      📥 Download GIF
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>

    </div>
  );
}