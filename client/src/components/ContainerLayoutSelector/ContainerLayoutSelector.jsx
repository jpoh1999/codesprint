import React, { useState, useEffect } from 'react';
import Papa from 'papaparse';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import './ContainerLayoutSelector.css'; // Ensure this CSS file exists
import ContainerLayoutVisualizer from '../ContainerLayoutVisualizer/ContainerLayoutVisualizer';

const ContainerLayoutSelector = () => {
  const [layoutFile, setLayoutFile] = useState(null);
  const [movesFile, setMovesFile] = useState(null);
  const [currentPage, setCurrentPage] = useState('selection');
  const [layoutData, setLayoutData] = useState(null);
  const [movesData, setMovesData] = useState(null);
  const [error, setError] = useState(null); // For error handling
  const [isDarkMode, setIsDarkMode] = useState(false); // State for dark mode

  const navigate = useNavigate();

  const sendLayoutFile = async (file) => {
    try {
      // Send file to the server
      const formData = new FormData();
      formData.append('file', file);
      
      // Make POST request to the API
      const response = await axios.post('http://localhost:8080/api/files/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      alert('Layout file uploaded successfully!');
    } catch (err) {
      console.error('Error uploading layout file:', err);
      alert('Failed to upload layout file.');
    }
  };

  const sendMoveFile = async (file) => {
    try {
      // Send file to the server
      const formData = new FormData();
      formData.append('file', file);
      
      // Make POST request to the API
      const response = await axios.post('http://localhost:8080/api/files/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      alert('Moves file uploaded successfully!');
    } catch (err) {
      console.error('Error uploading moves file:', err);
      alert('Failed to upload moves file.');
    }
  };

  const handleLayoutChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setLayoutFile(file);
      sendLayoutFile(file);
      parseLayoutFile(file);
    }
  };

  const handleMovesChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setMovesFile(file);
      sendMoveFile(file);
      parseMovesFile(file);
    }
  };

  const parseLayoutFile = (file) => {
    Papa.parse(file, {
      complete: (results) => {
        if (results.data.length > 0) {
          setLayoutData(results.data); // Set layout data to the parsed data
          setError(null); // Reset error state
          console.log('Parsed Layout Data:', results.data); // Log parsed data for verification
        } else {
          setError('Layout file is empty or invalid.');
        }
      },
      error: (error) => {
        setError(`Error parsing layout file: ${error.message}`);
      },
      header: false,
      skipEmptyLines: true,
    });
  };

  const parseMovesFile = (file) => {
    const reader = new FileReader();
    reader.onload = (event) => {
      const movesData = event.target.result.trim().split('\n').slice(1);
      console.log('Raw Moves Data:', movesData);

      const moves = movesData
        .map(row => {
          const move = row.split(',').map(cell => cell.trim());
          if (move.length === 6) {
            return move.map(Number);
          }
          return null; // Ignore invalid rows
        })
        .filter(Boolean);

      console.log('Parsed Moves:', moves);
      if (moves.length > 0) {
        setMovesData(moves);
        setError(null); // Reset error state
      } else {
        setError('No valid moves found in the moves file.');
      }
    };
    reader.onerror = () => {
      setError('Error reading moves file.');
    };
    reader.readAsText(file);
  };

  const handlePlaySolution = () => {
    if (layoutData && movesData) {
      setCurrentPage('solution');
    } else {
      setError('Please select both layout and moves files.');
    }
  };

  const handleBackToSelection = () => {
    setCurrentPage('selection');
    setLayoutData(null);
    setMovesData(null);
    setError(null); // Reset error state
  };

  const handleBackToLandingPage = () => {
    navigate('/'); // Navigate to the landing page
  };

  const toggleTheme = () => {
    setIsDarkMode(prevMode => !prevMode);
  };

  return (
    <div className={`container ${isDarkMode ? 'dark-mode' : 'light-mode'}`}>
      <div className="theme-switcher">
        <label className="switch">
          <input type="checkbox" checked={isDarkMode} onChange={toggleTheme} />
          <span className="slider"></span>
        </label>
      </div>

      {error && <div className="error-message">{error}</div>} {/* Display error messages */}
      {currentPage === 'selection' ? (
        <div className="selection-box">
          <h2>Select Container Layout with Solution</h2>
          <div className="input-container">
            <input
              type="file"
              accept=".csv"
              onChange={handleLayoutChange}
              className="file-input"
            />
            <input
              type="file"
              accept=".txt"
              onChange={handleMovesChange}
              className="file-input"
            />
          </div>
          <div className="buttons-div">
            <button className="play-button" onClick={handlePlaySolution}>
              Play Solution
            </button>
            <button className="landing-button" onClick={handleBackToLandingPage}>Back </button> {/* Back to Landing Page Button */}
          </div>
          
        </div>
      ) : (
        <div className="solution-page">
          <h2>Solution Page</h2>
          {layoutData && movesData && (
            <ContainerLayoutVisualizer layoutData={layoutData} movesData={movesData} />
          )}
          <button className="back-button" onClick={handleBackToSelection}>Back to Selection</button>
          
        </div>
      )}
    </div>
  );
};

export default ContainerLayoutSelector;
