import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom'; 
import './LandingPage.css';
import './VisualSelectorTransition.css'; // Import transition CSS
import videoSrc from '../../assets/background_landing.mp4'; 

const LandingPage = () => {
  /**
   * React states
   */
  const [zoomLevel, setZoomLevel] = useState(1); 
  const [showVisualSelector, setShowVisualSelector] = useState(false); // State to control VisualSelector visibility
  
  /**
   * Helper functions to handle events
   */
  const navigate = useNavigate(); 
  const handleScroll = (event) => {
    const deltaY = event.deltaY;
    const zoomChange = deltaY > 0 ? 0.1 : -0.1; 

    setZoomLevel((prevZoomLevel) => {
      const newZoomLevel = Math.min(Math.max(prevZoomLevel + zoomChange, 1), 3);

      // Trigger the VisualSelector transition when zoomLevel reaches 3
      if (newZoomLevel === 3) {
        setShowVisualSelector(true);
        setTimeout(() => navigate('/visualselector'), 500); // Delay navigation to allow transition
      }

      return newZoomLevel;
    });
  };

  useEffect(() => {
    window.addEventListener('wheel', handleScroll);
    return () => {
      window.removeEventListener('wheel', handleScroll);
    };
  }, []);

  return (
    <div className="landing-page"
      style={{
        transition: 'transform 0.5s ease', // Apply transition to the landing page
        transform: `scale(${zoomLevel})`, 
        opacity: zoomLevel === 1 ? 1 : 1 - (zoomLevel - 1) / 2, 
      }}
    >
      <video 
        autoPlay 
        loop 
        muted 
        className="background-video"
        style={{
          transform: `scale(${zoomLevel})`,
        }}
      >
        <source src={videoSrc} type="video/mp4" />
        Your browser does not support HTML5 video.
      </video>
      <div
        className="banner"
        style={{
          opacity: zoomLevel === 1 ? 0.7 : 0.7 - (zoomLevel - 1) / 2,
          zIndex: 2,
        }}
      >
        <h1>CodeSprint 2024 PSA Solution</h1>
        <p>Scroll to zoom in and out. Zoom in enough to enter.</p>
      </div>

      {/* Conditional rendering of the VisualSelector */}
      {showVisualSelector && (
        <div className="visual-selector-enter visual-selector-enter-active">
          {/* Your VisualSelector content goes here */}
          <h2>Visual Selector Loading...</h2> {/* Placeholder content */}
        </div>
      )}
    </div>
  );
};

export default LandingPage;
