import React, { useState } from 'react';
import { motion } from 'framer-motion';
import './ContainerLayoutVisualizer.css'; // Import the CSS file

// Function to generate a color based on a string
const stringToColor = (str) => {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }
  return `hsl(${hash % 360}, 70%, 70%)`; // Use HSL for better distribution of colors
};

const ContainerLayoutVisualizer = ({ layoutData, movesData }) => {
  // Create a deep copy of original layout to avoid mutation
  const [originalLayout] = useState(JSON.parse(JSON.stringify(layoutData)) || []); 
  const [layout, setLayout] = useState(JSON.parse(JSON.stringify(originalLayout)));
  const [animationQueue] = useState(movesData || []); // Set initial moves data
  const [cranePosition, setCranePosition] = useState({ x: 0, y: 0 }); // Initial crane position
  const [craneArmLength, setCraneArmLength] = useState(0); // Initial crane arm length
  const [isAnimating, setIsAnimating] = useState(false); // State to track animation status
  const [highlightedCells, setHighlightedCells] = useState([]); // State to track highlighted cells
  const [currentScore, setCurrentScore] = useState(0); // State for the current score

  // Ensure layout is a valid 2D array
  if (!Array.isArray(layout) || !layout.every(Array.isArray)) {
    return <div>Error: Invalid layout data</div>; // Error message for invalid data
  }

  // Lerp function for smooth interpolation
const lerp = (start, end, t) => {
  return start * (1 - t) + end * t;
};

/**
 * Animate the moves from the solution data with smooth transitions using lerp
 */
const animateMoves = () => {
  const newLayout = JSON.parse(JSON.stringify(originalLayout)); // Copy the current layout
  setLayout(newLayout);

  const moveDuration = 1000; // Total time for each move (in milliseconds)
  const frameRate = 60; // Approximate frames per second
  const frameTime = 1000 / frameRate; // Time per frame

  let currentCraneX = null; // To store the current crane X position

  const performMove = (currentMoveIndex) => {
    if (currentMoveIndex >= animationQueue.length) {
      setIsAnimating(false); // Stop animating when all moves are complete
      return;
    }

    // Destructure the data for each row, including the score
    const [slot, fm_row, fm_level, to_row, to_level, score] = animationQueue[currentMoveIndex].map(Number);

    // Update the current score
    setCurrentScore(score);

    // Position the crane at the start (pick-up position)
    const startX = fm_row * (100 / 11) + 2; // Start X based on row index
    const endX = to_row * (100 / 11) + 2;   // End X (target row)

    const startY = 0; // Crane starts at the top
    const pickUpArmLength = (layout.length - fm_level) * (1000 / 10); // Crane arm length for picking up
    const dropOffArmLength = (layout.length - to_level) * (1000 / 10); // Arm length for dropping off
    const initialArmLength = 0; // Arm length when fully retracted

    let startTime = null;

    // Function to animate each frame
    const animateFrame = (timestamp, phase) => {
      if (!startTime) startTime = timestamp;
      const elapsedTime = timestamp - startTime;

      const t = Math.min(elapsedTime / moveDuration, 1); // Normalize time to range [0, 1]

      let currentArmLength;

      if (phase === 'moveToPickup') {
        // Move to pickup position (just X position)
        currentCraneX = lerp(startX, startX, t); // Crane stays at the pickup position

        // Arm length extends down to pickup the container
        currentArmLength = lerp(initialArmLength, pickUpArmLength, t);

        if (t < 1) {
          // Continue moving to pickup position
          requestAnimationFrame((ts) => animateFrame(ts, 'moveToPickup'));
        } else {
          // After reaching the position, retract the arm
          startTime = null; // Reset time for the next phase
          requestAnimationFrame((ts) => animateFrame(ts, 'retractPickup'));
        }
      } else if (phase === 'retractPickup') {
        // Retract the arm after picking up
        currentArmLength = lerp(pickUpArmLength, initialArmLength, t);

        if (t < 1) {
          // Continue retracting the arm
          requestAnimationFrame((ts) => animateFrame(ts, 'retractPickup'));
        } else {
          // Move horizontally to the target position
          startTime = null; // Reset time for horizontal movement
          requestAnimationFrame((ts) => animateFrame(ts, 'moveToDrop'));
        }
      } else if (phase === 'moveToDrop') {
        // Move horizontally to the drop position
        currentCraneX = lerp(currentCraneX, endX, t); // Update crane X position while keeping the arm retracted
        currentArmLength = initialArmLength; // Keep arm retracted while moving

        if (t < 1) {
          // Continue horizontal movement animation
          requestAnimationFrame((ts) => animateFrame(ts, 'moveToDrop'));
        } else {
          // After reaching the target, extend the arm downwards
          startTime = null; // Reset time for arm extension
          requestAnimationFrame((ts) => animateFrame(ts, 'extendToDrop'));
        }
      } else if (phase === 'extendToDrop') {
        // Extend the arm to drop the container
        currentArmLength = lerp(initialArmLength, dropOffArmLength, t);

        if (t < 1) {
          // Continue arm extension animation
          requestAnimationFrame((ts) => animateFrame(ts, 'extendToDrop'));
        } else {
          // Place the container in the new position immediately after extending
          const movedBox = newLayout[newLayout.length - fm_level][fm_row];
          newLayout[newLayout.length - fm_level][fm_row] = ''; // Remove the box from the old position
          newLayout[newLayout.length - to_level][to_row] = movedBox; // Place it in the new position
          setLayout(newLayout); // Update the layout

          // Retract the arm again after dropping the container
          startTime = null; // Reset time for retraction
          requestAnimationFrame((ts) => animateFrame(ts, 'retractAfterDrop'));
        }
      } else if (phase === 'retractAfterDrop') {
        // Retract the arm after dropping off
        currentArmLength = lerp(dropOffArmLength, initialArmLength, t);

        if (t < 1) {
          // Continue arm retraction animation
          requestAnimationFrame((ts) => animateFrame(ts, 'retractAfterDrop'));
        } else {
          // Move to the next move after retraction
          performMove(currentMoveIndex + 1); // Continue to the next move
        }
      }

      // Update the crane position and arm length state
      setCranePosition({ x: currentCraneX !== null ? currentCraneX : startX, y: startY });
      setCraneArmLength(currentArmLength);
    };

    // Start by moving to the pickup position
    requestAnimationFrame((ts) => animateFrame(ts, 'moveToPickup'));
  };

  performMove(0); // Start the recursive moving process
};
  /**
   * Listen for click event and handle it accordingly
   */
  const startAnimation = () => {
    // resetLayout(); // Reset the layout every time the animation starts
    if (!isAnimating && animationQueue.length > 0) {
      setIsAnimating(true);
      setCranePosition({ x: 0, y: 0 }); // Reset crane to top-left before starting
      setCraneArmLength(200); // Reset crane arm length
      animateMoves();
    } 
    
  };

  /**
   * Listen for resetLayout click and handle it accordingly
   */
  const resetLayout = () => {
    setLayout(JSON.parse(JSON.stringify(originalLayout))); // Reset layout to the original
    setHighlightedCells([]); // Optionally reset highlighted cells
    setCurrentScore(0); // Reset score when layout is reset
    // Ensure the animation state is not active after reset
    setIsAnimating(false);  // Set animation status to false
    
    // Reset crane position and arm length if necessary
    setCranePosition({ x: 0, y: 0 });
    setCraneArmLength(200);

  };

  /**
   * Highlight the targeted cell with a predefined yellow border
   * 
   * @param {int} rowIndex 
   * @param {int} cellIndex 
   */
  const toggleHighlight = (rowIndex, cellIndex) => {
    const cellKey = `${rowIndex}-${cellIndex}`; // Create a unique key for each cell
    if (highlightedCells.includes(cellKey)) {
      // Remove cell from highlighted if already highlighted
      setHighlightedCells(highlightedCells.filter((cell) => cell !== cellKey));
    } else {
      // Add cell to highlighted
      setHighlightedCells([...highlightedCells, cellKey]);
    }
  };

  return (
    <div className="container">
      <h1 className="title">Container Layout Visualizer</h1>

      {/* Button Elements */}
      <div className="button-container">
        <button
          onClick={startAnimation}
          disabled={isAnimating}
          className={`button button-start`}
        >
          Start Animation
        </button>
        <button
          onClick={resetLayout}
          disabled={isAnimating}
          className={`button button-reset`}
        >
          Reset Layout
        </button>
      </div>
      {/* Current Score Element */}
      <div className="current-score">
          <strong>Current Score: </strong> {currentScore}
      </div>

      {/* Crane row */}
      <div className="crane-row">
        <motion.div
          className="crane"
          style={{ left: `${cranePosition.x}%`, top: `${cranePosition.y}px` }}
          transition={{ duration: 0.5 }} // Add transition for smoother horizontal movement
        >
          <div className="crane-arm" style={{ height: `${craneArmLength}px` }}></div> {/* Dynamic height based on craneArmLength */}
          <div className="hook"></div> {/* Hook position */}
        </motion.div>
      </div>

      {/* Container layout */}
      <div className="layout">
        {layout.map((row, rowIndex) => (
          <div key={rowIndex} className="row">
            {Array.isArray(row) && row.map((cell, cellIndex) => {
              const cellKey = `${rowIndex}-${cellIndex}`;
              const isHighlighted = highlightedCells.includes(cellKey);

              return (
                <motion.div
                  key={cellIndex}
                  className={`cell ${isHighlighted ? 'highlighted' : ''}`} // Apply 'highlighted' class if true
                  style={{
                    backgroundColor: cell ? stringToColor(cell) : '#fff',
                  }}
                  onClick={() => toggleHighlight(rowIndex, cellIndex)} // Toggle highlight on click
                >
                  {cell}
                </motion.div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
};

export default ContainerLayoutVisualizer;
