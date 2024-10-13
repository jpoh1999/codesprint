import React from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import LandingPage from './components/LandingPage/LandingPage';
import ContainerLayoutSelector from './components/ContainerLayoutSelector/ContainerLayoutSelector';

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/visualselector" element={<ContainerLayoutSelector />} />
        <Route path="*" element={<Navigate to="/" />} /> {/* Redirects any unknown route to the landing page */}
      </Routes>
    </Router>
  );
};

export default App;
