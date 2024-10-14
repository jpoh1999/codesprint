// server.js
const express = require('express');
const cors = require('cors'); // Import the cors package
const fileRoutes = require('./routes/fileRoutes');

const app = express();
const port = 8080;

// Middleware to parse JSON
app.use(express.json());

// Enable CORS for all routes
app.use(cors()); // Use the cors middleware

// File handling routes
app.use('/api/files', fileRoutes);

// Server start
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
