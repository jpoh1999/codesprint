// routes/fileRoutes.js
const express = require('express');
const multer = require('multer');
const { uploadHandler, downloadHandler, deleteHandler } = require('../controllers/fileController');

const router = express.Router();
const upload = multer({ storage: multer.memoryStorage() });

// Route to upload a file
router.post('/upload', upload.single('file'), uploadHandler);

// Route to download a file
router.get('/download/:filename', downloadHandler);

// Route to delete a file
router.delete('/delete/:filename', deleteHandler);

module.exports = router;
