// controllers/fileController.js
const { uploadFile, downloadFile, deleteFile } = require('../utils/gcsUploader');

/**
 * Handle file upload.
 * @param {Object} req - Express request object.
 * @param {Object} res - Express response object.
 */
const uploadHandler = async (req, res) => {
  if (!req.file) {
    return res.status(400).send('No file uploaded.');
  }

  try {
    const message = await uploadFile(req.file.buffer, req.file.originalname);
    res.status(200).send(message);
  } catch (err) {
    res.status(500).send('Error uploading file: ' + err.message);
  }
};

/**
 * Handle file download.
 * @param {Object} req - Express request object.
 * @param {Object} res - Express response object.
 */
const downloadHandler = async (req, res) => {
  const filename = req.params.filename;

  try {
    const [fileBuffer] = await downloadFile(filename);
    res.status(200).send(fileBuffer);
  } catch (err) {
    res.status(404).send('File not found: ' + err.message);
  }
};

/**
 * Handle file deletion.
 * @param {Object} req - Express request object.
 * @param {Object} res - Express response object.
 */
const deleteHandler = async (req, res) => {
  const filename = req.params.filename;

  try {
    await deleteFile(filename);
    res.status(200).send(`File ${filename} deleted successfully.`);
  } catch (err) {
    res.status(500).send('Error deleting file: ' + err.message);
  }
};

module.exports = {
  uploadHandler,
  downloadHandler,
  deleteHandler
};
