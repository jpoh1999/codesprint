// utils/gcsUploader.js
const { Storage } = require('@google-cloud/storage');
const storage = new Storage({
  keyFilename: '../cloud/secret.json', 
});
const bucketName = 'codesprint-data-lake';


/**
 * Upload a file to Google Cloud Storage.
 * @param {Buffer} fileBuffer - The file's buffer to upload.
 * @param {string} filename - The name of the file to be saved as.
 * @returns {Promise} Resolves on successful upload, rejects on error.
 */
const uploadFile = (fileBuffer, filename) => {
  return new Promise((resolve, reject) => {
    const blob = storage.bucket(bucketName).file(filename);
    const blobStream = blob.createWriteStream();

    blobStream.on('error', err => {
      reject(err);
    });

    blobStream.on('finish', () => {
      resolve(`File ${filename} uploaded successfully.`);
    });

    blobStream.end(fileBuffer);
  });
};

/**
 * Download a file from Google Cloud Storage.
 * @param {string} filename - The name of the file to download.
 * @returns {Promise} Resolves with the file buffer, rejects on error.
 */
const downloadFile = (filename) => {
  return storage.bucket(bucketName).file(filename).download();
};

/**
 * Delete a file from Google Cloud Storage.
 * @param {string} filename - The name of the file to delete.
 * @returns {Promise} Resolves on successful deletion, rejects on error.
 */
const deleteFile = (filename) => {
  return storage.bucket(bucketName).file(filename).delete();
};

module.exports = {
  uploadFile,
  downloadFile,
  deleteFile
};
