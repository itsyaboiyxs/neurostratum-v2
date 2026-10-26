const express = require('express');
const path = require('path');
const multer = require('multer');
const fs = require('fs');
const { spawn } = require('child_process');

const app = express();
const PORT = 3000;

const uploadDir = path.join(__dirname, 'uploaded_data');
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir, { recursive: true });
}

// Multer using originalname
const storage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, uploadDir),
    filename: (req, file, cb) => cb(null, `${Date.now()}_${file.originalname}`)
});
const upload = multer({ storage: storage });

app.use(express.static(path.join(__dirname, 'public')));
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

// Routes
app.get('/', (req, res) => res.sendFile(path.join(__dirname, 'public', 'index.html')));
app.get('/live_gait_analysis', (req, res) => res.sendFile(path.join(__dirname, 'public', 'live_gait_analysis.html')));
app.get('/live_posture_detection', (req, res) => res.sendFile(path.join(__dirname, 'public', 'live_posture_detection.html')));
app.get('/uploadable_long_term_gait', (req, res) => res.sendFile(path.join(__dirname, 'public', 'uploadable_long_term_gait.html')));

// Live Video Frame Analysis API (uses python session_analyze.py)
app.post('/api/analyze_frame', (req, res) => {
    const pyProcess = spawn('python', ['python/session_analyze.py']); // Windows rule: use 'python'
    let resultData = '';

    // For live frame, we can write a temporary payload or call engine directly
    pyProcess.stdin.write(JSON.stringify(req.body));
    pyProcess.stdin.end();

    pyProcess.stdout.on('data', (data) => { resultData += data.toString(); });

    pyProcess.on('close', () => {
        try {
            res.json(JSON.parse(resultData));
        } catch (e) {
            res.status(500).json({ error: 'Failed to process biomechanics frame.' });
        }
    });
});

// CSV Upload Route for PDF Generation (Fixed Race Condition with Unique Filename)
app.post('/uploadable_long_term_gait', upload.single('gait_file'), (req, res) => {
    if (!req.file) return res.status(400).send('No CSV file uploaded.');

    const filePath = req.file.path;
    const uniquePdfName = `NeuroStratum_Report_${Date.now()}.pdf`;
    const pdfPath = path.join(__dirname, uniquePdfName);

    const pyProcess = spawn('python', ['python/gait_pdf_generator.py', filePath, pdfPath]);

    pyProcess.on('close', (code) => {
        if (code === 0 && fs.existsSync(pdfPath)) {
            res.download(pdfPath, uniquePdfName, (err) => {
                if (fs.existsSync(pdfPath)) fs.unlinkSync(pdfPath); // Cleanup after download
            });
        } else {
            res.status(500).send('Error generating PDF report.');
        }
    });
});

app.listen(PORT, () => console.log(`Server running at http://localhost:${PORT}`));