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

const storage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, uploadDir),
    filename: (req, file, cb) => cb(null, `${Date.now()}_${file.originalname}`)
});
const upload = multer({ storage: storage });

app.use(express.static(path.join(__dirname, 'public')));
app.use(express.urlencoded({ limit: '100mb', extended: true }));
app.use(express.json({ limit: '100mb' }));

// Static HTML Page Routes
app.get('/', (req, res) => res.sendFile(path.join(__dirname, 'public', 'index.html')));
app.get('/live_gait_analysis', (req, res) => res.sendFile(path.join(__dirname, 'public', 'live_gait_analysis.html')));
app.get('/live_posture_detection', (req, res) => res.sendFile(path.join(__dirname, 'public', 'live_posture_detection.html')));
app.get('/uploadable_long_term_gait', (req, res) => res.sendFile(path.join(__dirname, 'public', 'uploadable_long_term_gait.html')));

// API Endpoint: Process Per-Frame Keypoints for Live HUD Stats
app.post('/api/analyze_frame', (req, res) => {
    const pyProcess = spawn('python', ['python/session_analyze.py']);
    let resultData = '';

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

// API Endpoint: Live Session Stop -> Save CSV -> Process Metrics -> LLM -> PDF Download
app.post('/api/analyze_session', (req, res) => {
    const landmarks = req.body.landmarks;
    if (!landmarks || landmarks.length === 0) {
        return res.status(400).json({ error: 'No landmark data recorded.' });
    }

    const timestamp = Date.now();
    const csvFileName = `live_session_${timestamp}.csv`;
    const csvFilePath = path.join(uploadDir, csvFileName);
    const pdfFileName = `NeuroStratum_Report_${timestamp}.pdf`;
    const pdfFilePath = path.join(__dirname, pdfFileName);

    let csvLines = ['frame,timestamp_ms,landmark_index,x,y,z,visibility'];
    landmarks.forEach((frameObj) => {
        if (frameObj.keypoints && Array.isArray(frameObj.keypoints)) {
            frameObj.keypoints.forEach((lm, idx) => {
                csvLines.push(`${frameObj.frame},${frameObj.timestamp_ms},${idx},${lm.x},${lm.y},${lm.z || 0.0},${lm.visibility || 1.0}`);
            });
        }
    });

    fs.writeFileSync(csvFilePath, csvLines.join('\n'));

    const pyProcess = spawn('python', ['python/gait_pdf_generator.py', csvFilePath, pdfFilePath]);

    pyProcess.on('close', (code) => {
        if (code === 0 && fs.existsSync(pdfFilePath)) {
            res.download(pdfFilePath, pdfFileName, (err) => {
                if (fs.existsSync(pdfFilePath)) fs.unlinkSync(pdfFilePath);
            });
        } else {
            res.status(500).json({ error: 'Failed to generate PDF report.' });
        }
    });
});

// Route: Upload Pre-Recorded CSV File -> Generate PDF
app.post('/uploadable_long_term_gait', upload.single('gait_file'), (req, res) => {
    if (!req.file) return res.status(400).send('No CSV file uploaded.');

    const filePath = req.file.path;
    const uniquePdfName = `NeuroStratum_Report_${Date.now()}.pdf`;
    const pdfPath = path.join(__dirname, uniquePdfName);

    const pyProcess = spawn('python', ['python/gait_pdf_generator.py', filePath, pdfPath]);

    pyProcess.on('close', (code) => {
        if (code === 0 && fs.existsSync(pdfPath)) {
            res.download(pdfPath, uniquePdfName, (err) => {
                if (fs.existsSync(pdfPath)) fs.unlinkSync(pdfPath);
            });
        } else {
            res.status(500).send('Error generating PDF report.');
        }
    });
});

app.listen(PORT, () => console.log(`Server running at http://localhost:${PORT}`));