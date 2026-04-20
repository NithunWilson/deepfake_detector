/**
 * Main JavaScript for Deepfake Detection System
 */

$(document).ready(function() {
    // DOM Elements
    const uploadArea = $('#uploadArea');
    const videoInput = $('#videoInput');
    const fileInfo = $('#fileInfo');
    const videoPreview = $('#videoPreview');
    const previewVideo = $('#previewVideo');
    const fileName = $('#fileName');
    const fileSize = $('#fileSize');
    const removeFileBtn = $('#removeFile');
    const analyzeBtn = $('#analyzeBtn');
    const testSampleBtn = $('#testSampleBtn');
    const sequenceSlider = $('#sequenceLength');
    const sliderValue = $('#sliderValue');
    const progressBar = $('#progressBar .progress-bar');
    const progressText = $('#progressText');
    const progressPercent = $('#progressPercent');
    const resultsSection = $('#results-section');
    const resultsHeader = $('#resultsHeader');
    const predictionText = $('#predictionText');
    const confidenceScore = $('#confidenceScore');
    const realProbability = $('#realProbability');
    const fakeProbability = $('#fakeProbability');
    const realBar = $('#realBar');
    const fakeBar = $('#fakeBar');
    const framesAnalyzed = $('#framesAnalyzed');
    const processingTime = $('#processingTime');
    const timestamp = $('#timestamp');
    const videoName = $('#videoName');
    const modelAccuracy = $('#modelAccuracy');
    const framePreviews = $('#framePreviews');
    const analyzeAnotherBtn = $('#analyzeAnotherBtn');
    const downloadReportBtn = $('#downloadReportBtn');
    const modalAnalyzeAnotherBtn = $('#modalAnalyzeAnotherBtn');
    const loadingModal = $('#loadingModal');
    const resultsModal = $('#resultsModal');
    const loadingMessage = $('#loadingMessage');
    const modalProgressBar = $('#modalProgressBar');
    const modalProgressText = $('#modalProgressText');
    
    // State variables
    let currentVideoFile = null;
    let uploadProgress = 0;
    let processingStartTime = null;
    
    // Initialize
    init();
    
    function init() {
        console.log('Deepfake Detection System initialized');
        
        // Update slider value display
        updateSliderValue();
        
        // Setup event listeners
        setupEventListeners();
        
        // Load model info
        loadModelInfo();
    }
    
    function setupEventListeners() {
        // Upload area click
        uploadArea.on('click', function() {
            videoInput.click();
        });
        
        // File input change
        videoInput.on('change', handleFileSelect);
        
        // Drag and drop events
        uploadArea.on('dragover', handleDragOver);
        uploadArea.on('dragleave', handleDragLeave);
        uploadArea.on('drop', handleDrop);
        
        // Remove file button
        removeFileBtn.on('click', resetFileSelection);
        
        // Sequence slider
        sequenceSlider.on('input', updateSliderValue);
        
        // Analyze button
        analyzeBtn.on('click', analyzeVideo);
        
        // Test sample button
        testSampleBtn.on('click', testSampleVideo);
        
        // Analyze another button
        analyzeAnotherBtn.on('click', resetAnalysis);
        modalAnalyzeAnotherBtn.on('click', function() {
            resultsModal.modal('hide');
            resetAnalysis();
        });
        
        // Download report button
        downloadReportBtn.on('click', downloadReport);
        
        // Modal events
        loadingModal.on('shown.bs.modal', function() {
            modalProgressBar.css('width', '0%');
        });
    }
    
    function updateSliderValue() {
        const value = sequenceSlider.val();
        sliderValue.text(value);
    }
    
    function handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.addClass('dragover');
    }
    
    function handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.removeClass('dragover');
    }
    
    function handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.removeClass('dragover');
        
        const files = e.originalEvent.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    }
    
    function handleFileSelect(e) {
        const files = e.target.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    }
    
    function handleFile(file) {
        // Validate file
        if (!isValidFile(file)) {
            showAlert('Invalid file type or size. Please upload a video file (max 500MB).', 'danger');
            return;
        }
        
        currentVideoFile = file;
        
        // Update UI
        fileName.text(file.name);
        fileSize.text(formatFileSize(file.size));
        fileInfo.show();
        
        // Create preview
        if (file.type.startsWith('video/')) {
            const videoURL = URL.createObjectURL(file);
            previewVideo.attr('src', videoURL);
            videoPreview.show();
        }
        
        // Enable analyze button
        analyzeBtn.prop('disabled', false).html('<i class="fas fa-play-circle"></i> Analyze Video');
        
        // Scroll to analyze button
        $('html, body').animate({
            scrollTop: analyzeBtn.offset().top - 100
        }, 500);
    }
    
    function isValidFile(file) {
        const allowedTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/webm'];
        const maxSize = 500 * 1024 * 1024; // 500MB
        
        if (!allowedTypes.some(type => file.type.includes(type.replace('video/', '')))) {
            return false;
        }
        
        if (file.size > maxSize) {
            return false;
        }
        
        return true;
    }
    
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    function resetFileSelection() {
        videoInput.val('');
        currentVideoFile = null;
        fileInfo.hide();
        videoPreview.hide();
        previewVideo.attr('src', '');
        analyzeBtn.prop('disabled', true);
    }
    
    function analyzeVideo() {
        if (!currentVideoFile) {
            showAlert('Please select a video file first.', 'warning');
            return;
        }
        
        // Show loading modal
        loadingMessage.text('Processing video...');
        loadingModal.modal('show');
        
        // Reset progress
        uploadProgress = 0;
        updateProgressBar(0);
        
        // Create form data
        const formData = new FormData();
        formData.append('video', currentVideoFile);
        formData.append('sequence_length', sequenceSlider.val());
        
        // Start timing
        processingStartTime = Date.now();
        
        // Simulate progress (for demo)
        simulateProgress();
        
        // Send request
        $.ajax({
            url: '/upload',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            xhr: function() {
                const xhr = new XMLHttpRequest();
                xhr.upload.addEventListener('progress', function(e) {
                    if (e.lengthComputable) {
                        const percentComplete = (e.loaded / e.total) * 100;
                        updateProgressBar(percentComplete);
                    }
                });
                return xhr;
            },
            success: function(response) {
                if (response.success) {
                    showResults(response);
                } else {
                    showAlert('Analysis failed: ' + response.error, 'danger');
                }
                loadingModal.modal('hide');
            },
            error: function(xhr, status, error) {
                showAlert('Error uploading video: ' + error, 'danger');
                loadingModal.modal('hide');
            }
        });
    }
    
    function simulateProgress() {
        if (uploadProgress < 90) {
            uploadProgress += Math.random() * 15;
            updateProgressBar(uploadProgress);
            setTimeout(simulateProgress, 500);
        }
    }
    
    function updateProgressBar(percent) {
        const roundedPercent = Math.min(100, Math.round(percent));
        progressBar.css('width', roundedPercent + '%');
        modalProgressBar.css('width', roundedPercent + '%');
        progressPercent.text(roundedPercent + '%');
        modalProgressText.text(`Processing: ${roundedPercent}%`);
        
        // Update loading message based on progress
        if (percent < 30) {
            loadingMessage.text('Uploading video...');
        } else if (percent < 60) {
            loadingMessage.text('Extracting frames...');
        } else if (percent < 90) {
            loadingMessage.text('Analyzing with AI model...');
        } else {
            loadingMessage.text('Finalizing results...');
        }
    }
    
    function showResults(result) {
        // Calculate processing time
        const processingTimeMs = Date.now() - processingStartTime;
        const processingTimeSec = (processingTimeMs / 1000).toFixed(2);
        
        // Update results
        const isFake = result.prediction === 'FAKE';
        
        // Update prediction and confidence
        predictionText.text(`Prediction: ${result.prediction}`);
        predictionText.removeClass('real-result fake-result').addClass(isFake ? 'fake-result' : 'real-result');
        confidenceScore.text(`${result.confidence}%`);
        
        // Update probabilities
        realProbability.text(`${result.real_probability}%`);
        fakeProbability.text(`${result.fake_probability}%`);
        
        // Animate progress bars
        setTimeout(() => {
            realBar.css('width', result.real_probability + '%');
            fakeBar.css('width', result.fake_probability + '%');
        }, 100);
        
        // Update details
        framesAnalyzed.text(result.frames_analyzed);
        processingTime.text(`${processingTimeSec}s`);
        timestamp.text(result.timestamp);
        videoName.text(result.video_name || 'Unknown');
        
        // Update frame previews
        updateFramePreviews(result.previews || []);
        
        // Show results section
        resultsSection.show().addClass('fade-in');
        
        // Scroll to results
        $('html, body').animate({
            scrollTop: resultsSection.offset().top - 100
        }, 1000);
        
        // Show success alert
        showAlert(`Analysis complete! Video is ${result.prediction} with ${result.confidence}% confidence.`, 
                 isFake ? 'danger' : 'success');
    }
    
    function updateFramePreviews(previews) {
        framePreviews.empty();
        
        if (previews.length === 0) {
            framePreviews.html(`
                <div class="col-12 text-center">
                    <div class="alert alert-info">
                        <i class="fas fa-info-circle"></i> No frame previews available
                    </div>
                </div>
            `);
            return;
        }
        
        previews.forEach((preview, index) => {
            const frameHtml = `
                <div class="col-md-4 col-lg-2">
                    <div class="frame-preview card">
                        <img src="${preview}" alt="Frame ${index + 1}" class="img-fluid">
                        <div class="card-footer text-center py-2">
                            <small class="text-muted">Frame ${index + 1}</small>
                        </div>
                    </div>
                </div>
            `;
            framePreviews.append(frameHtml);
        });
    }
    
    function testSampleVideo() {
        // Show loading modal
        loadingModal.modal('show');
        loadingMessage.text('Testing with sample video...');
        
        // Simulate analysis with sample data
        setTimeout(() => {
            const sampleResults = {
                success: true,
                prediction: Math.random() > 0.5 ? 'REAL' : 'FAKE',
                confidence: (Math.random() * 30 + 70).toFixed(2),
                real_probability: (Math.random() * 100).toFixed(2),
                fake_probability: (Math.random() * 100).toFixed(2),
                frames_analyzed: sequenceSlider.val(),
                timestamp: new Date().toLocaleString(),
                video_name: 'sample_video.mp4',
                previews: [
                    'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgdmlld0JveD0iMCAwIDEwMCAxMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHJlY3Qgd2lkdGg9IjEwMCIgaGVpZ2h0PSIxMDAiIGZpbGw9IiM0YTZmYTUiLz48dGV4dCB4PSI1MCUiIHk9IjUwJSIgZm9udC1mYW1pbHk9IkFyaWFsIiBmb250LXNpemU9IjEyIiBmaWxsPSJ3aGl0ZSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPlNhbXBsZTwvdGV4dD48L3N2Zz4=',
                    'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgdmlld0JveD0iMCAwIDEwMCAxMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LndzLm9yZy8yMDAwL3N2ZyI+PHJlY3Qgd2lkdGg9IjEwMCIgaGVpZ2h0PSIxMDAiIGZpbGw9IiMxNjYwODgiLz48dGV4dCB4PSI1MCUiIHk9IjUwJSIgZm9udC1mYW1pbHk9IkFyaWFsIiBmb250LXNpemU9IjEyIiBmaWxsPSJ3aGl0ZSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkZyYW1lIDI8L3RleHQ+PC9zdmc+',
                    'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgdmlld0JveD0iMCAwIDEwMCAxMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHJlY3Qgd2lkdGg9IjEwMCIgaGVpZ2h0PSIxMDAiIGZpbGw9IiMyOGE3NDUiLz48dGV4dCB4PSI1MCUiIHk9IjUwJSIgZm9udC1mYW1pbHk9IkFyaWFsIiBmb250LXNpemU9IjEyIiBmaWxsPSJ3aGl0ZSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkZyYW1lIDM8L3RleHQ+PC9zdmc+'
                ]
            };
            
            // Adjust probabilities to sum to 100
            const realProb = parseFloat(sampleResults.real_probability);
            const fakeProb = parseFloat(sampleResults.fake_probability);
            const total = realProb + fakeProb;
            sampleResults.real_probability = ((realProb / total) * 100).toFixed(2);
            sampleResults.fake_probability = ((fakeProb / total) * 100).toFixed(2);
            
            showResults(sampleResults);
            loadingModal.modal('hide');
            
            showAlert('Sample analysis complete! This is a demonstration with simulated data.', 'info');
            
        }, 2000);
    }
    
    function resetAnalysis() {
        // Reset file selection
        resetFileSelection();
        
        // Hide results section
        resultsSection.hide();
        
        // Reset progress bar
        progressBar.css('width', '0%');
        progressText.hide();
        
        // Scroll back to upload section
        $('html, body').animate({
            scrollTop: $('#upload-section').offset().top - 100
        }, 500);
        
        showAlert('Ready for new analysis. Upload another video!', 'info');
    }
    
    function downloadReport() {
        const reportData = {
            prediction: predictionText.text(),
            confidence: confidenceScore.text(),
            realProbability: realProbability.text(),
            fakeProbability: fakeProbability.text(),
            framesAnalyzed: framesAnalyzed.text(),
            processingTime: processingTime.text(),
            timestamp: timestamp.text(),
            videoName: videoName.text()
        };
        
        const reportText = `
DEEPFAKE DETECTION REPORT
=========================
Date: ${new Date().toLocaleString()}

VIDEO INFORMATION:
------------------
Video: ${reportData.videoName}
Analysis Timestamp: ${reportData.timestamp}
Processing Time: ${reportData.processingTime}
Frames Analyzed: ${reportData.framesAnalyzed}

RESULTS:
--------
${reportData.prediction}
Confidence: ${reportData.confidence}

Probability Breakdown:
- Real: ${reportData.realProbability}
- Fake: ${reportData.fakeProbability}

MODEL INFORMATION:
------------------
Model: ResNext50 + LSTM
Training Accuracy: ${modelAccuracy.text() || '85%'}
Architecture: CNN for feature extraction, LSTM for temporal analysis

NOTES:
------
This report was generated by the Deepfake Detection System.
For more information, visit the application interface.

END OF REPORT
        `;
        
        const blob = new Blob([reportText], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `deepfake_report_${Date.now()}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        showAlert('Report downloaded successfully!', 'success');
    }
    
    function loadModelInfo() {
        $.get('/model_info', function(data) {
            if (data.model_loaded) {
                const accuracy = data.train_accuracy || data.val_accuracy || '85%';
                modelAccuracy.text(typeof accuracy === 'number' ? accuracy.toFixed(2) + '%' : accuracy);
            }
        });
    }
    
    function showAlert(message, type) {
        // Remove existing alerts
        $('.alert-dismissible').remove();
        
        // Create new alert
        const alertHtml = `
            <div class="alert alert-${type} alert-dismissible fade show position-fixed top-0 start-50 translate-middle-x mt-3" 
                 style="z-index: 1050; min-width: 300px;" role="alert">
                <div class="d-flex align-items-center">
                    <i class="fas fa-${getAlertIcon(type)} me-2"></i>
                    <div>${message}</div>
                </div>
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        
        $('body').append(alertHtml);
        
        // Auto dismiss after 5 seconds
        setTimeout(() => {
            $('.alert-dismissible').alert('close');
        }, 5000);
    }
    
    function getAlertIcon(type) {
        switch(type) {
            case 'success': return 'check-circle';
            case 'danger': return 'exclamation-circle';
            case 'warning': return 'exclamation-triangle';
            case 'info': return 'info-circle';
            default: return 'info-circle';
        }
    }
    
    // Keyboard shortcuts
    $(document).on('keydown', function(e) {
        // Ctrl/Cmd + U to focus upload
        if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
            e.preventDefault();
            videoInput.click();
        }
        
        // Esc to reset
        if (e.key === 'Escape') {
            if (resultsSection.is(':visible')) {
                resetAnalysis();
            }
        }
    });
});
