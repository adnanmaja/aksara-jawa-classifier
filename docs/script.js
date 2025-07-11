// Language translations
        const translations = {
            en: {
                mainTitle: "Aksara Jawa Scanner",
                mainSubtitle: "Upload your handwritten Javanese script image for recognition",
                uploadText: "Drop your image here or click to browse",
                uploadSubtext: "Supports JPG, PNG, WebP files up to 16MB",
                chooseFileBtn: "Choose File",
                processBtn: "Process Image",
                resetBtn: "Reset",
                backBtn: "Back",
                debugBtn: "Debug",
                hideDebugBtn: "Hide Debug",
                processingText: "Processing your Javanese script...",
                resultsTitle: "Recognition Results",
                debugTitle: "Debug Results",
                footerText1: "Free for personal and commercial use",
                footerText2: "This project was made as part of my OCR learning, feedback is absolutely welcomed",
                errorMessage: "Failed to process image. Please try again.",
                noDebugData: "No debug data available"
            },
            id: {
                mainTitle: "Scanner Aksara Jawa",
                mainSubtitle: "Unggah gambar tulisan tangan Aksara Jawa untuk di-scan",
                uploadText: "Letakkan gambar di sini atau klik untuk telusuri",
                uploadSubtext: "Mendukung file JPG, PNG, WebP hingga 16MB",
                chooseFileBtn: "Pilih File",
                processBtn: "Proses Gambar",
                resetBtn: "Reset",
                backBtn: "Kembali",
                debugBtn: "Debug",
                hideDebugBtn: "Sembunyikan Debug",
                processingText: "Memproses Aksara Jawa Anda...",
                resultsTitle: "Hasil Pengenalan",
                debugTitle: "Hasil Debug",
                footerText1: "Free for personal and commercial use",
                footerText2: "This project was made as part of my OCR learning, feedback is absolutely welcomed",
                errorMessage: "Gagal memproses gambar. Silakan coba lagi.",
                noDebugData: "Tidak ada data debug yang tersedia"
            }
        };

        let currentLanguage = 'en';
        let uploadedImage = null;
        let debugData = null;

        // Language change function
        function changeLanguage(lang) {
            currentLanguage = lang;
            const t = translations[lang];
            
            // Update all text elements
            document.getElementById('mainTitle').textContent = t.mainTitle;
            document.getElementById('mainSubtitle').textContent = t.mainSubtitle;
            document.getElementById('uploadText').textContent = t.uploadText;
            document.getElementById('uploadSubtext').textContent = t.uploadSubtext;
            document.getElementById('chooseFileBtn').textContent = t.chooseFileBtn;
            document.getElementById('processBtn').textContent = t.processBtn;
            document.getElementById('resetBtn1').textContent = t.resetBtn;
            document.getElementById('resetBtn2').textContent = t.backBtn;
            document.getElementById('debugBtn').textContent = t.debugBtn;
            document.getElementById('hideDebugBtn').textContent = t.hideDebugBtn;
            document.getElementById('processingText').textContent = t.processingText;
            document.getElementById('resultsTitle').textContent = t.resultsTitle;
            document.getElementById('debugTitle').textContent = t.debugTitle;
            document.getElementById('footerText1').textContent = t.footerText1;
            document.getElementById('footerText2').textContent = t.footerText2;
            
            // Update HTML lang attribute
            document.documentElement.lang = lang;
            
            // Save language preference
            localStorage.setItem('preferredLanguage', lang);
        }

        // Initialize language on page load
        function initializeLanguage() {
            const savedLanguage = localStorage.getItem('preferredLanguage') || 'en';
            document.getElementById('languageSelector').value = savedLanguage;
            changeLanguage(savedLanguage);
        }

        // File input handling
        const fileInput = document.getElementById('fileInput');
        const uploadSection = document.getElementById('uploadSection');
        const previewSection = document.getElementById('previewSection');
        const imagePreview = document.getElementById('imagePreview');
        const processingSection = document.getElementById('processingSection');
        const resultsSection = document.getElementById('resultsSection');
        const footerSection = document.getElementById('footerSection');
        const debugSection = document.getElementById('debugSection');

        fileInput.addEventListener('change', handleFileSelect);
        uploadSection.addEventListener('click', () => fileInput.click());
        uploadSection.addEventListener('dragover', handleDragOver);
        uploadSection.addEventListener('dragleave', handleDragLeave);
        uploadSection.addEventListener('drop', handleDrop);

        function handleFileSelect(e) {
            const file = e.target.files[0];
            if (file) {
                displayImagePreview(file);
            }
        }

        function handleDragOver(e) {
            e.preventDefault();
            uploadSection.classList.add('dragover');
        }

        function handleDragLeave(e) {
            e.preventDefault();
            uploadSection.classList.remove('dragover');
        }

        function handleDrop(e) {
            e.preventDefault();
            uploadSection.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                displayImagePreview(file);
            }
        }

        function displayImagePreview(file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                uploadedImage = file;
                uploadSection.style.display = 'none';
                previewSection.style.display = 'block';
                resultsSection.style.display = 'none';
                processingSection.style.display = 'none';
                debugSection.style.display = 'none';
            };
            reader.readAsDataURL(file);
        }

        async function processImage() {
            if (!uploadedImage) return;

            // Hide preview and show processing
            previewSection.style.display = 'none';
            processingSection.style.display = 'block';
            resultsSection.style.display = 'none';
            debugSection.style.display = 'none';

            // API call
            await callOCRAPI(uploadedImage);
        }

        async function callOCRAPI(imageFile) {
            const formData = new FormData();
            formData.append('file', imageFile);
            
            try {
                const response = await fetch('https://nulisjawa.my.id/', { 
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const results = await response.json();
                
                // Store debug data globally
                debugData = results.debug;
                
                displayResults(results.prediction); 
            } catch (error) {
                console.error('OCR API Error:', error);
                alert(translations[currentLanguage].errorMessage);
                resetInterface();
            }
        }

        function displayResults(predictions) {
            const predictionGrid = document.getElementById('predictionGrid');
            
            predictionGrid.innerHTML = '';

            const fullTransliteration = predictions; 

            const item = document.createElement('div');
            item.className = 'prediction-item'; 
            item.style.gridColumn = '1 / -1'; 
            item.style.fontSize = '2.5rem'; 
            item.style.padding = '30px'; 
            item.style.fontWeight = 'bold';
            item.style.color = '#4a5568';
            item.style.display = 'flex';
            item.style.justifyContent = 'center';
            item.style.alignItems = 'center';

            item.innerHTML = `
                <div class="character">${fullTransliteration}</div>
            `;
            predictionGrid.appendChild(item);

            processingSection.style.display = 'none';
            resultsSection.style.display = 'block';
            uploadSection.style.display = 'block';
            debugSection.style.display = 'none';
        }

        function populateDebugSection() {
            const debugGrid = document.getElementById('debugGrid');
            debugGrid.innerHTML = '';
            
            if (!debugData) {
                debugGrid.innerHTML = `<p>${translations[currentLanguage].noDebugData}</p>`;
                return;
            }

            // Iterate through debug object and display each key-value pair
            for (const [key, value] of Object.entries(debugData)) {
                const debugItem = document.createElement('div');
                debugItem.className = 'debug-item';
                
                debugItem.innerHTML = `
                    <h4>${key}</h4>
                    <pre>${JSON.stringify(value, null, 2)}</pre>
                `;
                
                debugGrid.appendChild(debugItem);
            }
        }

        function toggleDebugSection() {
            if (debugSection.style.display === 'none' || debugSection.style.display === '') {
                populateDebugSection();
                debugSection.style.display = 'block';
            } else {
                debugSection.style.display = 'none';
            }
        }

        function hideDebugSection() {
            debugSection.style.display = 'none';
        }

        function resetInterface() {
            uploadSection.style.display = 'block';
            previewSection.style.display = 'none';
            processingSection.style.display = 'none';
            resultsSection.style.display = 'none';
            debugSection.style.display = 'none';
            fileInput.value = '';
            uploadedImage = null;
            debugData = null;
        }

        // File input functionality
        document.getElementById('fileInput').addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                const file = e.target.files[0];
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('imagePreview').src = e.target.result;
                    document.getElementById('previewSection').style.display = 'block';
                    document.getElementById('uploadSection').style.display = 'none';
                };
                reader.readAsDataURL(file);
            }
        });

        // Upload section click handler
        document.getElementById('uploadSection').addEventListener('click', function() {
            document.getElementById('fileInput').click();
        });

        // Choose file button
        document.getElementById('chooseFileBtn').addEventListener('click', function(e) {
            e.stopPropagation();
            document.getElementById('fileInput').click();
        });

        // Drag and drop functionality
        document.getElementById('uploadSection').addEventListener('dragover', function(e) {
            e.preventDefault();
            this.classList.add('dragover');
        });

        document.getElementById('uploadSection').addEventListener('dragleave', function(e) {
            e.preventDefault();
            this.classList.remove('dragover');
        });

        document.getElementById('uploadSection').addEventListener('drop', function(e) {
            e.preventDefault();
            this.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                if (file.type.startsWith('image/')) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        document.getElementById('imagePreview').src = e.target.result;
                        document.getElementById('previewSection').style.display = 'block';
                        document.getElementById('uploadSection').style.display = 'none';
                    };
                    reader.readAsDataURL(file);
                }
            }
        });

        // Initialize language when page loads
        window.addEventListener('load', initializeLanguage);