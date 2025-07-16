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
                detailsBtn: "Details",
                hideDetailsBtn: "Hide Details",
                processingText: "Processing your Javanese script...",
                processingText2: "Please be patient, startup process may take 30 seconds",
                resultsTitle: "Recognition Results",
                detailsTitle: "Detailed Results",
                footerText1: "Free for personal and commercial use",
                footerText2: "This project was made as part of my OCR learning, feedback is absolutely welcomed",
                errorMessage: "Failed to process image. Please try again.",
                noDetailsData: "No details data available",
                seoHeader: "About aksara jawa scanner",
                seoHeader2: "About aksara jawa",
                seoParagraph1: "Aksara Jawa Scanner is an AI-powered tool specifically built to recognize and analyze handwritten Javanese script (Aksara Jawa). Using deep learning technology with a ResNet18 architecture, the system can accurately identify traditional Javanese characters and provide their transliteration.",
                seoParagraph2: "Currently in beta version. More features and improvements are coming soon! Check out my GitHub or social media below for feedback and contributions!",
                seoParagraph3: "Aksara Jawa, also called Javanese script or Hanacaraka, is a traditional alphabet used long ago in Java, Indonesia. It was mainly used to write the Javanese language, and it has a unique look with rounded letters and special marks called sandhangan that change the sound of the characters. Aksara Jawa comes from ancient Indian scripts and is part of Indonesia's rich cultural history.",
                seoParagraph4: "While people don't use it in daily life anymore, you can still find it in books, schools, temples, monuments, and even on street signs in Central Java and Yogyakarta. Learning it is a great way to connect with Javanese heritage and keep the culture alive!"
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
                detailsBtn: "Detail",
                hideDetailsBtn: "Sembunyikan Detail",
                processingText: "Memproses Aksara Jawa Anda...",
                processingText2: "Mohon bersabar, proses startup dapat membutuhkan waktu 30 detik",
                resultsTitle: "Hasil Pengenalan",
                detailsTitle: "Hasil Detail",
                footerText1: "Free for personal and commercial use",
                footerText2: "This project was made as part of my OCR learning, feedback is absolutely welcomed",
                errorMessage: "Gagal memproses gambar. Silakan coba lagi.",
                noDetailsData: "Tidak ada data detail yang tersedia",
                seoHeader: "Tentang scanner aksara jawa",
                seoHeader2: "Tentang aksara jawa",
                seoParagraph1: "Scanner Aksara Jawa adalah alat berbasis AI yang dibuat khusus untuk mengenali dan menganalisis tulisan tangan Aksara Jawa. Menggunakan teknologi deep learning dengan arsitektur ResNet18, sistem ini dapat mengenali karakter-karakter tradisional Aksara Jawa dengan akurat dan memberikan transliterasinya.",
                seoParagraph2: "Saat ini masih dalam versi beta, fitur dan peningkatan lainnya akan segera hadir! Kunjungi GitHub atau media sosialku di bawah untuk masukan dan kontribusi!",
                seoParagraph3: "Aksara Jawa, atau yang juga dikenal dengan sebutan Hanacaraka, adalah huruf tradisional yang dulu digunakan di Pulau Jawa, Indonesia. Aksara ini digunakan untuk menulis bahasa Jawa dan memiliki bentuk huruf yang khas dan melengkung, serta tanda-tanda tambahan yang disebut sandhangan untuk mengubah bunyi huruf. Aksara Jawa berasal dari aksara kuno India dan merupakan bagian dari sejarah budaya Indonesia.",
                seoParagraph4: "Meskipun sudah jarang dipakai sehari-hari, aksara ini masih bisa ditemukan di buku, pelajaran sekolah, candi, monumen, dan papan nama jalan di Jawa Tengah dan Yogyakarta. Belajar aksara Jawa adalah cara yang seru untuk mengenal warisan budaya dan menjaga tradisi tetap hidup."
            }
        };

        let currentLanguage = 'id';
        let uploadedImage = null;
        let detailsData = null;

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
            document.getElementById('detailsBtn').textContent = t.detailsBtn;
            document.getElementById('hideDetailsBtn').textContent = t.hideDetailsBtn;
            document.getElementById('processingText').textContent = t.processingText;
            document.getElementById('processingText2').textContent = t.processingText2;
            document.getElementById('resultsTitle').textContent = t.resultsTitle;
            document.getElementById('detailsTitle').textContent = t.detailsTitle;
            document.getElementById('footerText1').textContent = t.footerText1;
            document.getElementById('footerText2').textContent = t.footerText2;
            document.getElementById('seoHeader').textContent = t.seoHeader;
            document.getElementById('seoHeader2').textContent = t.seoHeader2;
            document.getElementById('seoParagraph1').textContent = t.seoParagraph1;
            document.getElementById('seoParagraph2').textContent = t.seoParagraph2;
            document.getElementById('seoParagraph3').textContent = t.seoParagraph3;
            document.getElementById('seoParagraph4').textContent = t.seoParagraph4;
            
            
            // Update HTML lang attribute
            document.documentElement.lang = lang;
            
            // Save language preference
            localStorage.setItem('preferredLanguage', lang);
        }

        // Initialize language on page load
        function initializeLanguage() {
            const savedLanguage = localStorage.getItem('preferredLanguage') || 'id';
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
        const detailsSection = document.getElementById('detailsSection');

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
                detailsSection.style.display = 'none';
            };
            reader.readAsDataURL(file);
        }

        async function processImage() {
            if (!uploadedImage) return;

            // Hide preview and show processing
            previewSection.style.display = 'none';
            processingSection.style.display = 'block';
            resultsSection.style.display = 'none';
            detailsSection.style.display = 'none';

            // API call
            await callOCRAPI(uploadedImage);
        }

        async function callOCRAPI(imageFile) {
            const formData = new FormData();
            formData.append('file', imageFile);
            
            try {
                const response = await fetch('https://aksara-container.delightfulcliff-10a792b4.southeastasia.azurecontainerapps.io/', { 
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const results = await response.json();
                
                // Store details data globally
                detailsData = results.details;
                
                displayResults(results.prediction); 
            } catch (error) {
                console.error('OCR API Error:', error);
                alert(translations[currentLanguage].errorMessage);
                resetInterface();
            }
        }

        function displayResults(predictions) {
            const predictionGrid = document.getElementById('predictionGrid');
            console.log("displayReluts called")
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
            detailsSection.style.display = 'none';
        }

        function populateDetailsSection() {
            console.log("populateDetails called")
            const detailsGrid = document.getElementById('detailsGrid');
            const bboxImage = document.getElementById('bboxImage');
            
            detailsGrid.innerHTML = '';
            
            if (!detailsData) {
                detailsGrid.innerHTML = `<p>${translations[currentLanguage].noDetailsData}</p>`;
                return;
            }

            // Iterate through details object and display each key-value pair
            for (const [key, value] of Object.entries(detailsData)) {
                const detailsItem = document.createElement('div');
                detailsItem.className = 'details-item';
                
                if (key === 'bbox'){
                    const image = document.createElement('img');
                    image.src = `data:image/png;base64,${value}`;
                    image.style.maxWidth = '100%';
                    image.style.border = '1px solid #ccc';
                    image.style.borderRadius = '8px';
                    detailsItem.innerHTML = `<h4>${key}</h4>`;
                    detailsItem.appendChild(image);

                } else {
                    detailsItem.innerHTML = `
                    <h4>${key}</h4>
                    <pre>${JSON.stringify(value, null, 2)}</pre>
                `;
                }

                detailsGrid.appendChild(detailsItem);
            }
        }

        function toggleDetailsSection() {
            if (detailsSection.style.display === 'none' || detailsSection.style.display === '') {
                populateDetailsSection();
                detailsSection.style.display = 'block';
            } else {
                detailsSection.style.display = 'none';
            }
        }

        function hideDetailsSection() {
            detailsSection.style.display = 'none';
        }

        function resetInterface() {
            uploadSection.style.display = 'block';
            previewSection.style.display = 'none';
            processingSection.style.display = 'none';
            resultsSection.style.display = 'none';
            detailsSection.style.display = 'none';
            fileInput.value = '';
            uploadedImage = null;
            detailsData = null;
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