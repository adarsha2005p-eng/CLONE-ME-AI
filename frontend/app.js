// CLONE-ME-AI Frontend Logic

document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const uploadPreviewContainer = document.getElementById('upload-preview-container');
    const uploadPromptContainer = document.getElementById('upload-prompt-container');
    const uploadPreviewImg = document.getElementById('upload-preview-img');
    const removeUploadBtn = document.getElementById('remove-upload-btn');

    const promptInput = document.getElementById('prompt-input');
    const negPromptInput = document.getElementById('neg-prompt-input');
    const seedInput = document.getElementById('seed-input');
    const stepsInput = document.getElementById('steps-input');
    const stepsVal = document.getElementById('steps-val');
    const cfgInput = document.getElementById('cfg-input');
    const cfgVal = document.getElementById('cfg-val');

    const generateBtn = document.getElementById('generate-btn');
    const generateBtnText = document.getElementById('generate-btn-text');
    const generateSpinner = document.getElementById('generate-spinner');

    const statusBadge = document.getElementById('comfy-status-badge');
    const statusText = document.getElementById('comfy-status-text');
    const statusDot = document.getElementById('comfy-status-dot');

    const emptyState = document.getElementById('empty-state');
    const loadingState = document.getElementById('loading-state');
    const loadingStageText = document.getElementById('loading-stage-text');
    const progressBar = document.getElementById('progress-bar');
    const resultState = document.getElementById('result-state');

    const resultImg = document.getElementById('result-img');
    const compBeforeImg = document.getElementById('comp-before-img');
    const compAfterImg = document.getElementById('comp-after-img');
    const comparisonWrapper = document.getElementById('comparison-wrapper');
    const comparisonHandle = document.getElementById('comparison-handle');
    const comparisonContainer = document.getElementById('comparison-container');

    const sideBySideView = document.getElementById('side-by-side-view');
    const splitSliderView = document.getElementById('split-slider-view');
    const singleResultView = document.getElementById('single-result-view');
    const sideRefImg = document.getElementById('side-ref-img');
    const sideResImg = document.getElementById('side-res-img');

    const viewModeSplitBtn = document.getElementById('view-mode-split');
    const viewModeSideBtn = document.getElementById('view-mode-side');
    const viewModeSingleBtn = document.getElementById('view-mode-single');

    const similarityBadge = document.getElementById('similarity-badge');
    const similarityScoreText = document.getElementById('similarity-score-text');
    const durationText = document.getElementById('duration-text');
    const seedResultText = document.getElementById('seed-result-text');

    const downloadBtn = document.getElementById('download-btn');
    const copyUrlBtn = document.getElementById('copy-url-btn');

    let selectedFile = null;
    let currentResultData = null;
    let progressTimer = null;

    // Presets
    const presets = {
        'white-crop-top': "photorealistic photograph of the same woman from the original reference photograph, preserve the original facial identity exactly as closely as technically possible, natural recognizable face, accurate original eyes, original eyebrows, original nose, original lips, original cheeks, original jawline, natural skin texture, natural hair, realistic human anatomy, realistic body proportions, natural standing three-quarter pose, anatomically correct shoulders arms elbows wrists hands fingers torso waist hips legs knees and feet, wearing a white ribbed crop top and blue jeans, natural indoor lighting, realistic camera perspective, DSLR photography",
        'casual-blazer': "photorealistic portrait photograph of the same woman from the reference image, exact facial identity preserved, natural face, recognizable eyebrows and eyes, standing 3/4 pose, wearing an elegant navy blue blazer over a clean silk shirt and tailored dark jeans, soft natural studio lighting, 8k resolution, photoreal",
        'summer-dress': "photorealistic full-body shot of the same woman from the reference photograph, strict facial identity preservation, identical facial structure and eyes, standing pose outdoors on a sunlit veranda, wearing a light floral summer sundress, natural sunlight, cinematic bokeh, highly detailed",
        'athletic-wear': "photorealistic fitness portrait of the same woman from the reference photograph, identical face and features, natural skin, standing active pose, wearing modern seamless workout crop top and black yoga leggings, bright modern gym interior lighting, athletic aesthetic"
    };

    document.querySelectorAll('.preset-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const presetKey = btn.getAttribute('data-preset');
            if (presets[presetKey]) {
                promptInput.value = presets[presetKey];
                // Highlight active preset
                document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('border-indigo-500', 'bg-indigo-950/40'));
                btn.classList.add('border-indigo-500', 'bg-indigo-950/40');
            }
        });
    });

    // Slider inputs display update
    stepsInput.addEventListener('input', (e) => stepsVal.textContent = e.target.value);
    cfgInput.addEventListener('input', (e) => cfgVal.textContent = parseFloat(e.target.value).toFixed(1));

    // Health check polling
    async function checkHealth() {
        try {
            const res = await fetch('/api/health');
            const data = await res.json();
            if (data.comfyui === 'online') {
                statusText.textContent = 'ComfyUI Online';
                statusDot.className = 'pulse-dot bg-emerald-500';
                statusBadge.className = 'inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium bg-emerald-950/60 border border-emerald-500/30 text-emerald-400';
            } else {
                statusText.textContent = 'ComfyUI Offline';
                statusDot.className = 'pulse-dot bg-rose-500';
                statusBadge.className = 'inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium bg-rose-950/60 border border-rose-500/30 text-rose-400';
            }
        } catch (e) {
            statusText.textContent = 'Backend Offline';
            statusDot.className = 'pulse-dot bg-amber-500';
            statusBadge.className = 'inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium bg-amber-950/60 border border-amber-500/30 text-amber-400';
        }
    }
    checkHealth();
    setInterval(checkHealth, 6000);

    // File selection & dropzone handling
    function handleFile(file) {
        if (!file || !file.type.startsWith('image/')) {
            alert('Please select a valid image file (JPG, PNG, WEBP).');
            return;
        }
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            uploadPreviewImg.src = e.target.result;
            uploadPromptContainer.classList.add('hidden');
            uploadPreviewContainer.classList.remove('hidden');
            generateBtn.disabled = false;
            generateBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        };
        reader.readAsDataURL(file);
    }

    dropzone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    });

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Clipboard paste support
    window.addEventListener('paste', (e) => {
        const items = (e.clipboardData || e.originalEvent.clipboardData).items;
        for (let item of items) {
            if (item.kind === 'file' && item.type.startsWith('image/')) {
                handleFile(item.getAsFile());
                break;
            }
        }
    });

    removeUploadBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        selectedFile = null;
        fileInput.value = '';
        uploadPreviewImg.src = '';
        uploadPreviewContainer.classList.add('hidden');
        uploadPromptContainer.classList.remove('hidden');
        generateBtn.disabled = true;
        generateBtn.classList.add('opacity-50', 'cursor-not-allowed');
    });

    // Generation handling
    generateBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // UI state: generating
        generateBtn.disabled = true;
        generateBtn.classList.add('opacity-75');
        generateBtnText.textContent = 'Generating Clone...';
        generateSpinner.classList.remove('hidden');

        emptyState.classList.add('hidden');
        resultState.classList.add('hidden');
        loadingState.classList.remove('hidden');

        // Progress simulation
        let currentProgress = 5;
        progressBar.style.width = `${currentProgress}%`;
        loadingStageText.textContent = 'Step 1/4: Analyzing face landmarks and alignment...';

        clearInterval(progressTimer);
        progressTimer = setInterval(() => {
            if (currentProgress < 30) {
                currentProgress += 3;
                loadingStageText.textContent = 'Step 1/4: Analyzing face landmarks and alignment...';
            } else if (currentProgress < 60) {
                currentProgress += 1.5;
                loadingStageText.textContent = 'Step 2/4: Positioning on master canvas & generating mask...';
            } else if (currentProgress < 85) {
                currentProgress += 0.8;
                loadingStageText.textContent = 'Step 3/4: SDXL Lightning Inpainting & OpenPose guidance...';
            } else if (currentProgress < 96) {
                currentProgress += 0.3;
                loadingStageText.textContent = 'Step 4/4: Decoding latent & evaluating cosine similarity...';
            }
            progressBar.style.width = `${Math.min(96, currentProgress)}%`;
        }, 800);

        const formData = new FormData();
        formData.append('image', selectedFile);
        formData.append('prompt', promptInput.value);
        formData.append('negative_prompt', negPromptInput.value);
        formData.append('seed', seedInput.value || '-1');
        formData.append('steps', stepsInput.value || '12');
        formData.append('cfg', cfgInput.value || '2.0');

        try {
            const resp = await fetch('/api/generate', {
                method: 'POST',
                body: formData
            });

            if (!resp.ok) {
                const errData = await resp.json().catch(() => ({ detail: 'Server Error' }));
                throw new Error(errData.detail || 'Generation failed');
            }

            const data = await resp.json();
            currentResultData = data;

            // Complete progress
            clearInterval(progressTimer);
            progressBar.style.width = '100%';
            loadingStageText.textContent = 'Generation Complete!';

            setTimeout(() => {
                displayResult(data);
            }, 600);

        } catch (err) {
            clearInterval(progressTimer);
            alert(`Error during generation: ${err.message}`);
            loadingState.classList.add('hidden');
            emptyState.classList.remove('hidden');
        } finally {
            generateBtn.disabled = false;
            generateBtn.classList.remove('opacity-75');
            generateBtnText.textContent = 'Generate Clone';
            generateSpinner.classList.add('hidden');
        }
    });

    // Display result
    function displayResult(data) {
        loadingState.classList.add('hidden');
        resultState.classList.remove('hidden');

        const resultUrl = data.result_url + `?t=${Date.now()}`;
        const refUrl = data.reference_url + `?t=${Date.now()}`;

        resultImg.src = resultUrl;
        compBeforeImg.src = refUrl;
        compAfterImg.src = resultUrl;
        sideRefImg.src = refUrl;
        sideResImg.src = resultUrl;

        // Metrics
        const sim = data.similarity_score;
        similarityScoreText.textContent = sim.toFixed(4);
        if (sim >= 0.85) {
            similarityBadge.className = 'px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-950/80 border border-emerald-500/40 text-emerald-300';
            similarityScoreText.textContent += ' (Exceptional Match)';
        } else if (sim >= 0.75) {
            similarityBadge.className = 'px-2.5 py-1 rounded-full text-xs font-semibold bg-blue-950/80 border border-blue-500/40 text-blue-300';
            similarityScoreText.textContent += ' (High Similarity)';
        } else {
            similarityBadge.className = 'px-2.5 py-1 rounded-full text-xs font-semibold bg-amber-950/80 border border-amber-500/40 text-amber-300';
            similarityScoreText.textContent += ' (Moderate Similarity)';
        }

        durationText.textContent = `${data.duration_seconds}s`;
        seedResultText.textContent = `${data.seed}`;

        // Ensure split slider dimensions match
        resetSplitSlider();
    }

    // Split Slider Dragging Logic
    let isDragging = false;

    function updateSplitSlider(xPos) {
        const rect = comparisonContainer.getBoundingClientRect();
        let posX = xPos - rect.left;
        posX = Math.max(0, Math.min(rect.width, posX));
        const percent = (posX / rect.width) * 100;

        comparisonWrapper.style.width = `${percent}%`;
        comparisonHandle.style.left = `${percent}%`;
    }

    function resetSplitSlider() {
        comparisonWrapper.style.width = '50%';
        comparisonHandle.style.left = '50%';
        if (compBeforeImg.naturalWidth && compAfterImg.naturalWidth) {
            compBeforeImg.style.width = `${comparisonContainer.clientWidth}px`;
        }
    }

    window.addEventListener('resize', resetSplitSlider);

    comparisonContainer.addEventListener('mousedown', (e) => {
        isDragging = true;
        updateSplitSlider(e.clientX);
    });
    window.addEventListener('mousemove', (e) => {
        if (isDragging) updateSplitSlider(e.clientX);
    });
    window.addEventListener('mouseup', () => isDragging = false);

    // Touch support
    comparisonContainer.addEventListener('touchstart', (e) => {
        isDragging = true;
        if (e.touches.length > 0) updateSplitSlider(e.touches[0].clientX);
    }, { passive: true });
    window.addEventListener('touchmove', (e) => {
        if (isDragging && e.touches.length > 0) updateSplitSlider(e.touches[0].clientX);
    }, { passive: true });
    window.addEventListener('touchend', () => isDragging = false);

    // View mode switching
    viewModeSplitBtn.addEventListener('click', () => {
        splitSliderView.classList.remove('hidden');
        sideBySideView.classList.add('hidden');
        singleResultView.classList.add('hidden');

        viewModeSplitBtn.classList.add('bg-indigo-600', 'text-white');
        viewModeSplitBtn.classList.remove('bg-gray-800', 'text-gray-300');
        viewModeSideBtn.classList.remove('bg-indigo-600', 'text-white');
        viewModeSideBtn.classList.add('bg-gray-800', 'text-gray-300');
        viewModeSingleBtn.classList.remove('bg-indigo-600', 'text-white');
        viewModeSingleBtn.classList.add('bg-gray-800', 'text-gray-300');
        resetSplitSlider();
    });

    viewModeSideBtn.addEventListener('click', () => {
        splitSliderView.classList.add('hidden');
        sideBySideView.classList.remove('hidden');
        singleResultView.classList.add('hidden');

        viewModeSideBtn.classList.add('bg-indigo-600', 'text-white');
        viewModeSideBtn.classList.remove('bg-gray-800', 'text-gray-300');
        viewModeSplitBtn.classList.remove('bg-indigo-600', 'text-white');
        viewModeSplitBtn.classList.add('bg-gray-800', 'text-gray-300');
        viewModeSingleBtn.classList.remove('bg-indigo-600', 'text-white');
        viewModeSingleBtn.classList.add('bg-gray-800', 'text-gray-300');
    });

    viewModeSingleBtn.addEventListener('click', () => {
        splitSliderView.classList.add('hidden');
        sideBySideView.classList.add('hidden');
        singleResultView.classList.remove('hidden');

        viewModeSingleBtn.classList.add('bg-indigo-600', 'text-white');
        viewModeSingleBtn.classList.remove('bg-gray-800', 'text-gray-300');
        viewModeSplitBtn.classList.remove('bg-indigo-600', 'text-white');
        viewModeSplitBtn.classList.add('bg-gray-800', 'text-gray-300');
        viewModeSideBtn.classList.remove('bg-indigo-600', 'text-white');
        viewModeSideBtn.classList.add('bg-gray-800', 'text-gray-300');
    });

    // Download button
    downloadBtn.addEventListener('click', () => {
        if (!currentResultData) return;
        const a = document.createElement('a');
        a.href = currentResultData.result_url;
        a.download = currentResultData.result_filename || 'clone_me_result.png';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    });

    // Copy URL button
    copyUrlBtn.addEventListener('click', () => {
        if (!currentResultData) return;
        const fullUrl = window.location.origin + currentResultData.result_url;
        navigator.clipboard.writeText(fullUrl).then(() => {
            const origText = copyUrlBtn.innerHTML;
            copyUrlBtn.innerHTML = `<span>Copied!</span>`;
            setTimeout(() => copyUrlBtn.innerHTML = origText, 2000);
        });
    });
});
