# Per-stage design choices (A2 deliverable). Fill every cell.
| Stage | Problem statement | Data | Model | Methods | Design | Development | Deployment | MLOps |
|---|---|---|---|---|---|---|---|---|
| 0 Frame |  |  |  |  |  |  |  |  |
| 1 Ingest+Enhance |  |  |  |  |  |  |  |  |
| 2 Layout | Detect & classify page regions (text, figure, table, heading) to prevent OCR from reading non-text content | NCTB Physics, Chemistry, Biology page images (PNG) | YOLO11 fine-tuned on DocLayNet (`Armaggheddon/yolo11-document-layout`) | Object detection; 11 DocLayNet categories mapped to 4 region types | score_thr=0.5; figures and tables excluded from OCR pipeline | ultralytics + HuggingFace Hub; lazy model load | CPU inference; model auto-downloaded from HuggingFace on first run | Pretrained weights; model version pinned via HuggingFace model ID |
| 3 OCR | Transcribe Bangla + Latin/English code-mixed text from cropped region images | Cropped region images from layout stage; Bangla script with conjuncts, diacritics, and Latin variable names | EasyOCR (CRNN: CNN + LSTM + CTC) with `bn+en` language pack | Multilingual CRNN recognition; GPU auto-detected, falls back to CPU | 5 backends switchable via cfg; EasyOCR chosen for best accuracy on code-mixed content | easyocr library; Python | CPU/GPU; weights auto-downloaded on first run | Pretrained weights; fine-tunable via deep-text-recognition-benchmark |
| 4 Index | Split OCR text into retrievable semantic chunks, embed in a cross-lingual vector space, and persist a searchable index | Raw OCR text per region; Bangla + English code-mixed | `intfloat/multilingual-e5-large` (embedding); FAISS `IndexFlatIP` (vector store) | Sentence-boundary semantic chunking (E4 bonus); multilingual dense embedding (E23 bonus); exact cosine search via FAISS | Splits on danda/pipe/newline/Latin punctuation; 500-char chunks, 50-char overlap; L2-normalised vectors | sentence-transformers, faiss-cpu; Python | CPU inference; index persisted to `data/index/`; loaded at retriever init | Pretrained e5-large weights; index rebuilt on corpus update via `build_index.sh` |
| 5 Retrieval |  |  |  |  |  |  |  |  |
| 6 Agent |  |  |  |  |  |  |  |  |
| 7 RL/RLVR |  |  |  |  |  |  |  |  |
| 8 Serving |  |  |  |  |  |  |  |  |
| 9 Eval |  |  |  |  |  |  |  |  |
