# 🚀 Kaggle Training Guide - Cat Breed Classification

Complete guide for training your cat breed classifier on Kaggle's free GPU.

---

## 📋 Table of Contents

1. [Why Kaggle?](#why-kaggle)
2. [Prerequisites](#prerequisites)
3. [Step 1: Prepare Dataset](#step-1-prepare-dataset)
4. [Step 2: Upload Dataset to Kaggle](#step-2-upload-dataset-to-kaggle)
5. [Step 3: Create Notebook on Kaggle](#step-3-create-notebook-on-kaggle)
6. [Step 4: Run Training](#step-4-run-training)
7. [Step 5: Download Results](#step-5-download-results)
8. [Troubleshooting](#troubleshooting)

---

## ⚡ Why Kaggle?

- **Free GPU**: 30 hours/week of GPU (P100 or T4)
- **No Setup**: Pre-installed TensorFlow, CUDA, cuDNN
- **Cloud Storage**: Keep datasets and models online
- **Long Training**: Can run for 9+ hours per session

---

## 🔧 Prerequisites

- **Kaggle Account**: [Create account](https://www.kaggle.com/account/login)
- **Phone Verification**: Required for GPU access (Settings → Account → Phone Verification)
- **Processed Dataset**: Your `data/processed/` folder with train/val/test splits

---

## 📦 Step 1: Prepare Dataset

Your processed data should have this structure:

```
data/processed/
├── train/
│   ├── Abyssinian/
│   ├── American Bobtail/
│   ├── ... (63 breeds)
│   └── Van Kedisi/
├── val/
│   └── ... (same 63 breeds)
└── test/
    └── ... (same 63 breeds)
```

### Compress Dataset (Optional but Recommended)

To speed up upload, compress your processed folder:

**On Linux/Mac:**
```bash
cd data
zip -r processed.zip processed/
```

**On Windows:**
- Right-click `processed` folder → Send to → Compressed (zipped) folder

**Expected size:** ~500MB - 2GB (depending on images)

---

## ☁️ Step 2: Upload Dataset to Kaggle

### Option A: Web Upload (Recommended for < 2GB)

1. **Go to Kaggle Datasets**
   - Visit: https://www.kaggle.com/datasets
   - Click "New Dataset"

2. **Upload Files**
   - Click "Upload Files"
   - Select your `processed.zip` (or drag the `processed` folder directly)
   - Wait for upload to complete (may take 5-30 minutes)

3. **Configure Dataset**
   - **Title**: `Cat Classification - Processed Dataset`
   - **Subtitle**: `Processed cat breed images (63 breeds, train/val/test split)`
   - **Visibility**: Private (recommended) or Public
   - **Tags**: `image classification`, `cats`, `deep learning`

4. **Create Dataset**
   - Click "Create"
   - Note the dataset URL: `kaggle.com/datasets/YOUR_USERNAME/DATASET_NAME`
   - **Important**: Copy the dataset name (e.g., `cat-classification-processed`)

### Option B: Kaggle CLI (For Larger Datasets)

1. **Install Kaggle CLI**
   ```bash
   pip install kaggle
   ```

2. **Setup API Token**
   - Go to: https://www.kaggle.com/settings/account
   - Scroll to "API" → Click "Create New Token"
   - Download `kaggle.json` and place it in:
     - Linux/Mac: `~/.kaggle/kaggle.json`
     - Windows: `C:\Users\YOUR_USERNAME\.kaggle\kaggle.json`

3. **Create Dataset Metadata**

   Create `data/dataset-metadata.json`:
   ```json
   {
     "title": "Cat Classification - Processed Dataset",
     "id": "YOUR_USERNAME/cat-classification-processed",
     "licenses": [{"name": "CC0-1.0"}]
   }
   ```

4. **Upload**
   ```bash
   cd data
   kaggle datasets create -p .
   ```

---

## 📓 Step 3: Create Notebook on Kaggle

1. **Open Kaggle Notebooks**
   - Visit: https://www.kaggle.com/code
   - Click "New Notebook"

2. **Upload Your Notebook**
   - Click "File" → "Upload Notebook"
   - Select: `notebooks/model_training_kaggle.ipynb`
   - Or copy-paste the entire notebook content

3. **Configure Notebook Settings**

   Click "Settings" (right panel):

   - **Accelerator**: **GPU P100** or **GPU T4** (IMPORTANT!)
   - **Language**: Python
   - **Environment**: Latest available
   - **Internet**: On (if you need to install packages)
   - **Persistence**: Files only (default)

4. **Add Dataset**

   In right panel "Input":
   - Click "+ Add Data"
   - Search for your dataset name (e.g., `cat-classification-processed`)
   - Click "Add"
   - Verify it appears under "Input Data" as `/kaggle/input/YOUR_DATASET_NAME`

5. **Update Dataset Path in Notebook**

   In cell 4 (Configuration), update this line:
   ```python
   # IMPORTANT: Change this to your dataset name!
   DATASET_NAME = 'cat-classification-processed'  # ← Your dataset name here
   ```

   Replace with your actual dataset name from Step 2.

---

## 🏃 Step 4: Run Training

### Quick Start

1. **Run First 6 Cells**
   - Click on cell 1
   - Press `Shift + Enter` to run each cell
   - Verify:
     - Cell 2: GPU detected
     - Cell 3: Configuration printed
     - Cell 6: Dataset found with 63 breeds

2. **Start Full Training**
   - Option A: Click "Run All" (top menu)
   - Option B: Keep pressing `Shift + Enter` through all cells

3. **Monitor Progress**

   Watch for:
   - **Stage 1 (Feature Extraction)**: ~2-3 hours, 50 epochs
   - **Stage 2 (Fine-tuning)**: ~1-2 hours, 30 epochs
   - **Total time**: ~3-5 hours on P100 GPU

### Expected Output

**Cell 2 - GPU Check:**
```
✓ TensorFlow version: 2.15.0
✓ GPU Available: 1 device(s)
```

**Cell 6 - Dataset Verification:**
```
✓ Dataset found: /kaggle/input/cat-classification-processed
✓ Train: /kaggle/input/.../train (63 breeds)
✓ Val  : /kaggle/input/.../val (63 breeds)
✓ Test : /kaggle/input/.../test (63 breeds)

✓ Total cat breeds: 63
```

**Cell 16 - Stage 1 Training:**
```
STAGE 1: FEATURE EXTRACTION (Base Frozen)
Steps per epoch: 184
Validation steps: 40
Epochs: 50
Learning rate: 0.0001

Epoch 1/50
184/184 [======] - 120s 650ms/step - loss: 3.2156 - accuracy: 0.2134 - val_loss: 2.1023 - val_accuracy: 0.4523
...
```

**Final Cell - Summary:**
```
TRAINING COMPLETE! 🎉
{
  "timestamp": "2025-11-05T10:30:00",
  "model": "ResNet50V2",
  "num_classes": 63,
  "test": {
    "accuracy": 0.7845,
    "top5_accuracy": 0.9523
  }
}
```

### Save Progress

**IMPORTANT**: If training is interrupted, you'll lose everything!

To save:
1. Click "Save Version" (top right)
2. Select "Save & Run All"
3. This creates a snapshot that runs to completion even if you close the browser

---

## 💾 Step 5: Download Results

### After Training Completes

1. **Check Outputs**

   All files are in `/kaggle/working/`:
   ```
   models/
     ├── cat_breed_classifier_final.keras    ← Main model (100MB)
     ├── best_stage1.keras
     ├── best_stage2.keras
     └── class_indices.json

   plots/
     ├── history_stage1.png
     ├── history_stage2.png
     └── confusion_matrix.png

   reports/
     ├── training_summary.json
     ├── classification_report.txt
     ├── training_stage1.log
     └── training_stage2.log
   ```

2. **Download Method 1: Via Notebook**

   After running all cells:
   - Right panel → Click "Output" tab
   - You'll see all files listed
   - Click download icon next to each file
   - Or click "Download All" to get everything as ZIP

3. **Download Method 2: Via Saved Version**

   - Click "Save Version" → "Save & Run All" (if not already)
   - Wait for version to complete (you can close browser)
   - Later, go to your notebook → "Versions" tab
   - Click on completed version
   - Click "Output" → "Download All"

4. **Copy to Your Project**

   After download:
   ```bash
   # Extract if downloaded as ZIP
   unzip kaggle-output.zip -d kaggle-output/

   # Copy model files
   cp kaggle-output/models/*.keras models/
   cp kaggle-output/models/class_indices.json models/

   # Copy results
   cp -r kaggle-output/plots outputs/
   cp -r kaggle-output/reports outputs/
   ```

### Test Your Downloaded Model

```bash
# In your local project
python inference.py --image test_cat.jpg --model models/cat_breed_classifier_final.keras
```

---

## 🔧 Troubleshooting

### Problem: GPU Not Available

**Error:**
```
✓ GPU Available: 0 device(s)
```

**Solutions:**
1. Go to notebook settings (right panel)
2. Accelerator → Select "GPU P100" or "GPU T4"
3. Save settings and restart notebook
4. If still fails: You may have exhausted weekly quota (30 hours)
   - Check: https://www.kaggle.com/settings → Usage
   - Quota resets weekly

### Problem: Dataset Not Found

**Error:**
```
❌ ERROR: Dataset not found at /kaggle/input/cat-classification-processed

Available datasets in /kaggle/input:
  - some-other-dataset
```

**Solutions:**
1. Check if dataset is attached: Right panel → Input → Should see your dataset
2. If not attached: "+ Add Data" → Search → Add your dataset
3. Update `DATASET_NAME` in cell 4 to match actual dataset folder name
4. Verify folder structure: Add new cell with `!ls /kaggle/input/YOUR_DATASET_NAME/`

### Problem: Out of Memory (OOM)

**Error:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**Solutions:**
1. Reduce batch size in cell 4:
   ```python
   BATCH_SIZE = 16  # Instead of 32
   ```
2. Reduce image size (not recommended):
   ```python
   IMG_WIDTH, IMG_HEIGHT = 192, 192  # Instead of 224
   ```
3. Restart kernel and run again

### Problem: Training Takes Too Long

**Expected times on P100:**
- Stage 1: ~2-3 hours (50 epochs)
- Stage 2: ~1-2 hours (30 epochs)

**If slower:**
1. Verify GPU is enabled (should see `GPU Available: 1`)
2. Reduce epochs in cell 4:
   ```python
   EPOCHS_STAGE1 = 30  # Instead of 50
   EPOCHS_STAGE2 = 20  # Instead of 30
   ```

### Problem: Can't Download Large Files

**If model file is too large:**

1. **Method 1**: Download via Kaggle CLI
   ```bash
   kaggle kernels output YOUR_USERNAME/YOUR_NOTEBOOK_NAME -p ./downloads/
   ```

2. **Method 2**: Upload to Google Drive (add cell at end of notebook)
   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   import shutil
   shutil.copytree('/kaggle/working/models', '/content/drive/MyDrive/cat-models/')
   ```

### Problem: Notebook Execution Fails

**If "Save & Run All" fails:**

1. Click "View Logs" to see error
2. Common issues:
   - Dataset path wrong → Fix `DATASET_NAME`
   - Not enough GPU memory → Reduce `BATCH_SIZE`
   - Package missing → Add install cell: `!pip install package-name`

---

## 📊 Expected Results

### Performance Metrics

With proper training, you should achieve:

- **Test Accuracy**: 75-85% (63 breeds is challenging!)
- **Top-5 Accuracy**: 92-96%
- **Training Time**: 3-5 hours on P100 GPU
- **Model Size**: ~100MB

### Good Signs During Training

**Stage 1:**
```
Epoch 10/50: loss: 2.1234 - accuracy: 0.45 - val_accuracy: 0.52
Epoch 20/50: loss: 1.5678 - accuracy: 0.58 - val_accuracy: 0.64
Epoch 40/50: loss: 0.9876 - accuracy: 0.72 - val_accuracy: 0.73
```

**Stage 2:**
```
Epoch 10/30: loss: 0.7234 - accuracy: 0.78 - val_accuracy: 0.77
Epoch 20/30: loss: 0.5123 - accuracy: 0.84 - val_accuracy: 0.80
```

### Warning Signs

- **Val accuracy decreasing**: Overfitting, but EarlyStopping will handle it
- **Loss becomes NaN**: Reduce learning rate or batch size
- **Accuracy stuck at low value**: Check dataset paths, verify images loaded correctly

---

## 🎯 Next Steps After Training

1. **Evaluate Model Locally**
   ```bash
   python inference.py --image examples/british_shorthair.jpg
   ```

2. **Test on Multiple Images**
   ```bash
   python inference.py --batch examples/*.jpg --output results.json
   ```

3. **Analyze Results**
   - Review confusion matrix: `outputs/plots/confusion_matrix.png`
   - Check classification report: `outputs/reports/classification_report.txt`
   - Find weakest breeds and consider collecting more data

4. **Improve Model** (Optional)
   - Add more augmentation
   - Try different architectures (EfficientNet, Vision Transformer)
   - Implement Grad-CAM for visualization
   - Fine-tune for more epochs

---

## 🆘 Need Help?

1. **Check Kaggle Docs**: https://www.kaggle.com/docs
2. **Kaggle Community**: https://www.kaggle.com/discussions
3. **Project Issues**: https://github.com/theAbyssOfTime2004/cat-classification/issues

---

## ✅ Quick Checklist

Before training:
- [ ] Kaggle account created and phone verified
- [ ] Dataset uploaded to Kaggle
- [ ] Notebook uploaded and settings configured
- [ ] GPU enabled (P100 or T4)
- [ ] Dataset attached to notebook
- [ ] `DATASET_NAME` variable updated in cell 4

During training:
- [ ] GPU detected in cell 2
- [ ] Dataset found in cell 6
- [ ] Training started successfully
- [ ] Monitoring progress (can close browser after "Save Version")

After training:
- [ ] All cells completed without errors
- [ ] Test accuracy displayed in final cell
- [ ] Output files downloaded
- [ ] Model copied to local `models/` folder
- [ ] Tested with `inference.py`

---

**Good luck with training! 🚀🐱**
