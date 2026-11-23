# VVC 32×32 Fast CU Partitioning (Heuristic + CNN / IAI)

![VVC](https://img.shields.io/badge/Standard-VVC%20%2F%20H.266-blue)
![VTM](https://img.shields.io/badge/Reference%20Software-VTM%2024.0-informational)
![C++](https://img.shields.io/badge/Core-C%2B%2B17%20%2F%20AVX2-00599C)
![Python](https://img.shields.io/badge/AI%20Pipeline-Python%20%2F%20PyTorch%20%2F%20ONNX-EE4C2C)

本專案建構於 **VTM 24.0**，針對 All-Intra 編碼中的 **32×32 Coding Unit (CU)** 提供兩階段快速分割加速框架。方法結合傳統紋理啟發式規則、輕量 CNN 與可解釋 AI（IAI），用於剪枝 QTMT 搜尋中的冗餘 RDO 評估，同時控制編碼效率損失。

## 1. 核心架構

在 32×32 CU 節點（`EncCu::xCompressCU`）將決策流程拆分為兩個串接階段，以平衡時間節省 $\Delta T$ 與 BD-Rate 退化。

### Stage 1：紋理感知啟發式剪枝

- 擷取 32×32 Luma 區塊的水平／垂直 Sobel 梯度能量比與子區塊亮度變異數。
- 對均勻區域直接終止分割（`NO_SPLIT`）。
- 對強方向性邊緣剪枝正交分割候選（`BTH`/`TTH` 與 `BTV`/`TTV`）。
- 設計目標：在神經推論前減少約 30–40% 的冗餘 RDO 評估。

### Stage 2：輕量 CNN 與 IAI 決策修正

- 將通過 Stage 1 的模糊、高複雜度紋理區塊，連同 32×32 Luma patch 與 QP，輸入輕量 CNN。
- CNN 輸出 6 種 QTMT 模式的機率分佈：`NO_SPLIT`、`QT`、`BTH`、`BTV`、`TTH`、`TTV`。
- IAI：使用 Grad-CAM / SHAP 歸因與知識蒸餾，將 CNN 特徵權重轉換為明確決策規則與硬體友善的查找閾值。

## 2. 倉庫結構

大型 YUV 測試序列與外部 VTM 原始碼不納入版本控制；建置採用 out-of-source CMake，保持倉庫輕量且可重現。

```text
VVC-32x32-Fast-CU-Partitioning/
├── configs/             # paths.yaml 與 CTC QP (22, 27, 32, 37) 設定
├── vtm_patches/         # VTM 24.0 C++ diff（EncCu.cpp）與 out-of-source 建置腳本
├── data_pipeline/       # 平行 VTM anchor 執行器與 32x32 Luma/QTMT ground-truth 擷取
├── models/              # Stage-1 heuristic baseline、Stage-2 CNN、IAI explainers
├── evaluation/          # BD-Rate / BD-PSNR 與 ΔT 計算
├── external/            # 未修改的 VTM 24.0（git-ignored）
└── datasets/            # 處理後 .npz / .h5 訓練資料（git-ignored）
```

## 3. 建置

### 環境需求

- Linux（Ubuntu 24.04 LTS）
- GCC 13+，支援 AVX2 SIMD
- CMake ≥ 3.16
- Python 3.10+（PyTorch、ONNX Runtime、NumPy、SciPy）
- 官方 [VTM 24.0](https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM) 置於 `external/VTM24.0`

### 建置 VTM 24.0 Anchor

```bash
chmod +x vtm_patches/build_vtm.sh
./vtm_patches/build_vtm.sh
```

編譯產物 `EncoderAppStatic` 隔離於 `./bin/`，不污染 `external/VTM24.0`。

## 4. 路線圖與進度

- [x] **Phase 1：基礎設施與 Anchor Baseline**
  - 完成 out-of-source CMake Release 建置流程（啟用 AVX2）與 Git patch 工作流。
  - 已驗證 All-Intra（`encoder_intra_vtm.cfg`）於 JEM/JVET 測試序列的 baseline 編碼。

- [ ] **Phase 2：Ground-Truth 資料管線**
  - 在 `EncCu::xCompressCU` 記錄 32×32 Luma patch、QP 與最佳 QTMT 分割模式，涵蓋 Class A1–F。
  - 多進程平行資料收集與 HDF5/NPZ 資料集封裝。

- [ ] **Phase 3：Stage-1 啟發式 Baseline**
  - 以 Python 與 C++ 實作梯度／變異數早期終止與方向性剪枝。
  - 評估純啟發式方法的 $\Delta T$ 與 BD-Rate 取捨。

- [ ] **Phase 4：Stage-2 輕量 CNN 與 IAI 分析**
  - 以 Stage-1 殘留困難樣本訓練輕量 CNN，執行 Grad-CAM 與決策樹蒸餾。
  - 匯出 ONNX / TorchScript，整合 C++ 推論至 `EncCu.cpp`。

- [ ] **Phase 5：硬體／Edge-AI 協同設計**
  - 將蒸餾後的 IAI 規則與 CNN 量化為定點運算（`Q15`/`INT8`）。
  - 探索 RTL 硬體加速器（Verilog HDL / RISC-V 自定義指令）於即時 VVC Intra 編碼器。

## 5. GitHub About 建議

- **Description**：`Two-stage fast 32x32 intra CU partitioning for VVC (VTM 24.0) combining texture heuristics, lightweight CNN, and Interpretable AI (IAI).`
- **Topics**：`vvc` `h266` `vtm` `video-coding` `cu-partitioning` `cnn` `explainable-ai` `cpp` `computer-vision` `edge-ai`

## gitignore

### Build artifacts
build/
bin/

### External source tree
external/VTM24.0/

### Datasets and raw sequences
datasets/
*.yuv
*.h5
*.npz

### Python
__pycache__/
*.pyc
.venv/

### Editor / OS
.vscode/
.DS_Store

