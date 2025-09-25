# GNN-Transformer: 圖神經網路如何增強表格資料的預測能力
本專題研究在使用圖神經網路（GNN）的幫助下，提升 ExcelFormer 在表格資料上的預測表現。  
此研究主要包括：
- **表格資料的建圖策略**
- **適合的 Mini-batch 訓練方式**
- **GNN 在表格資料中的角色定位**

## 表格資料的建圖策略
本研究設計 Multi-thread Adaptive Mutual kNN 構圖法，  
該演算法能夠快速且有效地為表格資料建構優化的圖結構，  
利於後續 GNN 模組的訊息傳遞與聚合機制。  
演算法細節請參閱 **[graph.py](./graph.py)** 與 **[mtam_knn.pyx](mtam_knn.pyx)**。

## 適合的 Mini-batch 訓練方式
由於模型會同時包含 GNN 與 ExcelFormer 模組，傳統 Mini-batch 訓練無法讓 GNN 依照圖結構進行節點訊息流通；  
另一方面，全圖訓練不只缺乏 Mini-batch 的泛化能力增強，更產生昂貴的記憶體成本。  
因此最終選用 GraphSAINT-RW 的子圖採樣策略，該方法不僅方便調整採樣過程所需的超參數設定，  
也同時保證模型的泛化程度和避免嚴重的記憶體開銷；在訓練速度上，甚至優於原始 ExcelFormer。

## GNN 在表格資料中的角色定位
在模型的架構中，除了一律使用兩層 GCN 作為 GNN 模組，並透過 PyG 的 GraphConv 讓每層 GCN 皆維持自身輸入訊息。  
在跨模組整合上，本研究在此提出三種 GNN 的角色定位：
- **[Encoder](./ENCODER.py)**
- **[Decoder](./DECODER.py)**
- **[Parallel](./PARALLEL.py)**

在數據集上的實驗結果顯示：
- Decoder 架構在二元分類任務上表現突出；
- Parallel 架構則在迴歸與二元分類任務上皆勝過原始 ExcelFormer。

---
**以上更多的研究細節、圖表數據以及實驗結果，請參閱本專題的完整報告**
