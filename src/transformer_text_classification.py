# transformer_text_classification.py - RTX 1660 Super最適化版
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import numpy as np
import time

# RTX 1660 Super向け設定
BATCH_SIZE = 8  # 小さなバッチサイズで6GB VRAM対応
MAX_LENGTH = 128  # 短めの系列長
EPOCHS = 3
LEARNING_RATE = 2e-5

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"使用デバイス: {device}")

# メモリ最適化設定
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(0.85)

# サンプルデータセット
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # トークン化
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# RTX 1660 Super向け軽量分類モデル
class TextClassifier(nn.Module):
    def __init__(self, model_name, num_classes, dropout_rate=0.3):
        super(TextClassifier, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
        # 一部の層を固定してメモリ使用量を削減
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        
        for layer in self.bert.encoder.layer[:6]:  # 最初の6層を固定
            for param in layer.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

# メモリ効率的な訓練関数
def train_with_accumulation(model, dataloader, optimizer, criterion, accumulation_steps=4):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 勾配累積を使用してメモリ使用量を削減
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels) / accumulation_steps
        
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        
        if batch_idx % 10 == 0:
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}, GPU Memory: {allocated:.2f}GB')
    
    return total_loss / len(dataloader)

# サンプルデータの準備
sample_texts = [
    "This movie is amazing!",
    "I hate this film.",
    "Great story and acting.",
    "Boring and predictable.",
    "Love the cinematography.",
    "Not worth watching.",
] * 100  # データを拡張

sample_labels = [1, 0, 1, 0, 1, 0] * 100  # 1: positive, 0: negative

# モデルとトークナイザーの初期化
model_name = "distilbert-base-uncased"  # 軽量版BERT
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TextClassifier(model_name, num_classes=2).to(device)

# データセットとデータローダー
dataset = TextDataset(sample_texts, sample_labels, tokenizer, MAX_LENGTH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 最適化とロス関数
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# 訓練実行
print("=== RTX 1660 Super Transformer Training ===")
print(f"モデル: {model_name}")
print(f"バッチサイズ: {BATCH_SIZE}")
print(f"最大系列長: {MAX_LENGTH}")

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    
    start_time = time.time()
    train_loss = train_with_accumulation(model, dataloader, optimizer, criterion)
    epoch_time = time.time() - start_time
    
    print(f"エポック {epoch+1} - 損失: {train_loss:.4f}, 時間: {epoch_time:.2f}秒")
    
    # メモリクリーンアップ
    torch.cuda.empty_cache()

print("\n訓練完了!")

