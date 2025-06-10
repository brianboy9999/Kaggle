# Digit Recognizer - Kaggle Competition
# 手寫數字識別競賽完整解決方案


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# 設定繪圖風格
plt.style.use('default')
sns.set_palette("husl")

# ================================
# 1. 資料載入和基本探索
# ================================

def load_data():
    """載入訓練和測試資料"""
    print("正在載入資料...")
    train_data = pd.read_csv('Dataset/Digit_Recognizer/train.csv')
    test_data = pd.read_csv('Dataset/Digit_Recognizer/test.csv')
    sample_submission = pd.read_csv('Dataset/Digit_Recognizer/sample_submission.csv')
    
    print(f"訓練資料形狀: {train_data.shape}")
    print(f"測試資料形狀: {test_data.shape}")
    print(f"提交範例形狀: {sample_submission.shape}")
    
    return train_data, test_data, sample_submission

def explore_data(train_data):
    """探索訓練資料"""
    print("\n=== 資料探索 ===")
    print("標籤分布:")
    label_counts = train_data['label'].value_counts().sort_index()
    print(label_counts)
    
    # 視覺化標籤分布
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    label_counts.plot(kind='bar')
    plt.title('數字標籤分布')
    plt.xlabel('數字')
    plt.ylabel('數量')
    
    # 顯示一些範例圖片
    plt.subplot(1, 2, 2)
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i in range(10):
        # 找到每個數字的第一個範例
        digit_data = train_data[train_data['label'] == i].iloc[0, 1:].values
        digit_image = digit_data.reshape(28, 28)
        
        row, col = i // 5, i % 5
        axes[row, col].imshow(digit_image, cmap='gray')
        axes[row, col].set_title(f'數字 {i}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return label_counts

def visualize_samples(train_data, num_samples=20):
    """隨機顯示一些訓練樣本"""
    plt.figure(figsize=(15, 8))
    random_indices = np.random.choice(len(train_data), num_samples, replace=False)
    
    for i, idx in enumerate(random_indices):
        plt.subplot(4, 5, i+1)
        digit_data = train_data.iloc[idx, 1:].values.reshape(28, 28)
        label = train_data.iloc[idx, 0]
        
        plt.imshow(digit_data, cmap='gray')
        plt.title(f'標籤: {label}')
        plt.axis('off')
    
    plt.suptitle('隨機訓練樣本', fontsize=16)
    plt.tight_layout()
    plt.show()

# ================================
# 2. 資料預處理
# ================================

def preprocess_data(train_data, test_data):
    """預處理資料"""
    print("\n=== 資料預處理 ===")
    
    # 分離特徵和標籤
    X = train_data.drop('label', axis=1).values / 255.0  # 正規化到 0-1
    y = train_data['label'].values
    X_test = test_data.values / 255.0
    
    # 分割訓練和驗證集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"訓練集大小: {X_train.shape}")
    print(f"驗證集大小: {X_val.shape}")
    print(f"測試集大小: {X_test.shape}")
    
    return X_train, X_val, y_train, y_val, X_test

# ================================
# 3. 傳統機器學習模型
# ================================

def train_random_forest(X_train, y_train, X_val, y_val):
    """訓練隨機森林模型"""
    print("\n=== 隨機森林模型 ===")
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    
    print("正在訓練隨機森林...")
    rf_model.fit(X_train, y_train)
    
    # 預測和評估
    rf_pred = rf_model.predict(X_val)
    rf_accuracy = accuracy_score(y_val, rf_pred)
    
    print(f"隨機森林驗證準確率: {rf_accuracy:.4f}")
    
    return rf_model, rf_accuracy

def train_svm(X_train, y_train, X_val, y_val):
    """訓練支持向量機模型"""
    print("\n=== SVM 模型 ===")
    
    # 使用較小的資料集進行 SVM 訓練（SVM 比較慢）
    sample_size = 5000
    indices = np.random.choice(len(X_train), sample_size, replace=False)
    X_train_sample = X_train[indices]
    y_train_sample = y_train[indices]
    
    svm_model = SVC(kernel='rbf', random_state=42)
    
    print("正在訓練 SVM（使用樣本資料）...")
    svm_model.fit(X_train_sample, y_train_sample)
    
    # 預測和評估
    svm_pred = svm_model.predict(X_val)
    svm_accuracy = accuracy_score(y_val, svm_pred)
    
    print(f"SVM 驗證準確率: {svm_accuracy:.4f}")
    
    return svm_model, svm_accuracy

# ================================
# 4. PyTorch 深度學習模型
# ================================

class DigitCNN(nn.Module):
    """卷積神經網路模型"""
    def __init__(self):
        super(DigitCNN, self).__init__()
        
        # 卷積層
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 池化層
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # 全連接層
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 10)
        
        # 激活函數
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # 卷積 + 池化
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        # Dropout
        x = self.dropout1(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全連接層
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x

def train_cnn(X_train, y_train, X_val, y_val, epochs=10):
    """訓練 CNN 模型"""
    print("\n=== CNN 深度學習模型 ===")
    
    # 轉換為 PyTorch 格式
    X_train_tensor = torch.FloatTensor(X_train.reshape(-1, 1, 28, 28))
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val.reshape(-1, 1, 28, 28))
    y_val_tensor = torch.LongTensor(y_val)
    
    # 建立資料載入器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    # 建立模型
    model = DigitCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 訓練模型
    train_losses = []
    val_accuracies = []
    
    print("開始訓練 CNN...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # 驗證
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        
        train_losses.append(epoch_loss)
        val_accuracies.append(epoch_acc)
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Val Acc: {epoch_acc:.4f}')
    
    # 繪製訓練過程
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('訓練損失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('驗證準確率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.show()
    
    return model, max(val_accuracies)

# ================================
# 5. 模型比較和預測
# ================================

def compare_models(rf_acc, svm_acc, cnn_acc):
    """比較不同模型的效果"""
    print("\n=== 模型比較 ===")
    
    models = ['Random Forest', 'SVM', 'CNN']
    accuracies = [rf_acc, svm_acc, cnn_acc]
    
    # 建立比較圖表
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'salmon'])
    
    # 在柱狀圖上標示數值
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{acc:.4f}', ha='center', va='bottom')
    
    plt.title('不同模型驗證準確率比較')
    plt.ylabel('準確率')
    plt.ylim(0, 1)
    plt.show()
    
    # 找出最佳模型
    best_idx = np.argmax(accuracies)
    print(f"最佳模型: {models[best_idx]} (準確率: {accuracies[best_idx]:.4f})")
    
    return models[best_idx]

def make_predictions(model, model_type, X_test):
    """使用最佳模型進行預測"""
    print(f"\n=== 使用 {model_type} 進行最終預測 ===")
    
    if model_type == 'CNN':
        model.eval()
        X_test_tensor = torch.FloatTensor(X_test.reshape(-1, 1, 28, 28))
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predictions = torch.max(outputs, 1)
            predictions = predictions.numpy()
    else:
        predictions = model.predict(X_test)
    
    return predictions

def create_submission(predictions, sample_submission):
    """建立提交檔案"""
    submission = sample_submission.copy()
    submission['Label'] = predictions
    
    submission.to_csv('digit_recognizer_submission.csv', index=False)
    print("提交檔案已儲存為 'digit_recognizer_submission.csv'")
    
    return submission

# ================================
# 6. 主程式
# ================================

def main():
    """主程式流程"""
    print("=== Digit Recognizer 競賽解決方案 ===\n")
    
    # 1. 載入資料
    train_data, test_data, sample_submission = load_data()
    
    # 2. 探索資料
    label_counts = explore_data(train_data)
    visualize_samples(train_data)
    
    # 3. 預處理資料
    X_train, X_val, y_train, y_val, X_test = preprocess_data(train_data, test_data)
    
    # 4. 訓練不同模型
    rf_model, rf_acc = train_random_forest(X_train, y_train, X_val, y_val)
    svm_model, svm_acc = train_svm(X_train, y_train, X_val, y_val)
    cnn_model, cnn_acc = train_cnn(X_train, y_train, X_val, y_val, epochs=5)
    
    # 5. 比較模型
    best_model_name = compare_models(rf_acc, svm_acc, cnn_acc)
    
    # 6. 選擇最佳模型進行預測
    if best_model_name == 'Random Forest':
        predictions = make_predictions(rf_model, 'Random Forest', X_test)
    elif best_model_name == 'SVM':
        predictions = make_predictions(svm_model, 'SVM', X_test)
    else:
        predictions = make_predictions(cnn_model, 'CNN', X_test)
    
    # 7. 建立提交檔案
    submission = create_submission(predictions, sample_submission)
    
    print("\n=== 完成！===")
    print("您現在可以將 'digit_recognizer_submission.csv' 上傳到 Kaggle 進行提交。")

if __name__ == "__main__":
    main()
