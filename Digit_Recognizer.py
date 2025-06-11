import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette('husl')

# 1. 資料匯入
def load_data():
    print("正在載入資料...")
    train_data = pd.read_csv('Dataset/Digit_Recognizer/train.csv')
    test_data = pd.read_csv('Dataset/Digit_Recognizer/test.csv')
    sample_submission = pd.read_csv('Dataset/Digit_Recognizer/sample_submission.csv')

    return train_data, test_data, sample_submission

def explore_data(train_data):
    print("\n=== 資料探索 ===")
    # 製作標籤分布
    print("標籤分布:")
    label_counts = train_data['label'].value_counts().sort_index()
    print(label_counts)

    # 視覺化標籤分布
    plt.figure(figsize = (10, 6))
    plt.subplot(1, 2, 1)
    label_counts.plot(kind = 'bar')
    plt.title('數字標籤分布')
    plt.xlabel('數字')
    plt.ylabel('數量')

    # 顯示範例圖片
    plt.subplot(1, 2, 2)
    fig, axes = plt.subplot(2, 5, figsize = (12, 6))
    for i in range(10):
        digit_data = train_data[train_data['label'] == i].iloc[0, 1:].values
        digit_image = digit_data.reshape(28, 28)

        row, col = i // 5, i % 5
        axes[row, col].imshow(digit_image, cmap = 'gray')
        axes[row, col].set_title(f'數字 {i}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

    return label_counts

def visualize_samples(train_data, num_samples=20):
    # 隨機顯示訓練樣本
    plt.figure(figsize = (15, 8))
    random_indices = np.random.choice(len(train_data), num_samples, replace = False)

    for i, idx in enumerate(random_indices):
        plt.subplot(4, 5, i + 1)
        digit_data = train_data.iloc[idx, 1:].values.reshape(28, 28)
        label = train_data.iloc[idx, 0]

        plt.imshow(digit_data, cmap = 'gray')
        plt.title(f'標籤: {label}')
        plt.axis('off')
    
    plt.suptitle('隨機訓練樣本', fontsize = 16)
    plt.tight_layout()
    plt.show()

# 2. 資料預處理
def processing_data(train_data, test_data):
    print("\n=== 資料預處理 ===")
    X = train_data.drop('label', axis = 1).values / 255.0
    y = train_data['label'].values
    X_test = test_data.values / 255.0

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size = 0.2, random_state = 42, stratify = y
    )

    return X_train, X_val, y_train, y_val, X_test

# 3. 隨機森林模型
def train_random_forest(X_train, y_train, X_val, y_val):
    print("\n=== 隨機森林模型 ===")
    rf_model = RandomForestClassifier(
        n_estimator = 100,
        random_state = 42,
        n_jobs = -1
    )

    print("正在訓練隨機森林...")
    rf_model.fit(X_train, y_train)

    # 預測和評估
    rf_pred = rf_model.predict(X_val)
    rf_accuracy = accuracy_score(y_val, rf_pred)

    print(f"隨機森林驗證準確率: {rf_accuracy:.4f}")

    return rf_model, rf_accuracy

# 4. 支持向量機模型
def train_svm(X_train, y_train, X_val, y_val):
    sample_size = 5000
    indices = np.random.choice(len(X_train), sample_size, replace = False)
    X_train_sample = X_train[indices]
    y_train_sample = y_train[indices]

    svm_model = SVC(kernal = 'rbf', random_state = 42)

    print("正在訓練 SVM（使用樣本資料）...")
    svm_model.fit(X_train_sample, y_train_sample)

    # 預測和評估
    svm_pred = svm_model.predict(X_val)
    svm_accuracy = accuracy_score(y_val, svm_pred)

    print(f"隨機森林驗證準確率: {svm_accuracy:.4f}")

    return svm_model, svm_accuracy

# 5. PyTorch 深度學習模型
class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()

        