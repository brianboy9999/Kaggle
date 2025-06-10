import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette('husl')

# 1. 資料匯入
def load_data():
    train_data = pd.read_csv('Dataset/Digit_Recognizer/train.csv')
    test_data = pd.read_csv('Dataset/Digit_Recognizer/test.csv')
    sample_submission = pd.read_csv('Dataset/Digit_Recognizer/sample_submission.csv')
    return train_data, test_data, sample_submission

def explore_data(train_data):
    # 製作標籤分布
    label_counts = train_data['label'].value_counts().sort_index()

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
        digit_data = train_data[train_data['label'] = i].iloc[0, 1:].values
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
