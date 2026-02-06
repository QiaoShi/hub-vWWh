import torch
from transformers import BertTokenizer, BertForSequenceClassification
import joblib

# ==================================
#         模型预测脚本
# ==================================

def predict_text(text_sample):
    """
    加载训练好的模型和LabelEncoder，对输入的文本进行分类预测。
    
    参数:
    text_sample (str): 需要进行分类的文本。
    
    返回:
    str: 预测的类别标签。
    """
    print("正在加载模型和必要的组件...")
    
    # 定义模型和LabelEncoder的路径
    model_path = "./results/checkpoint-25"
    label_encoder_path = "./results/label_encoder.joblib"
    
    try:
        # 加载训练好的模型和分词器
        # from_pretrained 会自动加载目录下的 config.json 和 model.safetensors
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        
        # 加载保存的 LabelEncoder
        lbl = joblib.load(label_encoder_path)
    except Exception as e:
        print(f"加载模型或LabelEncoder时出错: {e}")
        print("请确保您已经运行了训练脚本，并且'./results'目录下包含了模型文件和'label_encoder.joblib'。")
        return None

    # 将模型设置为评估模式
    model.eval()
    
    print(f"\n准备预测新样本: '{text_sample}'")

    # 使用分词器对新样本进行编码
    # return_tensors="pt" 表示返回 PyTorch 张量
    inputs = tokenizer(text_sample, return_tensors="pt", padding=True, truncation=True, max_length=64)

    # 不需要计算梯度
    with torch.no_grad():
        # 模型前向传播，得到 logits
        outputs = model(**inputs)
        logits = outputs.logits

    # 从 logits 中找到概率最高的类别索引
    predicted_class_id = torch.argmax(logits, dim=1).item()

    # 使用 LabelEncoder 将数字标签转换回原始文本标签
    predicted_label = lbl.inverse_transform([predicted_class_id])[0]
    
    return predicted_label

# --- 主程序入口 ---
if __name__ == '__main__':
    # 在这里输入您想测试的任何新样本
    new_sample = "茶语时光经典锡兰红茶饮料500ml"
    
    # 调用预测函数
    predicted_category = predict_text(new_sample)
    
    if predicted_category:
        print(f"\n✅ 预测完成！")
        print(f"模型预测的类别是: {predicted_category}")

    # 您也可以测试其他样本
    # new_sample_2 = "这件衣服的料子很舒服，就是有点小"
    # predicted_category_2 = predict_text(new_sample_2)
    # if predicted_category_2:
    #     print(f"\n✅ 预测完成！")
    #     print(f"模型预测的类别是: {predicted_category_2}")
