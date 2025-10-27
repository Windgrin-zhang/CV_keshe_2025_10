import torch
import torchvision.models as models
import os


def choose_and_download_resnet():
    """
    交互式选择并下载 PyTorch 官方 ResNet 模型，
    自动保存到当前程序同级目录下的 ./model 文件夹中。
    """

    # 模型与权重对应表
    resnet_models = {
        "resnet18":  (models.resnet18,  models.ResNet18_Weights.IMAGENET1K_V1),
        "resnet34":  (models.resnet34,  models.ResNet34_Weights.IMAGENET1K_V1),
        "resnet50":  (models.resnet50,  models.ResNet50_Weights.IMAGENET1K_V1),
        "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V1),
        "resnet152": (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V1),
    }

    # 显示选项
    print("可选择下载的 ResNet 模型：")
    print("1. resnet18\n2. resnet34\n3. resnet50\n4. resnet101\n5. resnet152\n6. 全部下载")
    choice = input("\n请输入要下载的模型编号（如 3 或 1,3,5 或 6 表示全部）：").strip()

    # 映射输入编号到模型名
    options = {
        "1": "resnet18",
        "2": "resnet34",
        "3": "resnet50",
        "4": "resnet101",
        "5": "resnet152"
    }

    if choice == "6":
        selected_models = list(resnet_models.keys())
    else:
        selected_models = [options[c.strip()] for c in choice.split(",") if c.strip() in options]

    if not selected_models:
        print("❌ 未选择任何有效模型，已退出。")
        return

    # 设置保存目录（当前同级文件夹下的 model 目录）
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
    os.makedirs(save_dir, exist_ok=True)

    # 下载与保存
    for name in selected_models:
        model_fn, weights_enum = resnet_models[name]
        print(f"\n🚀 正在下载 {name} 模型（ImageNet 预训练权重）...")
        model = model_fn(weights=weights_enum)
        model.eval()

        file_path = os.path.join(save_dir, f"{name}_imagenet.pth")
        torch.save(model.state_dict(), file_path)
        print(f"✅ {name} 权重已保存至：{file_path}")

    print("\n🎉 所有选择的模型已下载并保存到 ./model 文件夹！")


# ✅ 调用函数
if __name__ == "__main__":
    choose_and_download_resnet()
