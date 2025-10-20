import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import timm
import streamlit as st
import os


st.set_page_config(page_title="Emotion Ensemble Recognition", layout="centered")
st.title("😊 Emotion Recognition using Ensemble Models")
st.write("Upload an image or capture via webcam. The app uses an ensemble of 9 models for emotion detection.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


models_path = "/content/drive/MyDrive/models"
num_classes = 7
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=(2,3), keepdim=True)
        max_out, _ = torch.max(x, dim=2, keepdim=True)
        max_out, _ = torch.max(max_out, dim=3, keepdim=True)
        avg_out = self.fc2(self.relu1(self.fc1(avg_out)))
        max_out = self.fc2(self.relu1(self.fc1(max_out)))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BAM(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x_out = self.ca(x) * x
        x_out = self.sa(x_out) * x_out
        return x_out

class CBAMBlock(nn.Module):
    def __init__(self, channels, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class BAM_ResNet50(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.resnet = models.resnet50(weights=None)
        self.bam1, self.bam2, self.bam3, self.bam4 = BAM(256), BAM(512), BAM(1024), BAM(2048)
        self.resnet.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x); x = self.bam1(x)
        x = self.resnet.layer2(x); x = self.bam2(x)
        x = self.resnet.layer3(x); x = self.bam3(x)
        x = self.resnet.layer4(x); x = self.bam4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return self.resnet.fc(x)

class CBAM_ResNet50(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.resnet = models.resnet50(weights=None)
        self.cbam1, self.cbam2, self.cbam3, self.cbam4 = CBAMBlock(256), CBAMBlock(512), CBAMBlock(1024), CBAMBlock(2048)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x); x = self.cbam1(x)
        x = self.resnet.layer2(x); x = self.cbam2(x)
        x = self.resnet.layer3(x); x = self.cbam3(x)
        x = self.resnet.layer4(x); x = self.cbam4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return self.resnet.fc(x)


@st.cache_resource
def load_all_models():
    def safe_load(model, path, num_classes=7):
        # Adjust classifier layer if needed before loading
        if isinstance(model, models.ResNet):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif isinstance(model, models.DenseNet):
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        elif isinstance(model, models.VGG):
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        elif isinstance(model, models.GoogLeNet):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif isinstance(model, models.Inception3):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif "efficientnet" in model.__class__.__name__.lower():
            # for timm models like EfficientNet-B2
            in_features = model.get_classifier().in_features
            model.classifier = nn.Linear(in_features, num_classes)

        # Load checkpoint safely
        state = torch.load(path, map_location=device)
        model.load_state_dict(state, strict=False)
        model.to(device).eval()
        return model

    models_dict = {
        "ResNet34": safe_load(models.resnet34(weights=None), f"{models_path}/resnet34.pth"),
        "ResNet152": safe_load(models.resnet152(weights=None), f"{models_path}/resnet152.pth"),
        "DenseNet121": safe_load(models.densenet121(weights=None), f"{models_path}/densenet121.pth"),
        "EfficientNet-B2": safe_load(timm.create_model("efficientnet_b2", pretrained=False, num_classes=num_classes), f"{models_path}/efficientnet_b2.pth"),
        "GoogLeNet": safe_load(models.googlenet(weights=None), f"{models_path}/googlenet.pth"),
        "InceptionV3": safe_load(models.inception_v3(weights=None, aux_logits=False), f"{models_path}/inception_v3.pth"),
        "VGG19": safe_load(models.vgg19(weights=None), f"{models_path}/vgg19.pth"),
        "BAM-ResNet50": safe_load(BAM_ResNet50(num_classes), f"{models_path}/bam_resnet50.pth"),
        "CBAM-ResNet50": safe_load(CBAM_ResNet50(num_classes), f"{models_path}/cbam_resnet50.pth"),
    }

    return models_dict


models_dict = load_all_models()
st.success("All ensemble models loaded successfully.")

transform_224 = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_299 = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


import pandas as pd

def ensemble_predict(image):
    results = {}
    for name, model in models_dict.items():
        transform = transform_299 if "Inception" in name else transform_224
        img_t = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_t)
            probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
            pred_idx = probs.argmax()
            results[name] = {
                "emotion": class_names[pred_idx],
                "confidence": probs[pred_idx] * 100,
                "all_probs": probs
            }

    # Ensemble voting
    votes = [res["emotion"] for res in results.values()]
    final_emotion = max(set(votes), key=votes.count)

    return final_emotion, results


option = st.radio("Choose Input Mode:", ("Upload Image", "Use Webcam"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Predict Emotion"):
          final_emotion, results = ensemble_predict(image)
          st.success(f"Final Detected Emotion: {final_emotion}")

          # Show confidence from each model
          st.subheader("Model-wise Predictions")
          table_data = []
          for name, res in results.items():
              table_data.append({
                  "Model": name,
                  "Predicted Emotion": res["emotion"],
                  "Confidence (%)": f"{res['confidence']:.2f}"
              })
          df = pd.DataFrame(table_data)
          st.dataframe(df)


          st.subheader("Confidence Distribution (Average across models)")
          avg_probs = sum([torch.tensor(r["all_probs"]) for r in results.values()]) / len(results)
          avg_probs = avg_probs.numpy()
          avg_df = pd.DataFrame({
              "Emotion": class_names,
              "Average Confidence (%)": avg_probs * 100
          })
          st.bar_chart(avg_df.set_index("Emotion"))


elif option == "Use Webcam":
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
        st.image(image, caption="Captured Image", use_column_width=True)
        if st.button("Predict Emotion"):
            final_emotion, results = ensemble_predict(image)
            st.success(f"Final Detected Emotion: {final_emotion}")

            # Show confidence from each model
            st.subheader("Model-wise Predictions")
            table_data = []
            for name, res in results.items():
                table_data.append({
                    "Model": name,
                    "Predicted Emotion": res["emotion"],
                    "Confidence (%)": f"{res['confidence']:.2f}"
                })
            df = pd.DataFrame(table_data)
            st.dataframe(df)


            st.subheader("Confidence Distribution (Average across models)")
            avg_probs = sum([torch.tensor(r["all_probs"]) for r in results.values()]) / len(results)
            avg_probs = avg_probs.numpy()
            avg_df = pd.DataFrame({
                "Emotion": class_names,
                "Average Confidence (%)": avg_probs * 100
            })
            st.bar_chart(avg_df.set_index("Emotion"))


st.markdown("---")
st.caption("HIT 401 - Ensemble Emotion Recognition (Colab + Streamlit + ngrok)")

