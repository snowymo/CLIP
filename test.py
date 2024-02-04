import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def pred(path, prompts):
    image = preprocess(Image.open(path)).unsqueeze(0).to(device)
    text = clip.tokenize(prompts).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print(path,"device",device,"Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

if __name__ == '__main__':
    # 0.4以上
    prompts = ["text covered by human","text covered by pen", "text covered by reflection", "text covered by light"]
    # [[0.4172  0.4653  0.10223 0.0152 ]]
    pred("D:/projects/chatgpt/covered/covered_text_human.jpeg",prompts)

    # [[0.1497  0.1921  0.6104  0.04782]]
    pred("D:/projects/chatgpt/covered/covered_text_reflection.jpeg",prompts)

    # [[0.1113 0.3027 0.5654 0.0206]]
    pred("D:/projects/chatgpt/covered/covered_text_reflection4.jpeg",prompts)

    # [[0.013954 0.919    0.0625   0.004745]]
    pred("D:/projects/chatgpt/covered/covered_text_pen_double.jpeg",prompts)