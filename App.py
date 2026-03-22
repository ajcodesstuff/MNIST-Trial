import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import random

device = torch.device("cpu")

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_data = datasets.MNIST(root="data", train=False, transform=transform, download=True)

def predict_tensor(img_tensor):
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)

    pred = probs.argmax(dim=1).item()
    confidence = probs.max().item()

    return pred, confidence

def predict_image(img):
    img = img.resize((28, 28)).convert("L")
    img = transform(img)
    return predict_tensor(img)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

app = ctk.CTk()
app.geometry("500x500")
app.resizable(False, False)
app.title("MNIST App")

main_frame = ctk.CTkFrame(app)
draw_frame = ctk.CTkFrame(app)
test_frame = ctk.CTkFrame(app)

def show_frame(frame):
    for f in (main_frame, draw_frame, test_frame):
        f.pack_forget()
    frame.pack(fill="both", expand=True)

ctk.CTkLabel(main_frame, text="MNIST App", font=("Arial", 24)).pack(pady=40)

ctk.CTkButton(main_frame, text="Draw Digit",
              command=lambda: show_frame(draw_frame)).pack(pady=10)

ctk.CTkButton(main_frame, text="Test Dataset",
              command=lambda: show_frame(test_frame)).pack(pady=10)

main_frame.pack(fill="both", expand=True)

canvas_size = 280
canvas = tk.Canvas(draw_frame, width=canvas_size, height=canvas_size,
                   bg="black", highlightthickness=0)
canvas.pack(pady=10)

image = Image.new("L", (canvas_size, canvas_size), "black")
draw = ImageDraw.Draw(image)

def draw_lines(event):
    x, y = event.x, event.y
    r = 10
    canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
    draw.ellipse([x-r, y-r, x+r, y+r], fill="white")

canvas.bind("<B1-Motion>", draw_lines)

draw_label = ctk.CTkLabel(draw_frame, text="Draw a digit", font=("Arial", 18))
draw_label.pack(pady=10)

def clear():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_size, canvas_size], fill="black")
    draw_label.configure(text="Draw a digit")

def predict_drawn():
    pred, conf = predict_image(image)
    draw_label.configure(text=f"Prediction: {pred} ({conf*100:.1f}%)")

btn_frame1 = ctk.CTkFrame(draw_frame)
btn_frame1.pack(pady=10)

ctk.CTkButton(btn_frame1, text="Predict", command=predict_drawn).grid(row=0, column=0, padx=5)
ctk.CTkButton(btn_frame1, text="Clear", command=clear).grid(row=0, column=1, padx=5)
ctk.CTkButton(btn_frame1, text="Back",
              command=lambda: show_frame(main_frame)).grid(row=0, column=2, padx=5)

canvas2 = tk.Canvas(test_frame, width=280, height=280,
                    bg="black", highlightthickness=0)
canvas2.pack(pady=10)

test_label = ctk.CTkLabel(test_frame, text="Random MNIST", font=("Arial", 18))
test_label.pack(pady=10)

def show_random():
    global tk_img

    idx = random.randint(0, len(test_data)-1)
    img, label_true = test_data[idx]
    pred, conf = predict_tensor(img)

    img_display = img.squeeze().numpy()
    img_display = (img_display * 255).astype("uint8")
    pil_img = Image.fromarray(img_display).resize((200, 200))

    tk_img = ImageTk.PhotoImage(pil_img)

    canvas2.delete("all")
    canvas2.create_image(140, 140, image=tk_img)

    test_label.configure(text=f"Pred: {pred} ({conf*100:.1f}%) | True: {label_true}")

btn_frame2 = ctk.CTkFrame(test_frame)
btn_frame2.pack(pady=10)

ctk.CTkButton(btn_frame2, text="Next Sample", command=show_random).grid(row=0, column=0, padx=5)
ctk.CTkButton(btn_frame2, text="Back",
              command=lambda: show_frame(main_frame)).grid(row=0, column=1, padx=5)

show_frame(main_frame)
app.mainloop()