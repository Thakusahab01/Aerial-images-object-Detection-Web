from flask import Flask, render_template, url_for, request, redirect, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_required as flask_login_required, login_user, logout_user, LoginManager, UserMixin
import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from functools import wraps

# -------------------- FLASK SETUP --------------------
app = Flask(__name__)
app.secret_key = 'secret_key'

# Database config
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Upload & Result folders
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['RESULT_FOLDER'] = os.path.join('static', 'results')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# -------------------- LOGIN SETUP --------------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id       = db.Column(db.Integer, primary_key=True)
    name     = db.Column(db.String(100))
    email    = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(200))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

with app.app_context():
    db.create_all()

# -------------------- CUSTOM LOGIN REQUIRED --------------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in first', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# -------------------- MODEL --------------------
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
    )

class ResNetUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.base_model = models.resnet18(weights=None)
        self.base_layers = list(self.base_model.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer0_1 = nn.Sequential(*self.base_layers[3:5])
        self.layer1, self.layer2, self.layer3 = self.base_layers[5], self.base_layers[6], self.base_layers[7]
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up3 = double_conv(512 + 256, 256)
        self.conv_up2 = double_conv(256 + 128, 128)
        self.conv_up1 = double_conv(128 + 64, 64)
        self.conv_up0 = double_conv(64 + 64, 32)
        self.conv_last = nn.Conv2d(32, n_class, 1)
        
    def forward(self, x):
        l0 = self.layer0(x)
        l0_1 = self.layer0_1(l0)
        l1 = self.layer1(l0_1)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        x = self.upsample(l3)
        x = torch.cat([x, l2], dim=1)
        x = self.conv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, l1], dim=1)
        x = self.conv_up2(x)
        x = self.upsample(x)
        x = F.interpolate(x, size=l0_1.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, l0_1], dim=1)
        x = self.conv_up1(x)
        x = self.upsample(x)
        x = F.interpolate(x, size=l0.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, l0], dim=1)
        x = self.conv_up0(x)
        out = self.upsample(self.conv_last(x))
        return out

# ---------------------------------------------------------
# 2. CONFIGURATION
# ---------------------------------------------------------
CLASS_LABELS = {
    0: "Unlabeled", 1: "Paved Area", 2: "Dirt", 3: "Grass", 
    4: "Gravel", 5: "Water", 6: "Rocks", 7: "Pool", 
    8: "Vegetation", 9: "Roof", 10: "Wall", 11: "Window", 
    12: "Door", 13: "Fence", 14: "Fence Pole", 15: "Person", 
    16: "Dog", 17: "Car", 18: "Bicycle", 19: "Tree", 
    20: "Bald Tree", 21: "Arid Vegetation", 22: "Obstacle"
}

COLOR_MAP = np.array([
    (0, 0, 0), (128, 128, 128), (150, 75, 0), (0, 154, 23), (192, 192, 192),
    (0, 0, 255), (105, 105, 105), (0, 255, 255), (0, 255, 0), (255, 0, 0),
    (165, 42, 42), (0, 191, 255), (255, 165, 0), (218, 165, 32), (184, 134, 11),
    (255, 192, 203), (255, 20, 147), (255, 255, 0), (127, 0, 255), (34, 139, 34),
    (210, 180, 140), (255, 215, 0), (128, 0, 0)
], dtype=np.uint8)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNetUNet(23).to(DEVICE)
if os.path.exists('resnetunet_aerial.pth'):
    sd = torch.load('resnetunet_aerial.pth', map_location=DEVICE)
    model.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()})
    model.eval()

# ---------------------------------------------------------
# 3. FULL RESOLUTION TILING ENGINE
# ---------------------------------------------------------
def predict_full_res_tiled(img_pil, patch_size=512, stride=448):
    """
    Tiles the image at ITS ORIGINAL RESOLUTION. 
    Matches the scale the model was trained on.
    """
    w, h = img_pil.size
    
    img_tensor = transforms.ToTensor()(img_pil).to(DEVICE)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(DEVICE)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(DEVICE)
    img_tensor = (img_tensor - mean) / std

    pad_h = (patch_size - (h - patch_size) % stride) % stride if h > patch_size else patch_size - h
    pad_w = (patch_size - (w - patch_size) % stride) % stride if w > patch_size else patch_size - w
    img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h))
    new_h, new_w = img_tensor.shape[1], img_tensor.shape[2]

    # full_mask stores the labels. Use float to handle max voting if needed.
    full_mask = np.zeros((new_h, new_w), dtype=np.uint8)
    
    # Process patches
    for y in range(0, new_h - patch_size + 1, stride):
        for x in range(0, new_w - patch_size + 1, stride):
            patch = img_tensor[:, y:y+patch_size, x:x+patch_size].unsqueeze(0)
            with torch.no_grad():
                output = model(patch)
                pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                
                # Logic: Don't overwrite useful classes (like Person=15) with background (0/1)
                # We prioritize classes > 4 (objects) over background
                existing = full_mask[y:y+patch_size, x:x+patch_size]
                mask_new = (pred > 4) | (existing <= 4)
                full_mask[y:y+patch_size, x:x+patch_size][mask_new] = pred[mask_new]
                
    return full_mask[:h, :w]

# -------------------- ROUTES --------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'error')
            return redirect(url_for('register'))

        hashed = generate_password_hash(password)
        user = User(name=name, email=email, password=hashed)
        db.session.add(user)
        db.session.commit()

        flash('Registered successfully', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form['email']).first()

        if user and check_password_hash(user.password, request.form['password']):
            login_user(user)
            session['user_id'] = user.id
            return redirect(url_for('upload'))   # CHANGED: redirect to upload after login

        flash('Invalid credentials', 'error')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    return redirect(url_for('login'))

# -------------------- UPLOAD PAGE --------------------
# CHANGED: route renamed from /detector → /upload, function from detector → upload
@app.route('/upload')
@login_required
def upload():
    return render_template('upload.html')

# -------------------- PREDICTION --------------------
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file: return redirect('/')
    
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    img_pil = Image.open(filepath).convert('RGB')
    orig_w, orig_h = img_pil.size
    
    start = time.time()
    # USE FULL RESOLUTION TILING (Matches training scale)
    mask = predict_full_res_tiled(img_pil)
    
    inf_time = round((time.time() - start) * 1000, 2)
    
    # Generate Stats
    unique_labels, counts = np.unique(mask, return_counts=True)
    total_pixels = mask.size
    stats = []
    for lbl, count in zip(unique_labels, counts):
        pct = round((count / total_pixels) * 100, 1)
        if pct < 0.01: continue # Very sensitive for tiny objects
        stats.append({'name': CLASS_LABELS.get(lbl, "Other"), 'pct': pct, 'color': '#%02x%02x%02x' % tuple(COLOR_MAP[lbl])})
    stats = sorted(stats, key=lambda x: x['pct'], reverse=True)

    # Prepare Visuals
    mask_rgb = COLOR_MAP[mask]
    orig_img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(orig_img_cv, 0.6, mask_bgr, 0.4, 0)
    
    cv2.imwrite(os.path.join(app.config['RESULT_FOLDER'], 'mask_' + filename), mask_bgr)
    cv2.imwrite(os.path.join(app.config['RESULT_FOLDER'], 'overlay_' + filename), overlay)
    
    return render_template('predict.html', filename=filename, time=inf_time, stats=stats)

@app.route('/upload')
def upload():
    return render_template('upload.html')

# -------------------- RUN --------------------
if __name__ == '__main__':
    app.run(debug=True)