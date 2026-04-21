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
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class ResNetUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.base_model = models.resnet18(pretrained=False)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer0_1 = nn.Sequential(*self.base_layers[3:5])
        self.layer1 = self.base_layers[5]
        self.layer2 = self.base_layers[6]
        self.layer3 = self.base_layers[7]

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

        return self.upsample(self.conv_last(x))

# -------------------- LOAD MODEL --------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNetUNet(23).to(DEVICE)

if os.path.exists('resnetunet_aerial.pth'):
    sd = torch.load('resnetunet_aerial.pth', map_location=DEVICE)
    model.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()})
    model.eval()

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

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
@login_required
def predict():
    file = request.files.get('file')
    if not file:
        return redirect('/upload')   # CHANGED: redirect to /upload if no file

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    img_pil = Image.open(filepath).convert('RGB')
    orig_w, orig_h = img_pil.size

    input_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

    start = time.time()
    with torch.no_grad():
        output = model(input_tensor)
        mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    inf_time = round((time.time() - start) * 1000, 2)

    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    mask_img = cv2.resize(mask_rgb, (orig_w, orig_h))

    overlay = cv2.addWeighted(
        cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR),
        0.6,
        mask_img,
        0.4,
        0
    )

    cv2.imwrite(os.path.join(app.config['RESULT_FOLDER'], 'overlay_' + file.filename), overlay)

    return render_template('predict.html',
                           filename=file.filename,
                           time=inf_time)

# -------------------- RUN --------------------
if __name__ == '__main__':
    app.run(debug=True)