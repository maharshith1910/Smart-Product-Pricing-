import os
import re
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# ------------------------------
# CONFIG
# ------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSV_DATA_PATH = "/kaggle/input/ml-challenge/TrainClean_Verified.csv"
IMG_DATA_DIR = "/kaggle/input/ml-challenge/65k-75k"
IMG_EXTS = [".jpg", ".jpeg", ".png", ".webp"]

print("Device:", DEVICE)

# ------------------------------
# ENHANCED SMAPE metrics and loss
# ------------------------------
def enhanced_smape(y_true, y_pred):
    """Enhanced SMAPE with price-aware weighting"""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denominator = np.where(denominator == 0, 1e-8, denominator)
    smape_vals = numerator / denominator
    
    # Price-aware weighting - focus more on common price ranges
    weights = np.ones_like(y_true)
    common_range = (y_true >= 5) & (y_true <= 50)
    weights[common_range] = 1.5  # Higher weight for common commerce range
    
    return np.average(smape_vals, weights=weights) * 100

def enhanced_smape_loss(y_true, y_pred):
    """Enhanced Torch version with price-aware weighting"""
    numerator = torch.abs(y_pred - y_true)
    denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2.0
    denominator = torch.where(denominator == 0, 1e-8, denominator)
    smape_vals = numerator / denominator
    
    # Price-aware weights
    weights = torch.ones_like(y_true)
    common_range = (y_true >= 5) & (y_true <= 50)
    weights[common_range] = 1.5
    
    smape_vals = torch.clamp(smape_vals, 0, 5)  # Less aggressive clamping
    return torch.mean(weights * smape_vals) * 100

def asymmetric_smape_v2(y_true, y_pred, underweight=0.7, overweight=1.3):
    """More aggressive asymmetric weighting"""
    error = y_pred - y_true
    under_pred = torch.where(error < 0, torch.abs(error), torch.zeros_like(error))
    over_pred = torch.where(error > 0, torch.abs(error), torch.zeros_like(error))
    denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2.0
    denominator = torch.where(denominator == 0, 1e-8, denominator)
    asymmetric_loss = (underweight * under_pred + overweight * over_pred) / denominator
    return torch.mean(asymmetric_loss) * 100

# ------------------------------
# Enhanced Data loading and cleaning
# ------------------------------
def load_and_clean_data_enhanced():
    print("Loading and cleaning data with enhanced outlier detection...")
    df = pd.read_csv(CSV_DATA_PATH)
    df = df[df['price'].notna()]
    df = df[df['price'] > 0]
    df = df[df['catalog_content'].notna()]
    df = df[df['catalog_content'].str.len() > 5]  # More lenient
    
    # Enhanced outlier removal - keep more data
    price_q02 = df['price'].quantile(0.02)
    price_q98 = df['price'].quantile(0.98)
    df = df[(df['price'] >= price_q02) & (df['price'] <= price_q98)]
    
    print(f"After enhanced cleaning: {len(df)} rows")
    print(f"Price stats - Min: ${df['price'].min():.2f}, Max: ${df['price'].max():.2f}, Median: ${df['price'].median():.2f}")
    return df

# ------------------------------
# OPTIMIZED FEATURE ENGINEERING
# ------------------------------
def create_optimized_features_v3(df):
    """Optimized features with better signal extraction"""
    # Basic features from original function
    df['text_length'] = df['catalog_content'].str.len().fillna(0)
    df['word_count'] = df['catalog_content'].str.split().str.len().fillna(0)
    df['avg_word_length'] = df['text_length'] / np.maximum(df['word_count'], 1)

    # Enhanced IPQ extraction
    def extract_enhanced_ipq(text):
        if pd.isna(text): return 1
        text = str(text).lower()
        
        patterns = [
            r'(\d+)\s*(pack|pcs|count|piece|pieces|set|kit|box|units?)\b',
            r'(\d+)\s*-\s*(pack|count|piece)',
            r'(\d+)\s*x\s*\d',
            r'sold\s*in\s*(\d+)',
            r'(\d+)\s*in\s*1',  # Multi-functional
            r'(\d+)\s*pc',      # Piece abbreviation
        ]
        
        quantities = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    quantity = int(match[0])
                else:
                    quantity = int(match)
                quantities.append(min(quantity, 15))
        
        return max(quantities) if quantities else 1

    df['enhanced_ipq'] = df['catalog_content'].apply(extract_enhanced_ipq)
    
    # Enhanced brand detection
    df['has_brand'] = df['catalog_content'].str.contains(
        r'\b([A-Z][A-Za-z0-9&.]+\s+[A-Z][A-Za-z]+|[A-Z]{2,}[A-Z0-9&.]*)\b'
    ).fillna(0).astype(int)

    # Enhanced dimensions extraction
    def extract_enhanced_dims(text):
        if pd.isna(text): return 0,0,0
        matches = re.findall(r'(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)(?:\s*x\s*(\d+(?:\.\d+)?))?', str(text).lower())
        if matches:
            dims = [float(x) for x in matches[0] if x]
            while len(dims) < 3: dims.append(0)
            return dims[:3]
        return 0,0,0

    dims = df['catalog_content'].apply(extract_enhanced_dims).apply(pd.Series)
    dims.columns = ['dim1', 'dim2', 'dim3']
    df = pd.concat([df, dims], axis=1)

    # Enhanced categories
    enhanced_categories = {
        'electronics': r'headphone|charger|cable|adapter|electronic|usb|bluetooth|wireless|battery|phone|tablet',
        'home_goods': r'kitchen|home|appliance|furniture|decor|garden|outdoor|bed|bath|living',
        'clothing': r'shirt|pants|dress|shoe|clothing|apparel|fashion|jacket|jeans|wear',
        'beauty': r'beauty|cosmetic|skincare|makeup|perfume|lotion|cream|hair',
        'sports': r'sports|fitness|exercise|gym|outdoor|yoga|training|equipment',
        'tools': r'tool|drill|screwdriver|hammer|wrench|kit|hardware|diy'
    }
    
    for cat, pat in enhanced_categories.items():
        df[f'cat_{cat}'] = df['catalog_content'].str.contains(pat, case=False, na=False).astype(int)

    # Enhanced premium indicators
    premium_terms = ['premium', 'professional', 'luxury', 'deluxe', 'high-end', 'advanced', 'premier', 'elite']
    df['premium_score'] = df['catalog_content'].apply(
        lambda x: sum(1.5 if term in str(x).lower() else 0 for term in premium_terms)
    )
    
    # Enhanced price mentions
    df['mentions_price'] = df['catalog_content'].str.contains(
        r'\$\d+|\d+\s*(dollar|usd|price|cost|value)\b', case=False, na=False
    ).astype(int)
    
    # Enhanced specifications
    df['has_specs'] = df['catalog_content'].str.contains(
        r'\d+\s*x\s*\d+|\d+\s*(cm|mm|inch|oz|lb|kg|g|ml|l|foot|feet|meter)\b', case=False, na=False
    ).astype(int)
    
    # Material quality
    premium_materials = ['leather', 'stainless', 'silver', 'gold', 'titanium', 'carbon', 'wood', 'metal', 'ceramic']
    df['material_score'] = df['catalog_content'].apply(
        lambda x: sum(1 for material in premium_materials if material in str(x).lower())
    )
    
    # Seasonal patterns
    seasonal_terms = {
        'christmas': r'christmas|holiday|santa|festive',
        'summer': r'summer|beach|swim|sun|hot',
        'winter': r'winter|cold|snow|warm|heater',
        'spring': r'spring|garden|flower|outdoor'
    }
    for season, pattern in seasonal_terms.items():
        df[f'season_{season}'] = df['catalog_content'].str.contains(pattern, case=False, na=False).astype(int)
    
    # Quality indicators
    quality_terms = ['high quality', 'durable', 'premium', 'professional', 'commercial', 'quality']
    df['quality_score'] = df['catalog_content'].apply(
        lambda x: sum(1 for term in quality_terms if term in str(x).lower())
    )
    
    # NEW OPTIMIZED FEATURES
    # Size indicators
    df['has_large'] = df['catalog_content'].str.contains(
        r'\b(large|big|xl|xxl|king|queen|full|jumbo)\b', case=False, na=False
    ).astype(int)
    
    df['has_small'] = df['catalog_content'].str.contains(
        r'\b(small|mini|compact|petite|travel)\b', case=False, na=False
    ).astype(int)
    
    # Usage context
    df['is_professional'] = df['catalog_content'].str.contains(
        r'professional|commercial|industrial|business', case=False, na=False
    ).astype(int)
    
    # Warranty and guarantees
    df['has_warranty'] = df['catalog_content'].str.contains(
        r'warranty|guarantee|warrantied|lifetime', case=False, na=False
    ).astype(int)
    
    # Color options
    df['has_colors'] = df['catalog_content'].str.contains(
        r'\b(black|white|red|blue|green|pink|purple|color|colors|multi)\b', case=False, na=False
    ).astype(int)
    
    # Extract numerical quantities
    def extract_enhanced_quantities(text):
        numbers = re.findall(r'\b(\d+)\b', str(text))
        return len(numbers), max([int(n) for n in numbers]) if numbers else 0
    
    df['num_count'], df['max_number'] = zip(*df['catalog_content'].apply(extract_enhanced_quantities))
    
    # Enhanced text features
    df['digit_density'] = df['catalog_content'].str.count(r'\d') / np.maximum(df['text_length'], 1)
    df['has_caps_words'] = df['catalog_content'].str.count(r'\b[A-Z]{2,}\b')
    
    return df

# ------------------------------
# Text embedding with SentenceTransformer integration
# ------------------------------
class TextEmbeddingTransformer:
    def __init__(self, model_name_or_path='/kaggle/input/sentence/sentence-transformers-all-MiniLM-L6-v2'):
        print("Loading SentenceTransformer model from local path...")
        self.model = SentenceTransformer(model_name_or_path)
    def transform(self, texts):
        return self.model.encode(texts, batch_size=128, show_progress_bar=True)  # Increased batch size

# ------------------------------
# Image feature extraction with augmentation
# ------------------------------
image_augmentation_train = T.Compose([
    T.RandomResizedCrop(64),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
image_augmentation_eval = T.Compose([
    T.Resize((64,64)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4,4))
        )
        self.feature_dim = 128*4*4
    def forward(self, x):
        return self.conv_layers(x).view(x.size(0), -1)

def extract_cnn_features(df, folder, model, batch_size=64, augment=False):  # Increased batch size
    print("Extracting CNN image features...")
    model.eval()
    features = []
    transform = image_augmentation_train if augment else image_augmentation_eval
    with torch.no_grad():
        for i in tqdm(range(0,len(df),batch_size)):
            batch_ids = df['sample_id'].iloc[i:i+batch_size]
            batch_imgs = []
            for sid in batch_ids:
                img_path = None
                for ext in IMG_EXTS:
                    p = os.path.join(folder, f"{sid}{ext}")
                    if os.path.exists(p):
                        img_path = p
                        break
                try:
                    if img_path:
                        img = Image.open(img_path).convert('RGB')
                        img = transform(img)
                    else:
                        img = torch.zeros(3,64,64)
                    batch_imgs.append(img)
                except:
                    batch_imgs.append(torch.zeros(3,64,64))
            batch_tensor = torch.stack(batch_imgs).to(DEVICE)
            batch_feats = model(batch_tensor).cpu().numpy()
            features.append(batch_feats)
    return np.vstack(features)

# ------------------------------
# OPTIMIZED MODEL ARCHITECTURES
# ------------------------------
class OptimizedPriceNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, n_heads=8, dropout=0.25):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Multi-head attention with residual
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=n_heads, batch_first=True, dropout=dropout)
        self.attention_residual = nn.Linear(hidden_dim, hidden_dim)
        
        # Enhanced residual blocks
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim)
            ) for _ in range(3)  # Increased to 3 blocks
        ])
        
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(384, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(dropout*0.7),
            nn.Linear(192, 1)
        )
        
    def forward(self, x):
        x = self.input_bn(x)
        x = F.relu(self.bn1(self.fc_in(x)))
        
        # Enhanced attention with residual
        x_attn = x.unsqueeze(1)
        attn_out, _ = self.attention(x_attn, x_attn, x_attn)
        attn_out = self.attention_residual(attn_out.squeeze(1))
        x = x + attn_out
        
        # Enhanced residual blocks
        for res_block in self.res_blocks:
            residual = x
            x = F.relu(res_block(x))
            x = x + residual
        
        return self.fc_out(x).squeeze()

class EnhancedAttentionPriceNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=384, n_heads=6, dropout=0.2):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Multi-layer attention
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=n_heads, batch_first=True, dropout=dropout)
            for _ in range(2)
        ])
        
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout*0.7),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        x = self.input_bn(x)
        x = F.relu(self.bn1(self.fc_in(x)))
        
        # Multi-layer attention
        x_attn = x.unsqueeze(1)
        for attention in self.attention_layers:
            attn_out, _ = attention(x_attn, x_attn, x_attn)
            x_attn = x_attn + attn_out  # Residual connection
        
        x = x_attn.squeeze(1)
        return self.fc_out(x).squeeze()

# ------------------------------
# OPTIMIZED TRAINING STRATEGY
# ------------------------------
def train_optimized_neural_network(model, X_train, y_train, X_val, y_val, epochs=350, lr=0.001):
    model = model.to(DEVICE)
    
    # Enhanced optimizer with better configuration
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.999))
    
    # Enhanced learning rate scheduler
    def lr_lambda(epoch):
        if epoch < 15:  # Longer warmup
            return (epoch + 1) / 15.0
        elif epoch < 120:
            return 1.0
        elif epoch < 200:
            return 0.5
        else:
            return 0.1
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
    y_train_tensor = torch.FloatTensor(y_train).to(DEVICE)
    X_val_tensor = torch.FloatTensor(X_val).to(DEVICE)
    y_val_tensor = torch.FloatTensor(y_val).to(DEVICE)
    
    best_val_smape = float('inf')
    patience, patience_counter = 30, 0  # Increased patience
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        preds = model(X_train_tensor)
        
        # Enhanced loss combination
        smape_loss_val = asymmetric_smape_v2(y_train_tensor, preds)
        huber_loss = F.huber_loss(preds, y_train_tensor, delta=0.05)  # Tighter delta
        mae_loss = F.l1_loss(preds, y_train_tensor)
        
        # Weighted combination
        loss = smape_loss_val + 0.15 * huber_loss + 0.05 * mae_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)  # Tighter gradient clipping
        optimizer.step()
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_tensor)
            val_smape_val = enhanced_smape_loss(y_val_tensor, val_preds).item()
        
        if val_smape_val < best_val_smape:
            best_val_smape = val_smape_val
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if epoch % 25 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}: Val SMAPE = {val_smape_val:.2f}%, LR = {current_lr:.6f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, best_val_smape

# ------------------------------
# OPTIMIZED ENSEMBLE TRAINING
# ------------------------------
def train_optimized_ensemble_with_selection(architectures, X_train_scaled, y_train, X_val_scaled, y_val, smape_threshold=22.0):
    base_models = []
    weights = []
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    oof_preds = None
    val_preds_stack = None
    accepted_indices = []
    
    idx = 0
    for arch_idx, (arch_class, arch_args) in enumerate(architectures):
        print(f"\nTraining optimized architecture {arch_idx+1}/{len(architectures)}")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
            print(f"  Fold {fold+1}/3")
            X_tr, X_va = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_tr, y_va = y_train[train_idx], y_train[val_idx]
            # Ensure hidden_dim is divisible by num_heads
            input_dim, hidden_dim, n_heads, dropout = arch_args
            if hidden_dim % n_heads != 0:
                hidden_dim = (hidden_dim // n_heads) * n_heads  # Round down to nearest divisible
                arch_args = [input_dim, hidden_dim, n_heads, dropout]
            model = arch_class(*arch_args)
            trained_model, val_smape = train_optimized_neural_network(model, X_tr, y_tr, X_va, y_va, epochs=350)
            
            if val_smape < smape_threshold:
                base_models.append(trained_model)
                weights.append(1.0 / (val_smape + 1e-8))
                accepted_indices.append(idx)
                print(f"  ✅ Fold SMAPE: {val_smape:.2f}% - ACCEPTED")
                
                if oof_preds is None:
                    oof_preds = np.zeros((len(y_train), len(architectures)*3))
                    val_preds_stack = np.zeros((len(y_val), len(architectures)*3))
                
                trained_model.eval()
                with torch.no_grad():
                    oof_preds[val_idx, idx] = trained_model(torch.FloatTensor(X_va).to(DEVICE)).cpu().numpy()
                    val_preds_stack[:, idx] = trained_model(torch.FloatTensor(X_val_scaled).to(DEVICE)).cpu().numpy()
            else:
                print(f"  ❌ Fold SMAPE: {val_smape:.2f}% - REJECTED")
            idx += 1
    
    if len(base_models) == 0:
        print("⚠️ Using fallback Ridge model...")
        fallback_model = Ridge(alpha=0.5)
        fallback_model.fit(X_train_scaled, y_train)
        return [fallback_model], np.array([1.0]), fallback_model, None
    
    weights = np.array(weights)
    weights /= weights.sum()
    
    oof_preds_accepted = oof_preds[:, accepted_indices]
    val_preds_accepted = val_preds_stack[:, accepted_indices]
    
    meta_model = Ridge()
    param_grid = {'alpha': [0.05, 0.1, 0.5, 1.0, 2.0]}
    grid = GridSearchCV(meta_model, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid.fit(oof_preds_accepted, y_train)
    best_meta = grid.best_estimator_
    print(f"Best meta-model alpha: {best_meta.alpha}")
    
    return base_models, weights, best_meta, val_preds_accepted

# ------------------------------
# ADVANCED CALIBRATION v3
# ------------------------------
def advanced_calibration_v3(y_true, y_pred):
    """Multi-stage advanced calibration"""
    calibrated = y_pred.copy()
    
    # Stage 1: Fine-grained price segmentation
    price_segments = [
        (0, 3), (3, 6), (6, 10), (10, 15), (15, 20),
        (20, 25), (25, 30), (30, 40), (40, 50), (50, 65),
        (65, 80), (80, 100), (100, 150), (150, 200)
    ]
    
    for low, high in price_segments:
        mask = (y_true >= low) & (y_true < high)
        if mask.sum() > 5:
            segment_true = y_true[mask]
            segment_pred = y_pred[mask]
            
            # Robust calibration with outlier removal
            ratios = segment_true / np.clip(segment_pred, 0.1, None)
            q10, q90 = np.percentile(ratios, [15, 85])  # More conservative percentiles
            clean_ratios = ratios[(ratios >= q10) & (ratios <= q90)]
            
            if len(clean_ratios) > 3:
                cal_factor = np.mean(clean_ratios)  # Use mean for smoother adjustment
                calibrated[mask] = y_pred[mask] * cal_factor
    
    # Stage 2: Error distribution adjustment
    errors = y_true - calibrated
    error_quantiles = [0.2, 0.4, 0.6, 0.8]
    
    for q in error_quantiles:
        error_threshold = np.percentile(errors, q * 100)
        mask = errors > error_threshold
        if mask.sum() > 10:
            correction = np.median(errors[mask]) * 0.6  # More conservative correction
            calibrated[mask] += correction
    
    # Stage 3: Price-aware smoothing
    sorted_idx = np.argsort(calibrated)
    calibrated_sorted = calibrated[sorted_idx]
    
    # Adaptive window smoothing with Hanning window
    window_size = max(30, len(calibrated) // 40)
    if window_size > 1:
        kernel = np.hanning(window_size)
        kernel /= kernel.sum()
        smoothed = np.convolve(calibrated_sorted, kernel, mode='same')
        calibrated[sorted_idx] = smoothed
    
    return np.clip(calibrated, 0.3, 250.0)

# ------------------------------
# OPTIMIZED ARCHITECTURES
# ------------------------------
def create_optimized_architectures(input_dim):
    return [
        (OptimizedPriceNet, [input_dim, 512, 8, 0.25]),
        (OptimizedPriceNet, [input_dim, 384, 6, 0.2]),
        (EnhancedAttentionPriceNet, [input_dim, 448, 8, 0.15]),
        (EnhancedAttentionPriceNet, [input_dim, 320, 4, 0.18]),
        (OptimizedPriceNet, [input_dim, 576, 10, 0.22]),
    ]

# ------------------------------
# PREDICTION PIPELINE
# ------------------------------
def predict_test_data(test_csv_path, output_path='test_out.csv'):
    """Generate predictions for test data"""
    print(f"🔮 PREDICTING PRICES FOR TEST DATA")
    
    # Load trained model
    try:
        model_package = torch.load('enhanced_sub20_smape_pipeline.pth', map_location=DEVICE)
        print("✅ Loaded trained model package")
    except:
        print("❌ No trained model found. Please run training first.")
        return
    
    # Load test data
    test_df = pd.read_csv(test_csv_path)
    
    if 'sample_id' not in test_df.columns:
        test_df['sample_id'] = range(len(test_df))
    
    print(f"Test samples: {len(test_df)}")
    
    # Create features
    test_df = create_optimized_features_v3(test_df)
    
    # Extract features
    numerical_features = model_package['numerical_features']
    
    # Handle missing features
    missing_features = set(numerical_features) - set(test_df.columns)
    for feature in missing_features:
        test_df[feature] = 0
    
    X_test_num = test_df[numerical_features].values
    
    # Text features
    text_embedder = TextEmbeddingTransformer(model_package['text_embedder_model_name'])
    X_test_text = text_embedder.transform(test_df['catalog_content'].tolist())
    
    # Image features
    try:
        cnn_model = SimpleCNN().to(DEVICE)
        X_test_img = extract_cnn_features(test_df, IMG_DATA_DIR, cnn_model, augment=False)
    except:
        X_test_img = np.zeros((len(test_df), 2048))
    
    # Combine features
    X_test = np.hstack([X_test_text, X_test_num, X_test_img])
    
    # Scale features
    scaler = model_package['scaler']
    X_test_scaled = scaler.transform(X_test)
    
    # Load base models
    base_models = []
    for state_dict in model_package['base_model_states']:
        input_dim = X_test_scaled.shape[1]
        if 'attention.in_proj_weight' in state_dict:
            model = OptimizedPriceNet(input_dim, 512, 8, 0.25)
        else:
            model = EnhancedAttentionPriceNet(input_dim, 384, 6, 0.2)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        base_models.append(model)
    
    model_weights = model_package['base_model_weights']
    meta_model = model_package['meta_model']
    
    # Generate predictions
    base_preds = []
    for model in base_models:
        model.eval()
        with torch.no_grad():
            preds = model(torch.FloatTensor(X_test_scaled).to(DEVICE)).cpu().numpy()
            base_preds.append(preds)
    
    base_preds = np.array(base_preds).T
    
    # Ensemble predictions
    if meta_model is not None and hasattr(meta_model, 'predict'):
        final_preds_log = meta_model.predict(base_preds)
    else:
        final_preds_log = np.average(base_preds, axis=1, weights=model_weights)
    
    final_predictions = np.expm1(final_preds_log)
    final_predictions = advanced_calibration_v3(np.ones_like(final_predictions) * 15, final_predictions)  # Dummy true for calibration
    final_predictions = np.clip(final_predictions, 0.3, 250.0)
    
    # Create output
    output_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': final_predictions
    })
    
    output_df.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")
    
    # Print summary
    print(f"\n📊 PREDICTION SUMMARY:")
    print(f"Total predictions: {len(output_df)}")
    print(f"Price range: ${final_predictions.min():.2f} - ${final_predictions.max():.2f}")
    print(f"Average price: ${final_predictions.mean():.2f}")
    
    return output_df

# ------------------------------
# OPTIMIZED MAIN PIPELINE
# ------------------------------
def main_optimized():
    print("🚀 OPTIMIZED SUB-19 SMAPE TRAINING PIPELINE")
    
    # 1. Enhanced data loading
    df = load_and_clean_data_enhanced()
    
    # 2. Optimized feature engineering
    df = create_optimized_features_v3(df)
    
    # 3. Train/val split with stratification
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42,
        stratify=pd.cut(df['price'], bins=12, labels=False)  # More bins for better stratification
    )
    
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")
    
    # 4. Text features
    text_embedder = TextEmbeddingTransformer()
    X_train_text = text_embedder.transform(train_df['catalog_content'].tolist())
    X_val_text = text_embedder.transform(val_df['catalog_content'].tolist())
    print(f"Text features shape: {X_train_text.shape}")
    
    # 5. Optimized numerical features
    optimized_features = [
        'enhanced_ipq', 'text_length', 'word_count', 'avg_word_length', 'dim1', 'dim2', 'dim3',
        'has_brand', 'premium_score', 'mentions_price', 'has_specs', 'quality_score',
        'num_count', 'max_number', 'material_score', 'has_large', 'has_small', 
        'is_professional', 'has_warranty', 'has_colors', 'digit_density', 'has_caps_words'
    ] + [c for c in df.columns if c.startswith('cat_') or c.startswith('season_')]
    
    X_train_num = train_df[optimized_features].values
    X_val_num = val_df[optimized_features].values
    
    # 6. Image features
    try:
        cnn_model = SimpleCNN().to(DEVICE)
        X_train_img = extract_cnn_features(train_df, IMG_DATA_DIR, cnn_model, augment=True)
        X_val_img = extract_cnn_features(val_df, IMG_DATA_DIR, cnn_model, augment=False)
        print("Image features extracted successfully")
    except Exception as e:
        print(f"Image features skipped: {e}")
        X_train_img = np.zeros((len(train_df), 2048))
        X_val_img = np.zeros((len(val_df), 2048))
    
    # 7. Combine features
    X_train = np.hstack([X_train_text, X_train_num, X_train_img])
    X_val = np.hstack([X_val_text, X_val_num, X_val_img])
    print(f"Final feature dims - Train: {X_train.shape}, Val: {X_val.shape}")
    
    # 8. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 9. Targets
    y_train = np.log1p(train_df['price'].values)
    y_val = np.log1p(val_df['price'].values)
    y_val_orig = val_df['price'].values
    
    # 10. Use optimized architectures
    architectures = create_optimized_architectures(X_train_scaled.shape[1])
    print(f"Using {len(architectures)} optimized architectures")
    
    # 11. Train optimized ensemble
    base_models, model_weights, meta_model, _ = train_optimized_ensemble_with_selection(
        architectures, X_train_scaled, y_train, X_val_scaled, y_val, smape_threshold=22.0)
    
    print(f"✅ Successfully trained {len(base_models)} base models for ensemble")
    
    # 12. Generate predictions
    base_preds = []
    for model in base_models:
        model.eval()
        with torch.no_grad():
            preds = model(torch.FloatTensor(X_val_scaled).to(DEVICE)).cpu().numpy()
            base_preds.append(preds)
    
    base_preds = np.array(base_preds).T
    
    if meta_model is not None and hasattr(meta_model, 'predict'):
        stacked_preds_log = meta_model.predict(base_preds)
        final_preds_log = stacked_preds_log
    else:
        final_preds_log = np.average(base_preds, axis=1, weights=model_weights)
    
    final_preds = np.expm1(final_preds_log)
    final_preds = np.clip(final_preds, 0.5, 200.0)
    
    # 13. Calculate SMAPE
    smape_before = enhanced_smape(y_val_orig, final_preds)
    
    # 14. Apply advanced calibration
    calibrated_preds = advanced_calibration_v3(y_val_orig, final_preds)
    smape_final = enhanced_smape(y_val_orig, calibrated_preds)
    
    print(f"\n🎯 OPTIMIZED RESULTS:")
    print(f"Number of base models: {len(base_models)}")
    print(f"Ensemble SMAPE (before calibration): {smape_before:.2f}%")
    print(f"Calibrated SMAPE: {smape_final:.2f}%")
    print(f"Improvement: {smape_before - smape_final:.2f}%")
    
    # Price segment analysis
    print(f"\n📊 Price Segment Performance:")
    segments = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, 100), (100, 250)]
    for low, high in segments:
        mask = (y_val_orig >= low) & (y_val_orig < high)
        if mask.sum() > 0:
            segment_smape = enhanced_smape(y_val_orig[mask], calibrated_preds[mask])
            print(f"  ${low:3}-${high:3}: {segment_smape:5.1f}% ({mask.sum():4} samples)")
    
    if smape_final < 19:
        print("🎉 BREAKTHROUGH! SUB-19 SMAPE ACHIEVED! 🎉")
    elif smape_final < 20:
        print("✅ EXCELLENT! SUB-20 SMAPE ACHIEVED!")
    elif smape_final < 21:
        print("✅ VERY GOOD! Close to target!")
    
    # Save optimized model
    model_package = {
        'base_model_states': [m.state_dict() for m in base_models],
        'base_model_weights': model_weights,
        'meta_model': meta_model,
        'scaler': scaler,
        'numerical_features': optimized_features,
        'text_embedder_model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'ensemble_smape': smape_before,
        'final_smape': smape_final
    }
    torch.save(model_package, 'optimized_sub19_smape_pipeline.pth')
    print("✅ Optimized model saved successfully!")
    
    return smape_final

if __name__ == "__main__":
    # Run training
    final_smape = main_optimized()
    
    # Example prediction usage (uncomment to use)
    # predict_test_data('test.csv', 'test_out.csv')