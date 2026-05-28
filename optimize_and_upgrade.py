"""
=============================================================================
快速性能提升方案: 超参优化(Optuna) + SHAP特征筛选 + 集成升级
=============================================================================

三步走:
  Phase 1: Optuna 贝叶斯超参优化 → 每个模型自动找到最佳参数
  Phase 2: SHAP 驱动特征选择    → 剔除冗余特征, 保留最有物理意义的
  Phase 3: 集成策略升级          → 从简单Stacking升级为加权融合

与现有代码的兼容:
  - 复用 kflodstack.py 中的 DNN, FocalLoss, TorchDNNClassifier
  - 复用 physics_features.py 的物理特征
  - 保持 RepeatedStratifiedKFold (5×10) 的评估框架

预期收益:
  Phase 1: AUC +0.005~0.015 (超参优化)
  Phase 2: 不损失AUC, 特征数减少30-50%
  Phase 3: AUC +0.01~0.02 (更好的集成)

运行:
  python optimize_and_upgrade.py --data your_data.csv
  python optimize_and_upgrade.py --simulated  (快速验证流程)
=============================================================================
"""
import numpy as np
import pandas as pd
import os
import sys
import argparse
import warnings
from joblib import dump
warnings.filterwarnings('ignore')

from sklearn.model_selection import (RepeatedStratifiedKFold, cross_validate,
                                      StratifiedKFold)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import BaseEstimator, ClassifierMixin, clone

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from physics_features import (compute_physics_features, ORIGINAL_FEATURES,
                               PHYSICS_FEATURES)

# ============================================================================
# 复用现有 DNN 定义 (与 kflodstack.py 一致)
# ============================================================================

class DNN(nn.Module):
    def __init__(self, input_dim=8, hidden1=128, hidden2=64, hidden3=32,
                 dropout=0.1):
        super(DNN, self).__init__()
        self.hidden1 = nn.Linear(input_dim, hidden1)
        self.hidden2 = nn.Linear(hidden1, hidden2)
        self.hidden3 = nn.Linear(hidden2, hidden3)
        self.output = nn.Linear(hidden3, 2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.dropout(x)
        x = self.relu(self.hidden2(x))
        x = self.dropout(x)
        x = self.relu(self.hidden3(x))
        x = self.dropout(x)
        x = self.softmax(self.output(x))
        return x


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (self.alpha * (1 - pt) ** self.gamma).detach()
        return (focal_weight * ce_loss).mean()


class TorchDNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim=8, hidden1=128, hidden2=64, hidden3=32,
                 dropout=0.1, epochs=50, lr=0.001, batch_size=32,
                 focal_gamma=2, focal_alpha=0.25):
        self.input_dim = input_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

    def _build(self):
        self.model_ = DNN(self.input_dim, self.hidden1, self.hidden2,
                          self.hidden3, self.dropout)
        self.criterion_ = FocalLoss(self.focal_gamma, self.focal_alpha)
        self.optimizer_ = optim.Adam(self.model_.parameters(), lr=self.lr)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self._build()
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X), torch.LongTensor(y)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        self.model_.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in dataloader:
                outputs = self.model_(batch_x)
                loss = self.criterion_(outputs, batch_y)
                self.optimizer_.zero_grad()
                loss.backward()
                self.optimizer_.step()
        return self

    def predict(self, X):
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(torch.FloatTensor(X))
            _, predicted = torch.max(outputs, 1)
        return predicted.numpy()

    def predict_proba(self, X):
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(torch.FloatTensor(X))
        return outputs.numpy()


# ============================================================================
# Phase 1: Optuna 贝叶斯超参优化
# ============================================================================

def objective_catboost(trial, X, y, cv):
    """Optuna objective for CatBoost"""
    from catboost import CatBoostClassifier

    params = {
        'depth': trial.suggest_int('depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'iterations': trial.suggest_int('iterations', 100, 800),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10, log=True),
        'random_strength': trial.suggest_float('random_strength', 0.1, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.1, 5),
        'border_count': trial.suggest_int('border_count', 32, 255),
    }

    class_weights = compute_class_weight('balanced', classes=[0, 1], y=y)
    weights = {0: class_weights[0], 1: class_weights[1]}

    aucs = []
    for train_idx, val_idx in cv.split(X, y):
        Xt, Xv = X[train_idx], X[val_idx]
        yt, yv = y[train_idx], y[val_idx]

        model = CatBoostClassifier(
            **params, class_weights=weights, verbose=0,
            eval_metric='AUC', random_seed=42, od_type='Iter', od_wait=50,
        )
        model.fit(Xt, yt, eval_set=(Xv, yv), verbose=False)
        yp = model.predict_proba(Xv)[:, 1]
        aucs.append(roc_auc_score(yv, yp))

    return np.mean(aucs)


def objective_xgboost(trial, X, y, cv):
    """Optuna objective for XGBoost"""
    from xgboost import XGBClassifier

    ratio = float(np.sum(y == 0)) / np.sum(y == 1)
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 10, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10, log=True),
    }

    aucs = []
    for train_idx, val_idx in cv.split(X, y):
        Xt, Xv = X[train_idx], X[val_idx]
        yt, yv = y[train_idx], y[val_idx]

        model = XGBClassifier(
            **params, scale_pos_weight=ratio, random_state=42, verbosity=0,
            use_label_encoder=False, eval_metric='auc',
        )
        model.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
        yp = model.predict_proba(Xv)[:, 1]
        aucs.append(roc_auc_score(yv, yp))

    return np.mean(aucs)


def objective_random_forest(trial, X, y, cv):
    """Optuna objective for Random Forest"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 40),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_float('max_features', 0.3, 1.0),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
    }

    aucs = []
    for train_idx, val_idx in cv.split(X, y):
        Xt, Xv = X[train_idx], X[val_idx]
        yt, yv = y[train_idx], y[val_idx]

        model = RandomForestClassifier(
            **params, random_state=42, n_jobs=-1, class_weight='balanced',
        )
        model.fit(Xt, yt)
        yp = model.predict_proba(Xv)[:, 1]
        aucs.append(roc_auc_score(yv, yp))

    return np.mean(aucs)


def objective_dnn(trial, X, y, cv):
    """Optuna objective for DNN"""
    input_dim = X.shape[1]
    params = {
        'hidden1': trial.suggest_int('hidden1', 32, 256),
        'hidden2': trial.suggest_int('hidden2', 16, 128),
        'hidden3': trial.suggest_int('hidden3', 8, 64),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'lr': trial.suggest_float('lr', 1e-4, 0.01, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        'epochs': trial.suggest_int('epochs', 20, 100),
        'focal_gamma': trial.suggest_float('focal_gamma', 0.5, 3.0),
    }

    aucs = []
    for train_idx, val_idx in cv.split(X, y):
        Xt, Xv = X[train_idx], X[val_idx]
        yt, yv = y[train_idx], y[val_idx]

        model = TorchDNNClassifier(input_dim=input_dim, **params)
        model.fit(Xt, yt)
        yp = model.predict_proba(Xv)[:, 1]
        aucs.append(roc_auc_score(yv, yp))

    return np.mean(aucs)


def objective_logistic_regression(trial, X, y, cv):
    """Optuna objective for Logistic Regression"""
    params = {
        'C': trial.suggest_float('C', 0.001, 10, log=True),
        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
        'solver': 'saga',  # supports both l1 and l2
    }

    aucs = []
    for train_idx, val_idx in cv.split(X, y):
        Xt, Xv = X[train_idx], X[val_idx]
        yt, yv = y[train_idx], y[val_idx]

        model = LogisticRegression(
            **params, max_iter=2000, class_weight='balanced', random_state=42,
        )
        model.fit(Xt, yt)
        yp = model.predict_proba(Xv)[:, 1]
        aucs.append(roc_auc_score(yv, yp))

    return np.mean(aucs)


def objective_svm(trial, X, y, cv):
    """Optuna objective for SVM"""
    params = {
        'C': trial.suggest_float('C', 0.01, 100, log=True),
        'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
    }

    aucs = []
    for train_idx, val_idx in cv.split(X, y):
        Xt, Xv = X[train_idx], X[val_idx]
        yt, yv = y[train_idx], y[val_idx]

        model = SVC(**params, probability=True, class_weight='balanced',
                     random_state=42)
        model.fit(Xt, yt)
        yp = model.predict_proba(Xv)[:, 1]
        aucs.append(roc_auc_score(yv, yp))

    return np.mean(aucs)


def run_optuna_optimization(X, y, n_trials=50, n_jobs=1):
    """
    对所有模型运行 Optuna 贝叶斯超参优化
    自动跳过未安装的模型包

    返回: {model_name: best_params}
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # 检查包可用性
    optimizers = {
        'CatBoost': (objective_catboost, True),
        'XGBoost': (objective_xgboost, True),
        'RandomForest': (objective_random_forest, True),
        'DNN': (objective_dnn, False),  # 手动禁用 (torch/sympy兼容问题)
        'LogisticRegression': (objective_logistic_regression, True),
        'SVM': (objective_svm, True),
    }

    # 检查导入; DNN被手动禁用
    for name in list(optimizers.keys()):
        fn, enabled = optimizers[name]
        if not enabled:
            del optimizers[name]
            continue
        if name == 'CatBoost':
            try:
                import catboost
            except ImportError:
                del optimizers[name]
        elif name == 'XGBoost':
            try:
                import xgboost
            except ImportError:
                del optimizers[name]
        elif name == 'DNN':
            try:
                import torch
            except ImportError:
                del optimizers[name]

    best_params = {}
    results = {}

    for name, (objective_fn, _) in optimizers.items():
        print(f"\n  optimizing {name} ({n_trials} trials)...")

        try:
            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(
                direction='maximize', sampler=sampler,
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            )
            study.optimize(
                lambda trial: objective_fn(trial, X, y, cv_inner),
                n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=False,
            )

            best_params[name] = study.best_params
            results[name] = study.best_value
            print(f"    best AUC: {study.best_value:.4f}")
            print(f"    best params: {study.best_params}")
        except Exception as e:
            print(f"    [SKIP] {name} 失败: {type(e).__name__}")

    return best_params, results


# ============================================================================
# Phase 2: SHAP 驱动特征选择
# ============================================================================

def shap_feature_selection(X, y, feature_names, min_features=5,
                            step=3, cv=None):
    """
    基于SHAP的特征选择: 迭代剔除最不重要特征, 找到最优子集

    参数:
        X:             特征矩阵
        y:             标签
        feature_names: 特征名列表
        min_features:  保留最少特征数
        step:          每轮剔除多少特征
        cv:            交叉验证

    返回:
        selected_indices: 最优特征子集的索引
        selected_names:   最优特征子集的名称
    """
    import shap
    from catboost import CatBoostClassifier

    if cv is None:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    print(f"\n  SHAP 特征选择: {len(feature_names)} → 寻找最优子集...")

    current_indices = list(range(len(feature_names)))
    current_names = list(feature_names)
    best_score = 0
    best_n_features = len(current_names)
    history = []

    while len(current_indices) >= min_features:
        # 训练模型
        cw = compute_class_weight('balanced', classes=[0, 1], y=y)
        model = CatBoostClassifier(
            iterations=300, depth=8, learning_rate=0.1,
            class_weights={0: cw[0], 1: cw[1]}, verbose=0, random_seed=42,
        )

        X_current = X[:, current_indices]
        aucs = []
        for train_idx, val_idx in cv.split(X_current, y):
            Xt, Xv = X_current[train_idx], X_current[val_idx]
            yt, yv = y[train_idx], y[val_idx]
            model.fit(Xt, yt, verbose=False)
            yp = model.predict_proba(Xv)[:, 1]
            aucs.append(roc_auc_score(yv, yp))

        mean_auc = np.mean(aucs)

        # 计算SHAP重要性
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_current[:min(300, len(X_current))])
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        elif shap_vals.ndim == 3:
            shap_vals = shap_vals[:, :, 1]

        mean_abs_shap = np.abs(shap_vals).mean(axis=0)

        history.append({
            'n_features': len(current_indices),
            'auc': mean_auc,
            'features': current_names.copy(),
        })

        print(f"    {len(current_indices):3d} features → AUC={mean_auc:.4f}")

        if mean_auc >= best_score:
            best_score = mean_auc
            best_n_features = len(current_indices)

        # 剔除最不重要的 step 个特征
        if len(current_indices) <= min_features:
            break

        n_remove = min(step, len(current_indices) - min_features)
        # 按SHAP排序, 去掉末尾的
        sorted_idx = np.argsort(mean_abs_shap)
        to_remove = sorted_idx[:n_remove]
        current_indices = [current_indices[i] for i in range(len(current_indices))
                           if i not in to_remove]
        current_names = [feature_names[i] for i in current_indices]

    # 找到AUC不下降的最少特征数
    # 策略: 选AUC不低于best_score - 0.005 的最少特征
    threshold = best_score - 0.005
    optimal = None
    for h in sorted(history, key=lambda x: x['n_features']):
        if h['auc'] >= threshold:
            optimal = h
            break

    if optimal is None:
        optimal = history[-1]

    print(f"\n  最优子集: {optimal['n_features']} 特征, AUC={optimal['auc']:.4f}")
    selected_indices = [feature_names.index(f) for f in optimal['features']]
    return selected_indices, optimal['features'], history


# ============================================================================
# Phase 3: 集成策略升级
# ============================================================================

def build_upgraded_ensemble(best_params, n_features, ratio, weights):
    """
    使用 Optuna 优化后的参数构建升级集成模型

    升级点:
      1. 每个基模型用最佳超参
      2. 元学习器从 LogisticRegression 升级为 CatBoost(小深度)
      3. 支持加权投票作为备选
    """
    try:
        from catboost import CatBoostClassifier
        _has_catboost = True
    except ImportError:
        _has_catboost = False

    try:
        from xgboost import XGBClassifier
        _has_xgboost = True
    except ImportError:
        _has_xgboost = False

    # 基学习器 — 全部用 Optuna 最佳参数 (未优化的用默认值)
    lr_params = best_params.get('LogisticRegression', {})
    base_lr = LogisticRegression(
        C=lr_params.get('C', 0.1),
        penalty=lr_params.get('penalty', 'l2'),
        solver='saga' if lr_params.get('penalty') == 'l1' else 'liblinear',
        max_iter=2000, class_weight='balanced', random_state=42,
    )

    svm_params = best_params.get('SVM', {})
    base_svm = SVC(
        C=svm_params.get('C', 1.0),
        kernel=svm_params.get('kernel', 'rbf'),
        gamma=svm_params.get('gamma', 'scale'),
        probability=True, class_weight='balanced', random_state=42,
    )

    rf_params = best_params.get('RandomForest', {})
    base_rf = RandomForestClassifier(
        n_estimators=rf_params.get('n_estimators', 200),
        max_depth=rf_params.get('max_depth', 20),
        min_samples_split=rf_params.get('min_samples_split', 2),
        min_samples_leaf=rf_params.get('min_samples_leaf', 1),
        max_features=rf_params.get('max_features', 0.8),
        bootstrap=rf_params.get('bootstrap', True),
        class_weight='balanced', random_state=42, n_jobs=-1,
    )

    base_learners = [('lr', base_lr), ('svm', base_svm), ('rf', base_rf)]

    if _has_xgboost:
        xgb_params = best_params.get('XGBoost', {})
        base_xgb = XGBClassifier(
            max_depth=xgb_params.get('max_depth', 8),
            learning_rate=xgb_params.get('learning_rate', 0.1),
            n_estimators=xgb_params.get('n_estimators', 500),
            subsample=xgb_params.get('subsample', 0.8),
            colsample_bytree=xgb_params.get('colsample_bytree', 0.8),
            min_child_weight=xgb_params.get('min_child_weight', 1),
            reg_alpha=xgb_params.get('reg_alpha', 0),
            reg_lambda=xgb_params.get('reg_lambda', 1),
            scale_pos_weight=ratio, random_state=42, verbosity=0,
        )
        base_learners.append(('xgb', base_xgb))

    if _has_catboost:
        cb_params = best_params.get('CatBoost', {})
        base_cb = CatBoostClassifier(
            depth=cb_params.get('depth', 8),
            learning_rate=cb_params.get('learning_rate', 0.1),
            iterations=cb_params.get('iterations', 300),
            l2_leaf_reg=cb_params.get('l2_leaf_reg', 3),
            random_strength=cb_params.get('random_strength', 1),
            bagging_temperature=cb_params.get('bagging_temperature', 1),
            border_count=cb_params.get('border_count', 128),
            class_weights=weights, verbose=0, random_seed=42,
        )
        base_learners.append(('catboost', base_cb))

    # DNN skipped: 当前conda环境的torch 2.5.1与sympy版本不兼容
    # 在正常环境中取消下面注释即可加入DNN:
    # dnn_params = best_params.get('DNN', {})
    # base_dnn = TorchDNNClassifier(
    #     input_dim=n_features, hidden1=dnn_params.get('hidden1', 128),
    #     hidden2=dnn_params.get('hidden2', 64), hidden3=dnn_params.get('hidden3', 32),
    #     dropout=dnn_params.get('dropout', 0.1), lr=dnn_params.get('lr', 0.001),
    #     batch_size=dnn_params.get('batch_size', 32), epochs=dnn_params.get('epochs', 50),
    #     focal_gamma=dnn_params.get('focal_gamma', 2),
    # )
    # base_learners.append(('dnn', base_dnn))

    # =============================================================
    # 升级 A: CatBoost 元学习器 (代替 LogisticRegression)
    # =============================================================
    meta_catboost = CatBoostClassifier(
        depth=3, iterations=200, learning_rate=0.05,
        verbose=0, random_seed=42,
    ) if _has_catboost else LogisticRegression(max_iter=1000)

    # =============================================================
    # 升级 B: 双层 Stacking
    # =============================================================
    from sklearn.ensemble import VotingClassifier

    # B1: 加权投票集成 (软投票)
    voting_ensemble = VotingClassifier(
        estimators=base_learners,
        voting='soft',
        weights=None,
    )

    # B2: 标准Stacking (优化后的基模型 + CatBoost元学习器)
    stacking_ensemble = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_catboost,
        cv=5,  # 内层CV防止过拟合
        n_jobs=-1,
    )

    return {
        'voting': voting_ensemble,
        'stacking': stacking_ensemble,
    }


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='优化+筛选+升级')
    parser.add_argument('--data', type=str, default=None,
                        help='CSV数据路径')
    parser.add_argument('--simulated', action='store_true',
                        help='使用模拟数据')
    parser.add_argument('--output', type=str, default='./results_optimized',
                        help='输出目录')
    parser.add_argument('--n-trials', type=int, default=30,
                        help='Optuna 试验次数 (建议50+)')
    parser.add_argument('--skip-optuna', action='store_true',
                        help='跳过超参优化 (使用默认参数)')
    parser.add_argument('--skip-shap', action='store_true',
                        help='跳过SHAP特征选择')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ================================================================
    # 0. 数据加载
    # ================================================================
    print("=" * 70)
    print("Phase 0: 数据准备")
    print("=" * 70)

    if args.data:
        df = pd.read_csv(args.data)
    else:
        np.random.seed(42)
        n = 3000
        gamma = pd.DataFrame({
            'NuM1': np.random.poisson(75, n), 'NuM2': np.random.poisson(30, n),
            'NuM3': np.random.poisson(12, n), 'NuM4': np.random.poisson(5, n),
            'NfiltM': np.random.poisson(210, n), 'NhitM': np.random.poisson(260, n),
            'NuW1': np.random.poisson(55, n), 'base': np.abs(np.random.normal(100, 25, n)),
            'rec_Eage': np.abs(np.random.normal(10, 5, n)), 'lable': 1,
        })
        proton = pd.DataFrame({
            'NuM1': np.random.poisson(42, n), 'NuM2': np.random.poisson(35, n),
            'NuM3': np.random.poisson(27, n), 'NuM4': np.random.poisson(20, n),
            'NfiltM': np.random.poisson(280, n), 'NhitM': np.random.poisson(330, n),
            'NuW1': np.random.poisson(45, n), 'base': np.abs(np.random.normal(100, 25, n)),
            'rec_Eage': np.abs(np.random.normal(9, 5, n)), 'lable': 0,
        })
        border = pd.DataFrame({
            'NuM1': np.random.poisson(55, n//3), 'NuM2': np.random.poisson(32, n//3),
            'NuM3': np.random.poisson(18, n//3), 'NuM4': np.random.poisson(12, n//3),
            'NfiltM': np.random.poisson(245, n//3), 'NhitM': np.random.poisson(290, n//3),
            'NuW1': np.random.poisson(50, n//3), 'base': np.abs(np.random.normal(100, 25, n//3)),
            'rec_Eage': np.abs(np.random.normal(9.5, 5, n//3)),
            'lable': np.random.choice([0, 1], n//3),
        })
        df = pd.concat([gamma, proton, border], ignore_index=True)

    # 物理特征
    df = compute_physics_features(df)
    feature_cols = ORIGINAL_FEATURES + PHYSICS_FEATURES
    X = df[feature_cols].values
    y = df['lable'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"  样本: {len(df)}, γ={y.sum()}, p={len(y)-y.sum()}")
    print(f"  特征: {X.shape[1]} ({len(ORIGINAL_FEATURES)}原始 + "
          f"{len(PHYSICS_FEATURES)}物理)")

    # ================================================================
    # Phase 1: Optuna 超参优化
    # ================================================================
    print(f"\n{'='*70}")
    print("Phase 1: Optuna 贝叶斯超参优化")
    print("=" * 70)

    if not args.skip_optuna:
        best_params, optuna_results = run_optuna_optimization(
            X_scaled, y, n_trials=args.n_trials
        )
        # 保存
        pd.DataFrame([
            {'model': k, 'best_auc': v, **best_params[k]}
            for k, v in optuna_results.items()
        ]).to_csv(f'{args.output}/optuna_best_params.csv', index=False)

        print(f"\n  Optuna 优化结果:")
        for name, auc in sorted(optuna_results.items(),
                                 key=lambda x: x[1], reverse=True):
            print(f"    {name:<20s} AUC={auc:.4f}")
    else:
        print("  [跳过]")
        best_params = {}
        optuna_results = {}

    # ================================================================
    # Phase 2: SHAP 特征选择
    # ================================================================
    print(f"\n{'='*70}")
    print("Phase 2: SHAP 驱动特征选择")
    print("=" * 70)

    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    if not args.skip_shap:
        selected_idx, selected_names, shap_history = shap_feature_selection(
            X_scaled, y, feature_cols, min_features=8, step=4, cv=inner_cv
        )
        X_selected = X_scaled[:, selected_idx]
        pd.DataFrame(shap_history).to_csv(
            f'{args.output}/shap_feature_selection.csv', index=False
        )
    else:
        print("  [跳过]")
        selected_idx = list(range(len(feature_cols)))
        selected_names = list(feature_cols)
        X_selected = X_scaled
        shap_history = []

    # ================================================================
    # Phase 3: 集成升级 + 完整评估
    # ================================================================
    print(f"\n{'='*70}")
    print("Phase 3: 集成策略升级 + 完整评估")
    print("=" * 70)

    ratio = float(np.sum(y == 0)) / np.sum(y == 1)
    class_weights = compute_class_weight('balanced', classes=[0, 1], y=y)
    weights = {0: class_weights[0], 1: class_weights[1]}

    ensembles = build_upgraded_ensemble(
        best_params, len(selected_names), ratio, weights
    )

    # 基线: 原始 Stacking (LogisticRegression 元学习器)
    try:
        from xgboost import XGBClassifier
        _has_xgb = True
    except ImportError:
        _has_xgb = False
    try:
        from catboost import CatBoostClassifier
        _has_cb = True
    except ImportError:
        _has_cb = False

    # 用筛选后的特征建基线集成
    baseline_estimators = []
    baseline_estimators.append(('lr', LogisticRegression(
        penalty='l2', C=0.1, solver='liblinear', max_iter=1000,
        class_weight='balanced',
    )))
    baseline_estimators.append(('svm', SVC(
        probability=True, kernel='linear', class_weight='balanced',
    )))
    baseline_estimators.append(('rf', RandomForestClassifier(
        n_estimators=100, max_depth=30, class_weight='balanced', random_state=42,
    )))
    if _has_xgb:
        baseline_estimators.append(('xgb', XGBClassifier(
            learning_rate=0.1, n_estimators=500, max_depth=8,
            scale_pos_weight=ratio, verbosity=0,
        )))
    if _has_cb:
        baseline_estimators.append(('catboost', CatBoostClassifier(
            iterations=300, depth=10, learning_rate=0.1, random_strength=10,
            bagging_temperature=1, od_type='Iter', od_wait=50,
            class_weights=weights, verbose=0,
        )))
    # DNN baseline skipped (torch/sympy incompatibility on this machine)

    baseline_stack = StackingClassifier(
        estimators=baseline_estimators,
        final_estimator=LogisticRegression(),
        cv=5, n_jobs=-1,
    )

    # 评估
    cv_outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'roc_auc']

    print("\n  评估: Baseline Stacking (原始参数 + SHAP特征)")
    scores_baseline = cross_validate(
        baseline_stack, X_selected, y, cv=cv_outer,
        scoring=scoring, return_train_score=False, n_jobs=-1,
    )

    print("\n  评估: Optimized Stacking (优化参数 + CatBoost元学习器)")
    scores_optimized = cross_validate(
        ensembles['stacking'], X_selected, y, cv=cv_outer,
        scoring=scoring, return_train_score=False, n_jobs=-1,
    )

    print("\n  评估: Voting Ensemble (优化参数)")
    scores_voting = cross_validate(
        ensembles['voting'], X_selected, y, cv=cv_outer,
        scoring=scoring, return_train_score=False, n_jobs=-1,
    )

    # ================================================================
    # 汇总比较
    # ================================================================
    print(f"\n{'='*90}")
    print("最终对比结果")
    print("=" * 90)
    print(f"\n{'模型':<30s} {'AUC':>8s}  {'F1':>8s}  {'Accuracy':>10s}  "
          f"{'Precision':>10s}  {'Recall':>8s}")
    print("-" * 90)

    comparison = {}
    metric_keys = ['test_roc_auc', 'test_f1_macro', 'test_accuracy',
                   'test_precision_macro', 'test_recall_macro']
    display_metrics = ['AUC', 'F1', 'Accuracy', 'Precision', 'Recall']

    for name, scores in [
        ('Baseline_Stack', scores_baseline),
        ('Optimized_Stack', scores_optimized),
        ('Voting_Ensemble', scores_voting),
    ]:
        means = [np.mean(scores[k]) for k in metric_keys]
        stds = [np.std(scores[k]) for k in metric_keys]
        comparison[name] = {'means': means, 'stds': stds}
        metric_str = "  ".join(f"{m:8.4f}" for m in means)
        print(f"  {name:<28s} {metric_str}")
        std_str = "  ".join(f"{s:8.4f}" for s in stds)
        print(f"  {'(±std)':<28s} {std_str}")
        print()

    # 计算提升
    baseline_auc = comparison['Baseline_Stack']['means'][0]
    print(f"相对于 Baseline Stacking 的 AUC 提升:")
    for name in ['Optimized_Stack', 'Voting_Ensemble']:
        if name in comparison:
            delta = (comparison[name]['means'][0] - baseline_auc) * 100
            sign = '+' if delta > 0 else ''
            print(f"  {name}: {sign}{delta:.2f} 个百分点")

    # ================================================================
    # 保存结果
    # ================================================================
    # 保存所有score到CSV
    all_scores = {}
    for name, scores in [
        ('Baseline_Stack', scores_baseline),
        ('Optimized_Stack', scores_optimized),
        ('Voting_Ensemble', scores_voting),
    ]:
        for k in metric_keys:
            all_scores[f'{name}_{k}'] = scores[k]

    df_scores = pd.DataFrame(all_scores)
    df_scores.to_csv(f'{args.output}/comparison_scores.csv', index=False)

    # 保存完整报告
    with open(f'{args.output}/optimization_report.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("优化报告\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"样本数: {len(df)}\n")
        f.write(f"特征数: {X.shape[1]} → {len(selected_names)} (SHAP筛选后)\n\n")

        if not args.skip_optuna:
            f.write("Optuna 最优参数:\n")
            for name, params in best_params.items():
                f.write(f"  {name}: {params}\n")
            f.write(f"  Optuna AUC: {optuna_results}\n\n")

        f.write("最终对比:\n")
        for name, info in comparison.items():
            f.write(f"  {name}:\n")
            for i, m in enumerate(display_metrics):
                f.write(f"    {m}: {info['means'][i]:.4f} ± {info['stds'][i]:.4f}\n")

    print(f"\n结果已保存到: {args.output}/")

    return comparison, best_params, selected_names


if __name__ == "__main__":
    main()
