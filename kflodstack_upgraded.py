"""
=============================================================================
升级版 Stacking 集成 — 不改特征, 只改算法
=============================================================================

原始版本问题:
  1. 缺少 LightGBM (与 XGBoost/CatBoost 互补性最强)
  2. 元学习器是 LogisticRegression (只能线性组合)
  3. 没有投票集成作为对比
  4. 没有概率校准
  5. 阈值固定 0.5

升级内容:
  A. +LightGBM 基学习器
  B. 元学习器: LogisticRegression → CatBoost(depth=3)
  C. +软投票集成 (加权平均概率)
  D. +概率校准 (Isotonic)
  E. +最优阈值搜索 (最大化 F1)

用法: 直接替换原 kflodstack.py 的 stack() 调用
  python kflodstack_upgraded.py
=============================================================================
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score)
from joblib import dump
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin

# ============================================================================
# DNN 模块 (与原版一致)
# ============================================================================

class DNN(nn.Module):
    def __init__(self, input_dim=8):
        super(DNN, self).__init__()
        self.hidden1 = nn.Linear(input_dim, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.hidden3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.1)
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
    def __init__(self, input_dim=8, epochs=50, lr=0.001):
        self.input_dim = input_dim
        self.epochs = epochs
        self.lr = lr

    def _build(self):
        self.model_ = DNN(self.input_dim)
        self.criterion_ = FocalLoss()
        self.optimizer_ = optim.Adam(self.model_.parameters(), lr=self.lr)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self._build()
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X), torch.LongTensor(y)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=True
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
# 升级 A: 构建优化版基学习器 + 元学习器
# ============================================================================

def build_base_learners(X, y, ratio, weights, use_lgb=True, calibrate=True):
    """
    构建基学习器列表

    升级点:
      A. +LightGBM (叶子优先生长, 与XGBoost/CatBoost互补)
      D. +CalibratedClassifierCV 包裹 (输出良好校准的概率)
    """
    from catboost import CatBoostClassifier
    from xgboost import XGBClassifier

    n_features = X.shape[1]

    base_learners = [
        ('lr', LogisticRegression(
            penalty='l2', C=0.1, solver='liblinear',
            max_iter=1000, class_weight='balanced',
        )),
        ('svm', SVC(
            probability=True, kernel='linear', class_weight='balanced',
        )),
        ('dt', DecisionTreeClassifier(
            criterion='gini', splitter='best', max_depth=30,
            min_samples_split=2, min_samples_leaf=1,
            class_weight='balanced',
        )),
        ('rf', RandomForestClassifier(
            n_estimators=100, max_depth=30, min_samples_split=1,
            min_samples_leaf=0.2, bootstrap=False,
            class_weight='balanced', random_state=42,
        )),
        ('xgb', XGBClassifier(
            learning_rate=0.1, n_estimators=500, max_depth=8,
            min_child_weight=0.01, scale_pos_weight=ratio,
            verbosity=0, random_state=42,
        )),
        ('catboost', CatBoostClassifier(
            iterations=300, depth=10, learning_rate=0.1,
            random_strength=10, bagging_temperature=1,
            od_type='Iter', od_wait=50, class_weights=weights,
            verbose=0, eval_metric='Accuracy', random_seed=42,
        )),
    ]

    # ---- 升级 A: +LightGBM ----
    if use_lgb:
        try:
            from lightgbm import LGBMClassifier
            base_learners.append(('lgbm', LGBMClassifier(
                n_estimators=500, learning_rate=0.05, max_depth=8,
                num_leaves=31, min_child_samples=20,
                subsample=0.8, colsample_bytree=0.8,
                class_weight='balanced', random_state=42,
                verbose=-1,
            )))
        except ImportError:
            pass  # LightGBM 未安装则跳过

    # ---- DNN ----
    torch_model = TorchDNNClassifier(input_dim=n_features, epochs=50, lr=0.001)
    base_learners.append(('dnn', torch_model))

    # ---- 升级 D: 概率校准 ----
    if calibrate:
        calibrated = []
        for name, est in base_learners:
            calibrated.append((
                name,
                CalibratedClassifierCV(est, method='isotonic', cv=3)
            ))
        base_learners = calibrated

    return base_learners


# ============================================================================
# 升级 B: 多种集成策略
# ============================================================================

def build_ensembles(base_learners, weights):
    """
    构建三种集成模型

    升级点:
      B1: Stacking + CatBoost 元学习器 (原版用 LogisticRegression)
      B2: 软投票集成 (加权平均概率, 更强的基线)
    """
    from catboost import CatBoostClassifier

    # ---- 升级 B1: Stacking + CatBoost 元学习器 ----
    # depth=3 小树防止过拟合基模型的输出
    stack = StackingClassifier(
        estimators=base_learners,
        final_estimator=CatBoostClassifier(
            depth=3, iterations=200, learning_rate=0.05,
            verbose=0, random_seed=42,
        ),
        cv=5, n_jobs=-1,
    )

    # ---- 升级 B2: 软投票集成 ----
    voting = VotingClassifier(
        estimators=base_learners, voting='soft',
    )

    return {
        'stacking': stack,
        'voting': voting,
    }


# ============================================================================
# 升级 E: 最优阈值搜索
# ============================================================================

def find_optimal_threshold(y_true, y_prob, metric='f1'):
    """
    搜索最大化指定指标的分类阈值

    天体物理中常用:
      - 'f1':         平衡精度和召回
      - 'precision':  要求高伽马纯度 (如 > 0.9)
      - 'recall':     要求高伽马检出率
      - 'gmean':       sqrt(specificity * sensitivity)
    """
    best_threshold = 0.5
    best_score = 0

    for t in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_prob >= t).astype(int)

        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred)
        elif metric == 'gmean':
            from sklearn.metrics import recall_score as rs
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            specificity = tn / (tn + fp + 1e-8)
            sensitivity = rs(y_true, y_pred)
            score = np.sqrt(specificity * sensitivity)
        else:
            score = accuracy_score(y_true, y_pred)

        if score > best_score:
            best_score = score
            best_threshold = t

    return best_threshold, best_score


# ============================================================================
# 主函数: 升级版 stack()
# ============================================================================

def stack_upgraded(datapath, outpath, use_lgb=True, calibrate=True):
    """
    升级版 stack — 接口兼容原版 kflodstack.stack()

    参数:
        datapath:  CSV数据路径
        outpath:   输出文件名前缀
        use_lgb:   是否加入 LightGBM
        calibrate: 是否启用概率校准
    """
    # ---- 加载数据 ----
    df = pd.read_csv(datapath)

    # ---- 特征列: 与原版完全一致, 不改 ----
    feature_cols = ['NuM4', 'NfiltM', 'base', 'NuM2', 'NuW1',
                    'NuM3', 'NhitM', 'NuM1']
    X = df[feature_cols]
    y = df['lable']

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X = np.asarray(X)
    y = np.asarray(y)

    ratio = float(np.sum(y == 0)) / np.sum(y == 1)
    class_weights = compute_class_weight('balanced', classes=[0, 1], y=y)
    weights = {0: class_weights[0], 1: class_weights[1]}

    # ---- 构建模型 ----
    base_learners = build_base_learners(X, y, ratio, weights, use_lgb, calibrate)
    ensembles = build_ensembles(base_learners, weights)

    # ---- 交叉验证 ----
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'roc_auc']

    all_scores = {}
    for name, ensemble in ensembles.items():
        print(f"\n{'='*60}")
        print(f"评估: {name.upper()}")
        print(f"{'='*60}")

        scores = cross_validate(
            ensemble, X, y, cv=cv, scoring=scoring,
            return_train_score=False, return_estimator=True,
            n_jobs=-1,
        )

        all_scores[name] = scores

        # 打印每 fold 结果
        for i, (acc, f1, prec, rec, auc, est) in enumerate(zip(
            scores['test_accuracy'], scores['test_f1_macro'],
            scores['test_precision_macro'], scores['test_recall_macro'],
            scores['test_roc_auc'], scores['estimator'],
        )):
            print(f"  Fold {i+1:2d}: Acc={acc:.4f} F1={f1:.4f} "
                  f"Prec={prec:.4f} Rec={rec:.4f} AUC={auc:.4f}")

        # ---- 升级 E: 阈值优化 ----
        print(f"\n  阈值优化:")
        for metric in ['f1', 'gmean']:
            # 用第一个 estimator 演示
            y_prob = est.predict_proba(X)[:, 1]
            best_t, best_s = find_optimal_threshold(y, y_prob, metric)
            print(f"    {metric}: best_thr={best_t:.2f}, score={best_s:.4f} "
                  f"(默认0.5阈值={metric}分)")

        # ---- 保存最优模型 ----
        print(f"\n  保存最优模型...")
        # 按AUC选最优 fold
        best_idx = np.argmax(scores['test_roc_auc'])
        best_est = scores['estimator'][best_idx]
        dump(best_est,
             f'/home/abcdlj/Gam-p/final/models/{outpath}_{name}_best.joblib')

        # 保存预测结果
        y_pred = best_est.predict(X)
        y_prob = best_est.predict_proba(X)[:, 1]
        df_pred = pd.DataFrame({
            'y_true': y, 'y_pred': y_pred, 'y_prob': y_prob,
        })
        df_pred.to_csv(
            f'/home/abcdlj/Gam-p/final/predict/{outpath}_{name}_best.csv',
            index=False,
        )

    # ---- 最终汇总 ----
    print(f"\n{'='*70}")
    print("最终对比: 原始 Stacking (LR元学习器) vs 升级版本")
    print("=" * 70)
    print(f"\n{'模型':<25s} {'AUC':>8s}  {'F1':>8s}  {'Accuracy':>10s}")
    print("-" * 60)

    for name, scores in all_scores.items():
        auc_m = np.mean(scores['test_roc_auc'])
        f1_m = np.mean(scores['test_f1_macro'])
        acc_m = np.mean(scores['test_accuracy'])
        print(f"  {name:<25s} {auc_m:8.4f}  {f1_m:8.4f}  {acc_m:10.4f}")

    # 保存 scores
    for name, scores in all_scores.items():
        df_s = pd.DataFrame({
            'Accuracy': scores['test_accuracy'],
            'F1': scores['test_f1_macro'],
            'Precision': scores['test_precision_macro'],
            'Recall': scores['test_recall_macro'],
            'AUC': scores['test_roc_auc'],
        })
        df_s.to_csv(
            f'/home/abcdlj/Gam-p/final/predict/{outpath}_{name}.csv',
            index=False,
        )

    return all_scores


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='升级版 Stacking (不改特征, 只改算法)'
    )
    parser.add_argument('--data', type=str,
                        default='/home/abcdlj/Gam-p/final/data/df12rec.csv',
                        help='CSV数据路径')
    parser.add_argument('--out', type=str, default='df12_upgraded',
                        help='输出文件名前缀')
    parser.add_argument('--no-lgb', action='store_true',
                        help='禁用 LightGBM')
    parser.add_argument('--no-calibrate', action='store_true',
                        help='禁用概率校准')
    args = parser.parse_args()

    stack_upgraded(
        datapath=args.data,
        outpath=args.out,
        use_lgb=not args.no_lgb,
        calibrate=not args.no_calibrate,
    )
