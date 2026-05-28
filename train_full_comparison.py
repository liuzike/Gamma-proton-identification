"""
=============================================================================
完整对比实验: Baseline vs Physics-Enhanced vs CNN-CBAM vs Multimodal Fusion
=============================================================================

四组对照实验:
  A) Baseline          : 8原始特征 + CatBoost (原有方法)
  B) +Physics Features : 8原始 + 24物理特征 + CatBoost
  C) +CNN-CBAM         : 探测器图像 + ResNet-CBAM (呼应Zhang+2023论文)
  D) +Multimodal Fusion: 表格特征 + 探测器图像 → 多模态融合 (原创创新)

论文对应:
  Zhang et al. 2023 (arXiv:2303.00370)
  用ResNet-CBAM在Fermi/GBM探测器计数图上做GRB/非GRB二分类

=============================================================================
"""
import numpy as np
import pandas as pd
import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score)

from physics_features import compute_physics_features, ORIGINAL_FEATURES, PHYSICS_FEATURES

# 尝试导入CNN模块 (需要PyTorch)
try:
    from cnn_cbam_model import (LHAASOEventMapper, ResNetCBAM,
                                 LHAASOResNetCBAMClassifier)
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARNING] PyTorch未安装, 将跳过CNN相关实验")


def run_experiment_A_baseline(X, y, cv, feature_name="Baseline"):
    """A: 基线 CatBoost (8原始特征)"""
    from catboost import CatBoostClassifier
    from sklearn.utils.class_weight import compute_class_weight

    metrics_list = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        Xt, Xs = X[train_idx], X[test_idx]
        yt, ys = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        Xt_s = scaler.fit_transform(Xt)
        Xs_s = scaler.transform(Xs)

        cw = compute_class_weight('balanced', classes=[0, 1], y=yt)
        model = CatBoostClassifier(
            iterations=200, depth=8, learning_rate=0.1,
            class_weights={0: cw[0], 1: cw[1]}, verbose=0, random_seed=42
        )
        model.fit(Xt_s, yt)
        yp = model.predict(Xs_s)
        yprob = model.predict_proba(Xs_s)[:, 1]

        metrics_list.append({
            'accuracy': accuracy_score(ys, yp), 'f1': f1_score(ys, yp),
            'precision': precision_score(ys, yp), 'recall': recall_score(ys, yp),
            'auc': roc_auc_score(ys, yprob)
        })

        if (fold + 1) % 5 == 0:
            print(f"  [{feature_name}] 完成 {fold+1} folds")

    return pd.DataFrame(metrics_list)


def run_experiment_B_physics(X, y, cv, feature_name="Physics"):
    """B: 物理特征 CatBoost (8原始 + 24物理特征)"""
    from catboost import CatBoostClassifier
    from sklearn.utils.class_weight import compute_class_weight

    metrics_list = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        Xt, Xs = X[train_idx], X[test_idx]
        yt, ys = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        Xt_s = scaler.fit_transform(Xt)
        Xs_s = scaler.transform(Xs)

        cw = compute_class_weight('balanced', classes=[0, 1], y=yt)
        model = CatBoostClassifier(
            iterations=200, depth=8, learning_rate=0.1,
            class_weights={0: cw[0], 1: cw[1]}, verbose=0, random_seed=42
        )
        model.fit(Xt_s, yt)
        yp = model.predict(Xs_s)
        yprob = model.predict_proba(Xs_s)[:, 1]

        metrics_list.append({
            'accuracy': accuracy_score(ys, yp), 'f1': f1_score(ys, yp),
            'precision': precision_score(ys, yp), 'recall': recall_score(ys, yp),
            'auc': roc_auc_score(ys, yprob)
        })

        if (fold + 1) % 5 == 0:
            print(f"  [{feature_name}] 完成 {fold+1} folds")

    return pd.DataFrame(metrics_list)


def run_experiment_C_cnn(X_tabular, y, cv, feature_name="CNN-CBAM"):
    """C: CNN-CBAM 纯图像模型"""
    if not HAS_TORCH:
        print(f"  [{feature_name}] 跳过 (无PyTorch)")
        return None

    # 转换为DataFrame以便事件映射器使用
    df_data = pd.DataFrame(X_tabular, columns=ORIGINAL_FEATURES)
    mapper = LHAASOEventMapper(grid_size=32)

    metrics_list = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X_tabular, y)):
        Xt, Xs = X_tabular[train_idx], X_tabular[test_idx]
        yt, ys = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        Xt_s = scaler.fit_transform(Xt)
        Xs_s = scaler.transform(Xs)

        # 为图像模型准备DataFrame
        df_train = pd.DataFrame(Xt_s, columns=ORIGINAL_FEATURES)
        df_test = pd.DataFrame(Xs_s, columns=ORIGINAL_FEATURES)

        model = LHAASOResNetCBAMClassifier(
            use_tabular=False, use_image=True,
            epochs=30, batch_size=32, img_grid_size=32
        )
        model.fit(df_train.values, yt, image_builder=mapper)
        yp = model.predict(df_test.values)
        yprob = model.predict_proba(df_test.values)[:, 1]

        metrics_list.append({
            'accuracy': accuracy_score(ys, yp), 'f1': f1_score(ys, yp),
            'precision': precision_score(ys, yp), 'recall': recall_score(ys, yp),
            'auc': roc_auc_score(ys, yprob)
        })

        if (fold + 1) % 5 == 0:
            print(f"  [{feature_name}] 完成 {fold+1} folds")

    return pd.DataFrame(metrics_list)


def run_experiment_D_multimodal(X_tabular, y, cv, feature_name="Multimodal"):
    """D: 多模态融合 (CNN图像 + MLP表格) ← 原创创新"""
    if not HAS_TORCH:
        print(f"  [{feature_name}] 跳过 (无PyTorch)")
        return None

    mapper = LHAASOEventMapper(grid_size=32)

    metrics_list = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X_tabular, y)):
        Xt, Xs = X_tabular[train_idx], X_tabular[test_idx]
        yt, ys = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        Xt_s = scaler.fit_transform(Xt)
        Xs_s = scaler.transform(Xs)

        model = LHAASOResNetCBAMClassifier(
            use_tabular=True, use_image=True,  # 两路融合
            epochs=30, batch_size=32, img_grid_size=32, fusion_dim=128
        )
        model.fit(Xt_s, yt, image_builder=mapper)
        yp = model.predict(Xs_s)
        yprob = model.predict_proba(Xs_s)[:, 1]

        metrics_list.append({
            'accuracy': accuracy_score(ys, yp), 'f1': f1_score(ys, yp),
            'precision': precision_score(ys, yp), 'recall': recall_score(ys, yp),
            'auc': roc_auc_score(ys, yprob)
        })

        if (fold + 1) % 5 == 0:
            print(f"  [{feature_name}] 完成 {fold+1} folds")

    return pd.DataFrame(metrics_list)


def print_results_table(all_results):
    """打印四组实验的对比表"""
    print(f"\n{'='*90}")
    print("完整对比实验结果")
    print("=" * 90)
    print(f"\n{'实验组':<25s} {'AUC':>9s} {'F1':>9s} {'Accuracy':>10s} "
          f"{'Precision':>10s} {'Recall':>9s}")
    print("-" * 90)

    summaries = {}
    for name, df_metrics in all_results.items():
        if df_metrics is not None and len(df_metrics) > 0:
            means = df_metrics.mean()
            stds = df_metrics.std()
            summaries[name] = (means, stds)
            print(f"{name:<25s} {means['auc']:9.4f} {means['f1']:9.4f} "
                  f"{means['accuracy']:10.4f} {means['precision']:10.4f} "
                  f"{means['recall']:9.4f}")
            print(f"  (±std){'':<18s} {stds['auc']:9.4f} {stds['f1']:9.4f} "
                  f"{stds['accuracy']:10.4f} {stds['precision']:10.4f} "
                  f"{stds['recall']:9.4f}")
            print()

    # 与基线对比
    if 'A_Baseline' in summaries:
        baseline_auc = summaries['A_Baseline'][0]['auc']
        print(f"\n{'='*90}")
        print("相对于基线的AUC变化")
        print("-" * 90)
        for name in ['B_Physics', 'C_CNN_CBAM', 'D_Multimodal']:
            if name in summaries:
                delta = (summaries[name][0]['auc'] - baseline_auc) * 100
                sign = '+' if delta > 0 else ''
                desc = {
                    'B_Physics': '物理特征工程',
                    'C_CNN_CBAM': 'CNN-CBAM (Zhang+2023方法)',
                    'D_Multimodal': '多模态融合 (原创)',
                }
                print(f"  {name} ({desc[name]}): {sign}{delta:.2f}个百分点")

    return summaries


def main():
    parser = argparse.ArgumentParser(description='完整对比实验')
    parser.add_argument('--data', type=str, default=None,
                        help='CSV数据文件路径')
    parser.add_argument('--simulated', action='store_true',
                        help='使用模拟数据')
    parser.add_argument('--output', type=str, default='./results',
                        help='输出目录')
    parser.add_argument('--folds', type=int, default=5,
                        help='交叉验证折数')
    parser.add_argument('--repeats', type=int, default=2,
                        help='重复次数')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # =====================================================================
    # 0. 数据准备
    # =====================================================================
    if args.data:
        print(f"[0/4] 加载数据: {args.data}")
        df = pd.read_csv(args.data)
    else:
        print("[0/4] 使用模拟数据")
        np.random.seed(42)
        n = 2000
        gamma = pd.DataFrame({
            'NuM1': np.random.poisson(80, n), 'NuM2': np.random.poisson(28, n),
            'NuM3': np.random.poisson(9, n), 'NuM4': np.random.poisson(3, n),
            'NfiltM': np.random.poisson(200, n), 'NhitM': np.random.poisson(250, n),
            'NuW1': np.random.poisson(50, n), 'base': np.abs(np.random.normal(100, 20, n)),
            'rec_Eage': np.abs(np.random.normal(10, 3, n)), 'lable': 1
        })
        proton = pd.DataFrame({
            'NuM1': np.random.poisson(38, n), 'NuM2': np.random.poisson(33, n),
            'NuM3': np.random.poisson(28, n), 'NuM4': np.random.poisson(22, n),
            'NfiltM': np.random.poisson(300, n), 'NhitM': np.random.poisson(350, n),
            'NuW1': np.random.poisson(40, n), 'base': np.abs(np.random.normal(100, 25, n)),
            'rec_Eage': np.abs(np.random.normal(8, 3, n)), 'lable': 0
        })
        df = pd.concat([gamma, proton], ignore_index=True)

    print(f"  样本数: {len(df)}, 伽马: {df['lable'].sum()}, "
          f"质子: {len(df)-df['lable'].sum()}")

    # 特征工程
    df_physics = compute_physics_features(df)
    X_baseline = df[ORIGINAL_FEATURES].values
    X_physics = df_physics[ORIGINAL_FEATURES + PHYSICS_FEATURES].values
    y = df['lable'].values

    cv = RepeatedStratifiedKFold(
        n_splits=args.folds, n_repeats=args.repeats, random_state=42
    )

    # =====================================================================
    # 1. 运行实验
    # =====================================================================
    all_results = {}

    print("\n[1/4] 实验A: Baseline CatBoost (8原始特征)")
    all_results['A_Baseline'] = run_experiment_A_baseline(X_baseline, y, cv)

    print("\n[2/4] 实验B: +Physics Features CatBoost (32特征)")
    all_results['B_Physics'] = run_experiment_B_physics(X_physics, y, cv)

    if HAS_TORCH:
        print("\n[3/4] 实验C: CNN-CBAM (探测器图像)")
        all_results['C_CNN_CBAM'] = run_experiment_C_cnn(X_baseline, y, cv)

        print("\n[4/4] 实验D: Multimodal Fusion (图像+表格)")
        all_results['D_Multimodal'] = run_experiment_D_multimodal(X_baseline, y, cv)
    else:
        print("\n[3/4, 4/4] CNN实验跳过 (需要PyTorch)")

    # =====================================================================
    # 2. 汇总输出
    # =====================================================================
    summaries = print_results_table(all_results)

    # =====================================================================
    # 3. 保存结果
    # =====================================================================
    detail_rows = []
    for name, df_metrics in all_results.items():
        if df_metrics is not None:
            for i, row in df_metrics.iterrows():
                detail_rows.append({
                    'experiment': name, 'fold': i,
                    'accuracy': row['accuracy'], 'f1': row['f1'],
                    'precision': row['precision'], 'recall': row['recall'],
                    'auc': row['auc']
                })
    pd.DataFrame(detail_rows).to_csv(
        f'{args.output}/full_comparison_detail.csv', index=False
    )

    summary_rows = []
    for name, (means, stds) in summaries.items():
        summary_rows.append({
            'experiment': name,
            'auc_mean': means['auc'], 'auc_std': stds['auc'],
            'f1_mean': means['f1'], 'f1_std': stds['f1'],
            'accuracy_mean': means['accuracy'], 'accuracy_std': stds['accuracy'],
        })
    pd.DataFrame(summary_rows).to_csv(
        f'{args.output}/full_comparison_summary.csv', index=False
    )
    print(f"\n结果已保存到: {args.output}/")

    return summaries


if __name__ == "__main__":
    main()
