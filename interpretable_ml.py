"""
=============================================================================
五层可解释性机器学习框架 (Five-Layer Interpretable ML Framework)
应用于 LHAASO 伽马/质子鉴别
=============================================================================

五层架构:
  Layer 1: SHAP 全局+局部解释     → 哪些特征驱动分类? 物理上合理吗?
  Layer 2: 代理决策树 + 规则提取   → 能否用简单规则复现? 与传统Cut方法对比?
  Layer 3: 反事实解释 (个体级)     → 对每个误分类事例, 什么微小改变能翻转预测?
  Layer 4: 物理一致性验证         → SHAP在不同能量/天顶角bin中是否稳定?
  Layer 5: 固有可解释模型 (EBM)   → 可解释模型 vs 黑箱模型性能差距?

论文切入点:
  "From Black Box to Physics: An Interpretable Machine Learning Framework
   for Gamma/Hadron Separation in LHAASO"

核心问题: 天体粒子物理中, ML虽然比传统Cut方法更准, 但物理学家不信任黑箱。
         你的工作就是搭建这座信任的桥梁。

运行:
  python interpretable_ml.py --data your_data.csv
  python interpretable_ml.py --simulated  (演示)
=============================================================================
"""
import numpy as np
import pandas as pd
import warnings
import os
import sys
import argparse
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight

# 导入项目物理特征模块
from physics_features import (compute_physics_features, ORIGINAL_FEATURES,
                               PHYSICS_FEATURES)

# ============================================================================
# 前置: 训练一个强黑箱模型 (所有可解释性分析的基础)
# ============================================================================

def train_blackbox_model(X, y, model_type='catboost'):
    """
    训练强分类器作为后续所有可解释性分析的目标模型。

    支持: CatBoost (推荐, 天然提供特征重要性), XGBoost, RandomForest
    """
    if model_type == 'catboost':
        from catboost import CatBoostClassifier
        cw = compute_class_weight('balanced', classes=[0, 1], y=y)
        model = CatBoostClassifier(
            iterations=500, depth=8, learning_rate=0.05,
            class_weights={0: cw[0], 1: cw[1]},
            verbose=0, random_seed=42, early_stopping_rounds=50,
        )
    elif model_type == 'xgboost':
        from xgboost import XGBClassifier
        ratio = float(np.sum(y == 0)) / np.sum(y == 1)
        model = XGBClassifier(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            scale_pos_weight=ratio, random_state=42, verbosity=0,
        )
    elif model_type == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1,
            class_weight='balanced',
        )
    else:
        raise ValueError(f"Unknown model: {model_type}")

    model.fit(X, y)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    print(f"  [{model_type}] Accuracy={accuracy_score(y, y_pred):.4f}, "
          f"F1={f1_score(y, y_pred):.4f}, AUC={roc_auc_score(y, y_prob):.4f}")
    return model


# ============================================================================
# Layer 1: SHAP 全局 + 局部可解释性分析
# ============================================================================

class SHAPInterpreter:
    """
    Layer 1: SHAP 分析

    回答三个问题:
      Q1 (全局): 哪些特征对鉴别贡献最大? → bar plot, beeswarm plot
      Q2 (局部): 为什么这个事例被判定为伽马? → waterfall plot
      Q3 (交互): 特征之间如何共同作用? → dependence plot

    物理验证: 排名靠前的特征是否在物理上是合理的?
      预期: muon_hit_ratio, core_concentration 等物理特征应排名靠前
    """

    def __init__(self, model, feature_names, X_background=None):
        self.model = model
        self.feature_names = feature_names

    def compute_shap_values(self, X, max_samples=500):
        """计算SHAP值"""
        import shap

        # 使用KernelExplainer或TreeExplainer(更快)
        if hasattr(self.model, 'get_booster'):  # XGBoost
            explainer = shap.TreeExplainer(self.model)
        elif hasattr(self.model, 'feature_importances_'):  # CatBoost, RF, etc.
            explainer = shap.TreeExplainer(self.model)
        else:
            # 通用explainer (较慢)
            X_sample = X[:min(100, len(X))]
            explainer = shap.KernelExplainer(
                self.model.predict_proba, X_sample
            )

        # 对子集计算SHAP值 (全量数据会很慢)
        indices = np.random.choice(len(X), min(max_samples, len(X)), replace=False)
        X_subset = X[indices]
        shap_values = explainer.shap_values(X_subset)

        # shap_values可能是 [n, p] 或 [n, p, 2] (二分类时)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # 取正类(伽马)的SHAP
        elif shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]  # [n, p, 2] -> 取正类

        return shap_values, X_subset, explainer

    def global_feature_importance(self, X, output_dir='./results'):
        """全局特征重要性: 排名 + 物理合理性检查"""
        import shap
        os.makedirs(output_dir, exist_ok=True)

        shap_values, X_subset, explainer = self.compute_shap_values(X)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        # 排序
        rankings = sorted(zip(self.feature_names, mean_abs_shap),
                          key=lambda x: x[1], reverse=True)

        print(f"\n{'='*70}")
        print("Layer 1: SHAP 全局特征重要性 (Top 15)")
        print("=" * 70)
        print(f"{'排名':<6s}{'特征名':<30s}{'|SHAP|均值':<14s}{'类型':<12s}{'物理合理性'}")
        print("-" * 80)

        # 物理特征标记
        physics_expected = {
            'muon_hit_ratio': '质子μ子更多',
            'core_concentration': '伽马更集中',
            'total_asymmetry': '质子更不对称',
            'lateral_spread': '质子更分散',
            'outer_muon_ratio': '质子外围μ子多',
            'compactness': '伽马更紧密',
            'gamma_likeness': '综合伽马似然',
            'ring_ratio_21': '径向轮廓',
            'core_to_outer_ratio': '核心-外围比',
            'discriminant_C': '传统判别量',
        }

        for i, (name, imp) in enumerate(rankings[:15]):
            ftype = '物理衍生' if name in PHYSICS_FEATURES else '原始'
            phys_check = physics_expected.get(name, '—')
            marker = ' ✓' if name in physics_expected else ''
            print(f"  {i+1:<4d} {name:<30s} {imp:<14.6f} {ftype:<12s} {phys_check}{marker}")

        # 物理特征贡献汇总
        physics_total = sum(v for n, v in rankings if n in PHYSICS_FEATURES)
        raw_total = sum(v for n, v in rankings if n not in PHYSICS_FEATURES)
        print(f"\n  原始特征贡献: {raw_total:.4f} ({raw_total/(raw_total+physics_total)*100:.1f}%)")
        print(f"  物理特征贡献: {physics_total:.4f} ({physics_total/(raw_total+physics_total)*100:.1f}%)")

        # 保存
        df_rank = pd.DataFrame(rankings, columns=['feature', 'mean_|SHAP|'])
        df_rank['type'] = df_rank['feature'].apply(
            lambda x: 'physics' if x in PHYSICS_FEATURES else 'raw'
        )
        df_rank.to_csv(f'{output_dir}/shap_global_importance.csv', index=False)

        self._shap_values = shap_values
        self._X_subset = X_subset
        self._explainer = explainer
        return df_rank

    def local_explanation(self, X, event_indices, event_labels=None):
        """
        局部解释: 对指定事例生成 Waterfall 式的解释

        参数:
            X:               特征矩阵
            event_indices:   要解释的事例索引列表
            event_labels:    对应的真实标签 (可选, 用于标注)

        返回:
            对每个事例的解释文本
        """
        import shap

        if not hasattr(self, '_explainer'):
            shap_values, X_subset, self._explainer = self.compute_shap_values(X)
            self._shap_values = shap_values
            self._X_subset = X_subset

        explanations = []
        for idx in event_indices:
            exp = shap.Explanation(
                values=self._shap_values[idx],
                base_values=self._explainer.expected_value
                if not isinstance(self._explainer.expected_value, list)
                else self._explainer.expected_value[1],
                data=self._X_subset[idx],
                feature_names=self.feature_names,
            )

            # 提取Top contributing features
            top_features = []
            for i in np.argsort(np.abs(exp.values))[-8:][::-1]:
                feat_name = self.feature_names[i]
                feat_val = exp.data[i]
                shap_val = exp.values[i]
                direction = '→ gamma' if shap_val > 0 else '→ proton'
                top_features.append(
                    f"    {feat_name:<25s} = {feat_val:10.4f}  |  SHAP={shap_val:+8.4f}  {direction}"
                )

            label_str = f" (True: {'gamma' if event_labels[idx]==1 else 'proton'})" \
                        if event_labels is not None else ""
            exp_text = f"\n  Event #{idx}{label_str}:\n" + "\n".join(top_features)
            explanations.append(exp_text)

        return "\n".join(explanations)

    def dependence_analysis(self, X, top_n=5, output_dir='./results'):
        """
        依赖图分析: 发掘特征间的交互效应

        物理意义: SHAP dependence plot 可以揭示两个特征如何共同影响预测
        例如: muon_hit_ratio 的SHAP值如何随 energy 变化?
              → 是否有能量依赖的 μ子含量判定偏差?
        """
        if not hasattr(self, '_shap_values'):
            shap_values, X_subset, _ = self.compute_shap_values(X)
            self._shap_values = shap_values
            self._X_subset = X_subset

        mean_abs = np.abs(self._shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs)[-top_n:][::-1]

        print(f"\n{'='*70}")
        print(f"Layer 1b: SHAP 依赖分析 (Top {top_n} 特征)")
        print("=" * 70)

        interactions = []
        for rank, feat_idx in enumerate(top_indices):
            feat_name = self.feature_names[feat_idx]
            shap_vals = self._shap_values[:, feat_idx]
            feat_vals = self._X_subset[:, feat_idx]

            # 寻找与当前特征SHAP值相关性最强的"第二特征"
            correlations = []
            for j in range(len(self.feature_names)):
                if j != feat_idx:
                    corr = np.corrcoef(shap_vals,
                                       self._X_subset[:, j])[0, 1]
                    correlations.append((j, abs(corr)))

            best_interact = max(correlations, key=lambda x: x[1])
            interact_name = self.feature_names[best_interact[0]]
            interact_corr = best_interact[1]

            # 描述特征的物理趋势
            slope = np.polyfit(feat_vals, shap_vals, 1)[0]
            trend_desc = ('SHAP↑ 随特征值↑' if slope > 0 else 'SHAP↓ 随特征值↑')

            print(f"  [{rank+1}] {feat_name}")
            print(f"      {trend_desc}")
            print(f"      最强交互: {interact_name} (|r|={interact_corr:.3f})")

            interactions.append({
                'rank': rank + 1, 'feature': feat_name,
                'trend': trend_desc,
                'interaction_with': interact_name,
                'interaction_corr': interact_corr,
            })

        pd.DataFrame(interactions).to_csv(
            f'{output_dir}/shap_dependence.csv', index=False
        )
        return interactions


# ============================================================================
# Layer 2: 代理决策树 + 物理规则提取
# ============================================================================

class SurrogateRuleExtractor:
    """
    Layer 2: 代理决策树规则提取

    核心思路:
      1. 用黑箱模型的预测作为"标签"训练一个浅层决策树
      2. 从决策树中提取人类可读的物理规则
      3. 将规则与传统Cut-Based方法中的判据对比

    价值: 桥接黑箱ML与传统物理分析方法
    """

    def __init__(self, max_depth=4, min_samples_leaf=50):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.surrogate = None
        self.rules = []

    def fit(self, X, y_blackbox_pred, feature_names):
        """
        训练代理决策树

        参数:
            X:                原始特征
            y_blackbox_pred:  黑箱模型的预测 (0/1)
            feature_names:    特征名列表
        """
        self.feature_names = feature_names
        self.surrogate = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
        )
        self.surrogate.fit(X, y_blackbox_pred)

        # 评估保真度 (surrogate复现黑箱预测的程度)
        y_surrogate = self.surrogate.predict(X)
        fidelity = accuracy_score(y_blackbox_pred, y_surrogate)

        print(f"\n{'='*70}")
        print("Layer 2: 代理决策树规则提取")
        print("=" * 70)
        print(f"  代理树深度: {self.max_depth}")
        print(f"  代理树保真度: {fidelity:.4f} (越高越好, >0.90说明规则有效)")

        # 提取文本规则
        self.tree_text = export_text(
            self.surrogate, feature_names=feature_names,
            max_depth=self.max_depth,
        )
        print(f"\n  --- 提取的决策规则 ---\n{self.tree_text}")

        return self

    def extract_physics_rules(self, X, y_true, feature_names=None):
        """
        将决策树规则翻译为物理语言, 并与真实标签比较

        返回: 每条规则的物理含义 + 纯度(伽马纯度/质子纯度)
        """
        if feature_names is None:
            feature_names = self.feature_names

        # 获取每个样本所在的叶节点
        leaf_ids = self.surrogate.apply(X)

        print(f"\n{'='*70}")
        print("Layer 2b: 物理规则翻译")
        print("=" * 70)

        # 分析每个叶节点
        unique_leaves = np.unique(leaf_ids)
        physics_rules = []

        for leaf_id in unique_leaves:
            mask = leaf_ids == leaf_id
            n_samples = mask.sum()
            if n_samples < 10:
                continue

            y_leaf = y_true[mask]
            gamma_fraction = y_leaf.mean()
            n_gamma = y_leaf.sum()
            n_proton = n_samples - n_gamma

            # 从决策树路径中提取规则条件
            # 遍历树结构找到到达此叶节点的路径
            path = self._get_leaf_path(leaf_id)

            rule_str = " AND ".join(path) if path else "ROOT"
            if gamma_fraction > 0.5:
                label_type = f"γ-rich ({gamma_fraction:.1%})"
            else:
                label_type = f"p-rich ({(1-gamma_fraction):.1%})"

            # 物理翻译
            physics_translation = self._translate_to_physics(path, gamma_fraction)

            print(f"\n  Leaf {leaf_id}: {n_samples} events, {label_type}")
            print(f"    规则: {rule_str}")
            print(f"    物理: {physics_translation}")
            print(f"    γ={n_gamma}, p={n_proton}")

            physics_rules.append({
                'leaf_id': leaf_id,
                'n_samples': n_samples,
                'gamma_fraction': gamma_fraction,
                'rule': rule_str,
                'physics': physics_translation,
            })

        return physics_rules

    def _get_leaf_path(self, leaf_id):
        """获取到达叶节点的决策路径"""
        # 获取树结构
        children_left = self.surrogate.tree_.children_left
        children_right = self.surrogate.tree_.children_right
        feature = self.surrogate.tree_.feature
        threshold = self.surrogate.tree_.threshold

        # 从根节点向下找leaf_id
        path = []

        def dfs(node, conditions):
            if node == leaf_id:
                path.extend(conditions)
                return True
            if children_left[node] != -1:  # 不是叶节点
                left_cond = f"{self.feature_names[feature[node]]} <= {threshold[node]:.4f}"
                if dfs(children_left[node], conditions + [left_cond]):
                    return True
                right_cond = f"{self.feature_names[feature[node]]} > {threshold[node]:.4f}"
                if dfs(children_right[node], conditions + [right_cond]):
                    return True
            return False

        dfs(0, [])
        return path

    def _translate_to_physics(self, path, gamma_fraction):
        """将决策路径翻译为物理语言"""
        translations = {
            'muon_hit_ratio': 'μ子含量',
            'core_concentration': '核心集中度',
            'total_asymmetry': '总不对称性',
            'lateral_spread': '横向扩展',
            'outer_muon_ratio': '外围μ子比',
            'compactness': '簇射紧密度',
            'ring_ratio_21': '径向轮廓陡度',
            'NhitM': '总命中数',
            'NuM_total': '总μ子计数',
        }

        parts = []
        for condition in path:
            for feat_key, phys_name in translations.items():
                if feat_key in condition:
                    direction = '高' if '>' in condition else '低'
                    parts.append(f"{phys_name}{direction}")
                    break
            else:
                parts.append(condition)

        if gamma_fraction > 0.5:
            return "伽马型簇射: " + ", ".join(parts)
        else:
            return "质子型簇射: " + ", ".join(parts)

    def compare_with_traditional_cuts(self, X, y_true, feature_names=None):
        """
        将代理树规则与传统Cut-Based方法的判别变量比较

        LHAASO 传统方法通常基于:
          - μ子含量阈值 (如 Nμ/Nhit < 0.X → 伽马)
          - 横向扩展阈值

        这里比较: 代理规则给出的阈值 vs 典型Cut-Based阈值
        """
        if feature_names is None:
            feature_names = self.feature_names

        thresholds = self.surrogate.tree_.threshold
        features = self.surrogate.tree_.feature

        print(f"\n{'='*70}")
        print("Layer 2c: 代理规则 vs 传统Cut方法对比")
        print("=" * 70)
        print(f"{'特征':<25s}{'代理规则阈值':<15s}{'传统Cut参考':<15s}{'一致性'}")
        print("-" * 70)

        # 提取所有分裂阈值
        extracted = {}
        for i in range(len(features)):
            if features[i] >= 0:  # 不是叶节点
                feat_name = feature_names[features[i]]
                if feat_name not in extracted:
                    extracted[feat_name] = []
                extracted[feat_name].append(thresholds[i])

        # 与传统Cut方法比较
        traditional_refs = {
            'muon_hit_ratio': '< 0.3',
            'core_concentration': '> 0.5',
            'total_asymmetry': '< 0.3',
            'lateral_spread': '< 2.0',
        }

        for feat_name, thresh_list in extracted.items():
            if feat_name in traditional_refs:
                avg_thresh = np.mean(thresh_list)
                trad = traditional_refs[feat_name]
                # 简单一致性检查
                if '>' in trad:
                    trad_val = float(trad.replace('>', '').strip())
                    # 检查代理阈值是否在同方向
                    direction_match = any(t > 0 for t in thresh_list)
                else:
                    trad_val = float(trad.replace('<', '').strip())
                    direction_match = True

                print(f"  {feat_name:<25s}{avg_thresh:<15.4f}{trad:<15s}"
                      f"{'✓ 一致' if direction_match else '? 需检查'}")
            else:
                print(f"  {feat_name:<25s}{np.mean(thresh_list):<15.4f}{'—':<15s}{'—'}")


# ============================================================================
# Layer 3: 反事实解释 (Counterfactual Explanations)
# ============================================================================

class CounterfactualExplainer:
    """
    Layer 3: 反事实解释

    核心问题: "对这个质子, 它的什么特征改变最小就能被判定为伽马?"

    物理价值:
      - 揭示模型的决策边界在物理空间中是什么样
      - 发现模型是否依赖于"非物理"的特征组合来做决策
      - 对误分类事例, 给出"为什么错了, 需要改变什么"的物理解释
    """

    def __init__(self, model, feature_names, X_train, y_train,
                 epsilon=0.1, max_iter=100):
        self.model = model
        self.feature_names = feature_names
        self.epsilon = epsilon
        self.max_iter = max_iter

        # 计算特征范围用于约束反事实
        self.X_min = X_train.min(axis=0)
        self.X_max = X_train.max(axis=0)
        self.X_std = X_train.std(axis=0)
        self.y_train = y_train

    def find_counterfactual(self, x, target_class=1, n_features_to_change=3):
        """
        为单个事例找到最小改变的反事实

        策略: 贪心改变对目标类贡献最大的特征

        参数:
            x:                   原始特征向量 [n_features]
            target_class:        目标类别 (1=gamma)
            n_features_to_change: 最多改变几个特征

        返回:
            counterfactual:      反事实特征向量
            changes:             [(特征名, 原值, 新值, 改变量(σ单位))]
            success:             是否成功翻转预测
        """
        x_cf = x.copy()
        prob_orig = self.model.predict_proba([x])[0, 1]

        # 如果原始预测已经是目标类, 不需要反事实
        pred_orig = self.model.predict([x])[0]
        if pred_orig == target_class:
            return x_cf, [], True, prob_orig

        # 计算每个特征的梯度 (对目标类概率的)
        gradients = self._compute_feature_gradient(x, target_class)
        # 排序: 哪些特征的改变最能增加目标类概率
        feature_order = np.argsort(np.abs(gradients))[::-1]

        changes = []
        success = False

        for rank, feat_idx in enumerate(feature_order[:n_features_to_change]):
            original_val = x_cf[feat_idx]
            grad = gradients[feat_idx]

            # 调整方向: 朝增加目标类概率的方向
            if grad > 0:
                new_val = min(original_val * (1 + self.epsilon),
                             self.X_max[feat_idx])
            else:
                new_val = max(original_val * (1 - self.epsilon),
                             self.X_min[feat_idx])

            x_cf[feat_idx] = new_val
            change_sigma = (new_val - original_val) / (self.X_std[feat_idx] + 1e-8)
            changes.append((
                self.feature_names[feat_idx],
                original_val, new_val, change_sigma
            ))

            # 检查是否翻转
            prob_new = self.model.predict_proba([x_cf])[0, 1]
            if self.model.predict([x_cf])[0] == target_class:
                success = True
                break

        return x_cf, changes, success, prob_orig

    def _compute_feature_gradient(self, x, target_class):
        """通过数值扰动估计特征梯度"""
        gradients = np.zeros(len(x))
        prob_base = self.model.predict_proba([x])[0, target_class]

        for i in range(len(x)):
            x_perturbed = x.copy()
            x_perturbed[i] *= (1 + self.epsilon)
            prob_perturbed = self.model.predict_proba([x_perturbed])[0, target_class]
            gradients[i] = (prob_perturbed - prob_base) / (
                x_perturbed[i] - x[i] + 1e-8
            )

        return gradients

    def analyze_misclassifications(self, X, y_true, max_examples=10,
                                    output_dir='./results'):
        """
        系统分析误分类事例的反事实解释

        价值: 发现模型失败的模式 → 指导特征工程改进
        """
        y_pred = self.model.predict(X)
        misclassified = np.where(y_pred != y_true)[0]

        if len(misclassified) == 0:
            print("\n  无误分类事例!")
            return []

        n_analyze = min(max_examples, len(misclassified))
        selected = np.random.choice(misclassified, n_analyze, replace=False)

        print(f"\n{'='*70}")
        print("Layer 3: 反事实解释 — 误分类分析")
        print("=" * 70)
        print(f"  总误分类: {len(misclassified)}/{len(X)}")
        print(f"  分析样本: {n_analyze}")

        analyses = []
        failure_patterns = {'high_muon_as_gamma': 0, 'low_concentration_as_gamma': 0,
                            'other': 0}

        for rank, idx in enumerate(selected):
            x = X[idx]
            true_label = y_true[idx]
            pred_label = y_pred[idx]
            target = 1 - pred_label  # 想要翻转到正确类别

            _, changes, success, prob_orig = self.find_counterfactual(
                x, target_class=target, n_features_to_change=5
            )

            if success:
                desc = f"Event #{idx}: True={'γ' if true_label==1 else 'p'}, " \
                       f"Pred={'γ' if pred_label==1 else 'p'} → 可翻转"
            else:
                desc = f"Event #{idx}: True={'γ' if true_label==1 else 'p'}, " \
                       f"Pred={'γ' if pred_label==1 else 'p'} → 需大幅改变"

            # 分类失败模式
            if true_label == 0 and pred_label == 1:  # 质子→伽马 (误报)
                # 检查是否因为μ子含量低
                muon_idx = list(self.feature_names).index('muon_hit_ratio') \
                    if 'muon_hit_ratio' in self.feature_names else None
                if muon_idx is not None and X[idx, muon_idx] < np.median(X[:, muon_idx]):
                    pattern = 'low_muon_proton_as_gamma'
                else:
                    pattern = 'other_fp'
            elif true_label == 1 and pred_label == 0:  # 伽马→质子 (漏报)
                pattern = 'gamma_as_proton_fn'
            else:
                pattern = 'correct'

            if rank < 5:  # 只打印前5个样例
                print(f"\n  [{rank+1}] {desc}")
                print(f"    原始概率(P=γ): {prob_orig:.4f}")
                if changes:
                    for feat_name, old_val, new_val, sigma in changes[:3]:
                        direction = '↑' if new_val > old_val else '↓'
                        print(f"    改变 {feat_name}: {old_val:.4f} → {new_val:.4f} "
                              f"({direction}{abs(sigma):.1f}σ)")

            analyses.append({
                'event_idx': idx, 'true_label': true_label,
                'pred_label': pred_label, 'pattern': pattern,
                'counterfactual_success': success,
                'top_change_feature': changes[0][0] if changes else 'N/A',
            })

        # 失败模式汇总
        print(f"\n  --- 失败模式分布 ---")
        for pattern, count in pd.DataFrame(analyses)['pattern'].value_counts().items():
            print(f"    {pattern}: {count}")

        pd.DataFrame(analyses).to_csv(
            f'{output_dir}/counterfactual_analysis.csv', index=False
        )
        return analyses


# ============================================================================
# Layer 4: 物理一致性验证
# ============================================================================

class PhysicsConsistencyValidator:
    """
    Layer 4: 物理一致性验证

    问题: 模型的SHAP/决策逻辑在不同物理条件下是否一致?

    在天体粒子物理中, 一个好的鉴别器应该在:
      - 不同能量bin中表现相似
      - 不同天顶角bin中表现相似
      - 物理特征的重要性不应出现意外翻转

    如果发现不一致 → 说明模型学到了非物理的伪关联 → 需要修正
    """

    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def validate_across_bins(self, X, y, bin_feature_name, bin_feature_idx,
                              n_bins=5, output_dir='./results'):
        """
        在某个物理量的不同bin中验证模型行为一致性

        参数:
            bin_feature_name: 用于分bin的特征 (如 'rec_Eage', 'NhitM')
            bin_feature_idx:  特征在X中的索引
            n_bins:           分bin数

        返回:
            各bin的性能指标和特征重要性排名的稳定性
        """
        bin_edges = np.percentile(X[:, bin_feature_idx],
                                   np.linspace(0, 100, n_bins + 1))
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        print(f"\n{'='*70}")
        print(f"Layer 4: 物理一致性验证 — 按 {bin_feature_name} 分bin")
        print("=" * 70)

        bin_results = []
        for i in range(n_bins):
            mask = (X[:, bin_feature_idx] > bin_edges[i]) & \
                   (X[:, bin_feature_idx] <= bin_edges[i + 1])
            X_bin = X[mask]
            y_bin = y[mask]

            if len(X_bin) < 20:
                continue
            if len(np.unique(y_bin)) < 2:
                continue  # 该bin只有一种类别, 跳过

            y_pred = self.model.predict(X_bin)
            y_prob = self.model.predict_proba(X_bin)[:, 1]

            bin_results.append({
                'bin': i + 1,
                'range': f'[{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f}]',
                'n_samples': len(X_bin),
                'gamma_fraction': y_bin.mean(),
                'accuracy': accuracy_score(y_bin, y_pred),
                'auc': roc_auc_score(y_bin, y_prob),
                'f1': f1_score(y_bin, y_pred),
            })

            print(f"  Bin {i+1} {bin_results[-1]['range']:<20s}: "
                  f"n={len(X_bin):5d}, γ%={y_bin.mean():.2f}, "
                  f"AUC={bin_results[-1]['auc']:.4f}")

        # 检查AUC稳定性
        aucs = [r['auc'] for r in bin_results]
        auc_std = np.std(aucs)

        print(f"\n  AUC跨bin标准差: {auc_std:.4f}")
        if auc_std < 0.02:
            print("  ✓ 模型在不同{bin_feature_name}区间表现一致")
        elif auc_std < 0.05:
            print(f"  ⚠ 存在一定bin依赖性, 最高AUC bin与最低AUC bin差{max(aucs)-min(aucs):.4f}")
        else:
            print(f"  ✗ 模型在不同{bin_feature_name}区间表现差异大, 可能存在伪关联")

        pd.DataFrame(bin_results).to_csv(
            f'{output_dir}/consistency_{bin_feature_name}.csv', index=False
        )
        return bin_results

    def shap_stability_across_bins(self, X, y, bin_feature_name,
                                    bin_feature_idx, n_bins=5,
                                    output_dir='./results'):
        """SHAP特征重要性跨bin稳定性"""
        import shap

        # 先尝试使用TreeExplainer
        try:
            explainer = shap.TreeExplainer(self.model)
        except Exception:
            explainer = shap.KernelExplainer(
                self.model.predict_proba,
                X[:min(50, len(X))]
            )

        bin_edges = np.percentile(X[:, bin_feature_idx],
                                   np.linspace(0, 100, n_bins + 1))
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        print(f"\n{'='*70}")
        print(f"Layer 4b: SHAP稳定性 — 按 {bin_feature_name}")
        print("=" * 70)

        all_rankings = {}
        for i in range(n_bins):
            mask = (X[:, bin_feature_idx] > bin_edges[i]) & \
                   (X[:, bin_feature_idx] <= bin_edges[i + 1])
            X_bin = X[mask]

            if len(X_bin) < 20:
                continue

            X_sample = X_bin[:min(100, len(X_bin))]
            shap_vals = explainer.shap_values(X_sample)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            elif shap_vals.ndim == 3:
                shap_vals = shap_vals[:, :, 1]

            mean_abs = np.abs(shap_vals).mean(axis=0)
            rankings = sorted(zip(self.feature_names, mean_abs),
                              key=lambda x: x[1], reverse=True)
            all_rankings[f'Bin_{i+1}'] = [name for name, _ in rankings[:10]]

        # 计算排名稳定性 (Top-5重叠度)
        if len(all_rankings) >= 2:
            bins_list = list(all_rankings.keys())
            overlap_sets = []
            for i in range(len(bins_list)):
                for j in range(i + 1, len(bins_list)):
                    overlap = len(set(all_rankings[bins_list[i]][:5]) &
                                  set(all_rankings[bins_list[j]][:5]))
                    overlap_sets.append(overlap / 5.0)

            avg_overlap = np.mean(overlap_sets)
            print(f"\n  Top-5特征跨bin平均重叠率: {avg_overlap:.1%}")
            if avg_overlap >= 0.8:
                print("  ✓ 特征重要性高度稳定，模型学到的判据具有物理一致性")
            else:
                print("  ⚠ 特征重要性不稳定，不同bin依赖不同特征")
                print("     → 建议: 对不同bin分别训练模型或添加bin信息作为特征")

        return all_rankings


# ============================================================================
# Layer 5: 固有可解释模型 (EBM)
# ============================================================================

class InterpretableByDesign:
    """
    Layer 5: 固有可解释模型 — Explainable Boosting Machine (EBM)

    EBM = 广义加性模型 (GAM) + Boosting
      f(x) = β₀ + Σ fᵢ(xᵢ) + Σ fᵢⱼ(xᵢ, xⱼ)

    每个特征xᵢ的贡献fᵢ是一个学习到的平滑函数 (可以直接画出来!)
    每个交互fᵢⱼ也可以直接可视化

    优势: 模型本身就是解释, 不需要SHAP等事后解释
    代价: 通常比黑箱模型略差 (AUC差0.01-0.02)

    创新点: 比较EBM vs 黑箱的性能差距 → 判断"可解释性的代价"
    """

    def __init__(self):
        pass

    def train_and_compare(self, X, y, X_test, y_test, feature_names):
        """
        训练EBM并与CatBoost黑箱对比

        返回值:
            ebm模型, 黑箱模型, 性能对比
        """
        from interpret.glassbox import ExplainableBoostingClassifier

        print(f"\n{'='*70}")
        print("Layer 5: 固有可解释模型 (EBM) vs 黑箱")
        print("=" * 70)

        # EBM 训练
        print("  训练EBM...")
        ebm = ExplainableBoostingClassifier(
            interactions=5,       # 最多5个交互项
            max_rounds=5000,
            early_stopping_rounds=50,
            random_state=42,
        )
        ebm.fit(X, y)

        # 黑箱训练
        print("  训练CatBoost...")
        blackbox = train_blackbox_model(X, y, model_type='catboost')

        # 评估
        ebm_pred = ebm.predict(X_test)
        ebm_prob = ebm.predict_proba(X_test)[:, 1]
        bb_pred = blackbox.predict(X_test)
        bb_prob = blackbox.predict_proba(X_test)[:, 1]

        ebm_auc = roc_auc_score(y_test, ebm_prob)
        bb_auc = roc_auc_score(y_test, bb_prob)
        auc_cost = bb_auc - ebm_auc

        print(f"\n  {'Model':<20s}{'AUC':>8s}{'F1':>8s}{'Accuracy':>10s}")
        print(f"  {'-'*45}")
        print(f"  {'EBM (可解释)':<20s}{ebm_auc:8.4f}"
              f"{f1_score(y_test, ebm_pred):8.4f}{accuracy_score(y_test, ebm_pred):10.4f}")
        print(f"  {'CatBoost (黑箱)':<20s}{bb_auc:8.4f}"
              f"{f1_score(y_test, bb_pred):8.4f}{accuracy_score(y_test, bb_pred):10.4f}")

        print(f"\n  可解释性代价 (ΔAUC): {auc_cost:.4f}")
        if auc_cost < 0.01:
            print("  ✓ EBM几乎不牺牲性能, 推荐直接使用EBM!")
        elif auc_cost < 0.02:
            print("  ⚖ 轻微性能损失, 用可解释性换取是可接受的")
        else:
            print("  ⚠ 性能损失较大, 推荐使用黑箱+SHAP的组合方案")

        # EBM特征重要性 (直接可解释)
        ebm_global = ebm.explain_global()
        feature_importances = ebm_global.data()['scores']

        print(f"\n  EBM 特征重要性 (Top 10):")
        for i, (name, score) in enumerate(
            sorted(zip(feature_names, feature_importances),
                   key=lambda x: abs(x[1]), reverse=True)[:10]
        ):
            print(f"    {i+1:2d}. {name:<30s} {score:+.4f}")

        return ebm, blackbox, auc_cost


# ============================================================================
# 主分析流程
# ============================================================================

def run_full_interpretability_analysis(data_path=None, use_simulated=True,
                                        output_dir='./results'):
    """
    运行完整的五层可解释性分析
    """
    os.makedirs(output_dir, exist_ok=True)

    # ================================================================
    # 0. 数据准备
    # ================================================================
    if data_path and os.path.exists(data_path):
        print(f"[0/5] 加载数据: {data_path}")
        df = pd.read_csv(data_path)
    else:
        print("[0/5] 使用模拟数据")
        np.random.seed(42)
        n = 3000
        # 使用更大的涨落使模拟数据更接近真实
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
        # 添加一部分边界样本 (物理上模糊的事例)
        border = pd.DataFrame({
            'NuM1': np.random.poisson(55, n // 3),
            'NuM2': np.random.poisson(32, n // 3),
            'NuM3': np.random.poisson(18, n // 3),
            'NuM4': np.random.poisson(12, n // 3),
            'NfiltM': np.random.poisson(245, n // 3),
            'NhitM': np.random.poisson(290, n // 3),
            'NuW1': np.random.poisson(50, n // 3),
            'base': np.abs(np.random.normal(100, 25, n // 3)),
            'rec_Eage': np.abs(np.random.normal(9.5, 5, n // 3)),
            'lable': np.random.choice([0, 1], n // 3),
        })
        df = pd.concat([gamma, proton, border], ignore_index=True)

    # 计算物理特征
    df = compute_physics_features(df)
    all_features = ORIGINAL_FEATURES + PHYSICS_FEATURES
    X = df[all_features].values
    y = df['lable'].values

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 训练/测试分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  训练集: {len(X_train)}, 测试集: {len(X_test)}")
    print(f"  特征维数: {X.shape[1]}  (8原始 + 24物理)")

    # ================================================================
    # 核心: 训练目标黑箱模型
    # ================================================================
    print(f"\n{'='*70}")
    print("训练目标模型 (所有解释分析的基础)")
    print("=" * 70)
    bb_model = train_blackbox_model(X_train, y_train, 'catboost')

    # 测试集性能
    y_test_pred = bb_model.predict(X_test)
    y_test_prob = bb_model.predict_proba(X_test)[:, 1]
    print(f"  测试集: AUC={roc_auc_score(y_test, y_test_prob):.4f}, "
          f"F1={f1_score(y_test, y_test_pred):.4f}")

    # ================================================================
    # Layer 1: SHAP分析
    # ================================================================
    interpreter = SHAPInterpreter(bb_model, all_features)
    df_importance = interpreter.global_feature_importance(X_test, output_dir)

    # 局部解释几个代表性事例
    correct_gamma = np.where((y_test == 1) & (y_test_pred == 1))[0]
    correct_proton = np.where((y_test == 0) & (y_test_pred == 0))[0]
    misclassified = np.where(y_test != y_test_pred)[0]

    examples = []
    if len(correct_gamma) > 0:
        examples.append(correct_gamma[0])
    if len(correct_proton) > 0:
        examples.append(correct_proton[0])
    if len(misclassified) > 0:
        examples.append(misclassified[0])

    if examples:
        exp_text = interpreter.local_explanation(X_test, examples, y_test)
        print(f"\n{'='*70}")
        print("Layer 1c: SHAP 局部解释 (个体事例)")
        print("=" * 70)
        print(exp_text)

    interpreter.dependence_analysis(X_test, top_n=3, output_dir=output_dir)

    # ================================================================
    # Layer 2: 代理规则提取
    # ================================================================
    y_bb_pred = bb_model.predict(X_train)
    surrogate = SurrogateRuleExtractor(max_depth=4)
    surrogate.fit(X_train, y_bb_pred, all_features)
    surrogate.extract_physics_rules(X_test, y_test)
    surrogate.compare_with_traditional_cuts(X_test, y_test)

    # ================================================================
    # Layer 3: 反事实解释
    # ================================================================
    cf = CounterfactualExplainer(bb_model, all_features, X_train, y_train)
    cf.analyze_misclassifications(X_test, y_test, output_dir=output_dir)

    # ================================================================
    # Layer 4: 物理一致性验证
    # ================================================================
    validator = PhysicsConsistencyValidator(bb_model, all_features)

    # 按NhitM (总命中数, 能量代理量) 分bin
    nhit_idx = list(all_features).index('NhitM')
    validator.validate_across_bins(
        X_test, y_test, 'NhitM', nhit_idx, n_bins=5, output_dir=output_dir
    )
    validator.shap_stability_across_bins(
        X_test, y_test, 'NhitM', nhit_idx, n_bins=5, output_dir=output_dir
    )

    # ================================================================
    # Layer 5: 固有可解释模型
    # ================================================================
    try:
        ebm_agent = InterpretableByDesign()
        ebm_model, bb, auc_cost = ebm_agent.train_and_compare(
            X_train, y_train, X_test, y_test, all_features
        )
    except ImportError:
        print("\n[Layer 5] EBM需要 'interpret' 包, 跳过 (pip install interpret)")

    # ================================================================
    # 总结
    # ================================================================
    print(f"\n{'='*70}")
    print("五层可解释性分析完成!")
    print("=" * 70)
    print(f"\n  结果文件保存在: {output_dir}/")
    print(f"    - shap_global_importance.csv   (Layer 1: SHAP全局重要性)")
    print(f"    - shap_dependence.csv          (Layer 1b: 特征交互)")
    print(f"    - counterfactual_analysis.csv  (Layer 3: 反事实分析)")
    print(f"    - consistency_NhitM.csv        (Layer 4: 物理一致性)")

    print(f"\n  论文写作切入点:")
    print(f"    1. SHAP揭示模型依赖的物理特征 → 验证物理合理性")
    print(f"    2. 代理规则与Cut-Based方法对比 → 桥接传统与现代")
    print(f"    3. 反事实分析揭示失败模式 → 指导特征工程")
    print(f"    4. 跨能量/角度bin一致性 → 模型稳健性证明")
    print(f"    5. EBM vs 黑箱 → 量化可解释性代价")

    return {
        'importance': df_importance,
        'model': bb_model,
        'scaler': scaler,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='五层可解释性ML分析'
    )
    parser.add_argument('--data', type=str, default=None,
                        help='CSV数据路径')
    parser.add_argument('--simulated', action='store_true',
                        help='使用模拟数据')
    parser.add_argument('--output', type=str, default='./results',
                        help='输出目录')
    args = parser.parse_args()

    run_full_interpretability_analysis(
        data_path=args.data,
        use_simulated=args.simulated or args.data is None,
        output_dir=args.output,
    )
