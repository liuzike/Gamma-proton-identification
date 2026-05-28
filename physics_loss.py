"""
=============================================================================
Step 2: 物理信息损失函数 (Physics-Informed Loss Function)
=============================================================================
将天体粒子物理的先验知识编码到损失函数中，约束模型做出
物理上一致的预测。

核心思想:
  L_total = L_classification + λ_physics * L_physics_constraint

物理约束 L_physics:
  - μ子约束: 预测为伽马但μ子含量异常高 → 惩罚
  - 对称性约束: 预测为伽马但不对称性异常高 → 惩罚
  - 集中度约束: 预测为质子但核心过于集中 → 惩罚

两种实现方式:
  A. PyTorch自定义损失 (适合DNN)
  B. 物理约束正则化器 (适合任何模型，作为数据增强/加权)
=============================================================================
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin


# =============================================================================
# A. PyTorch 物理信息损失
# =============================================================================

class PhysicsInformedLoss(nn.Module):
    """
    物理信息损失函数: 交叉熵 + 物理约束惩罚项

    L_total = CE(y_pred, y_true) + λ_muon * L_muon + λ_asym * L_asym + λ_conc * L_conc

    其中:
      L_muon  = prob_gamma * muon_hit_ratio  (预测伽马但μ子多→惩罚)
      L_asym  = prob_gamma * total_asymmetry  (预测伽马但不不对称→惩罚)
      L_conc  = prob_proton * (1 - core_concentration) (预测质子但核心集中→惩罚)

    参数:
        lambda_muon:   μ子约束权重，默认0.1
        lambda_asym:   不对称性约束权重，默认0.1
        lambda_conc:   集中度约束权重，默认0.05
        muon_threshold: μ子含量阈值，超过此值被认为是μ子丰富
    """

    def __init__(self, lambda_muon=0.1, lambda_asym=0.1, lambda_conc=0.05,
                 muon_threshold=0.3, asym_threshold=0.3, conc_threshold=0.5):
        super(PhysicsInformedLoss, self).__init__()
        self.lambda_muon = lambda_muon
        self.lambda_asym = lambda_asym
        self.lambda_conc = lambda_conc
        self.muon_threshold = muon_threshold
        self.asym_threshold = asym_threshold
        self.conc_threshold = conc_threshold

    def forward(self, logits, targets, physics_features):
        """
        参数:
            logits:           模型输出 [batch_size, 2]
            targets:          真实标签 [batch_size]
            physics_features: 物理特征字典, 包含:
                              - 'muon_hit_ratio'    [batch_size]
                              - 'total_asymmetry'   [batch_size]
                              - 'core_concentration'[batch_size]
        返回:
            total_loss: 标量损失
        """
        # 标准分类损失 (Cross-Entropy)
        ce_loss = F.cross_entropy(logits, targets.long())

        # 预测为伽马的概率
        probs = F.softmax(logits, dim=1)
        prob_gamma = probs[:, 1]   # 预测为gamma的概率
        prob_proton = probs[:, 0]  # 预测为proton的概率

        # ---- 物理约束1: μ子约束 ----
        # 如果样本 μ子含量高，但被预测为伽马 → 惩罚
        muon_content = physics_features['muon_hit_ratio']
        muon_penalty = prob_gamma * torch.clamp(
            muon_content - self.muon_threshold, min=0
        )
        L_muon = muon_penalty.mean()

        # ---- 物理约束2: 不对称性约束 ----
        asymmetry = physics_features['total_asymmetry']
        asym_penalty = prob_gamma * torch.clamp(
            asymmetry - self.asym_threshold, min=0
        )
        L_asym = asym_penalty.mean()

        # ---- 物理约束3: 核心集中度约束 ----
        # 如果核心集中度高(像伽马)，但被预测为质子 → 惩罚
        core_conc = physics_features['core_concentration']
        conc_penalty = prob_proton * torch.clamp(
            core_conc - self.conc_threshold, min=0
        )
        L_conc = conc_penalty.mean()

        # 总损失
        total_loss = ce_loss + \
                     self.lambda_muon * L_muon + \
                     self.lambda_asym * L_asym + \
                     self.lambda_conc * L_conc

        return total_loss


class AdaptivePhysicsLoss(nn.Module):
    """
    自适应物理损失: λ权重随训练进度自动调整

    早期: 大λ → 物理约束主导，引导模型朝物理合理方向发展
    后期: 小λ → 数据驱动主导，让模型从数据中精调
    """

    def __init__(self, lambda_muon=0.5, lambda_asym=0.5, lambda_conc=0.3,
                 decay_rate=0.95, decay_steps=5):
        super(AdaptivePhysicsLoss, self).__init__()
        self.base_lambda_muon = lambda_muon
        self.base_lambda_asym = lambda_asym
        self.base_lambda_conc = lambda_conc
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.current_step = 0

        # 内部使用固定阈值的PhysicsInformedLoss
        self.base_loss = PhysicsInformedLoss(
            lambda_muon=lambda_muon,
            lambda_asym=lambda_asym,
            lambda_conc=lambda_conc
        )

    def step(self):
        """每个epoch调用一次，衰减物理约束权重"""
        self.current_step += 1
        if self.current_step % self.decay_steps == 0:
            self.base_loss.lambda_muon *= self.decay_rate
            self.base_loss.lambda_asym *= self.decay_rate
            self.base_loss.lambda_conc *= self.decay_rate

    def forward(self, logits, targets, physics_features):
        return self.base_loss(logits, targets, physics_features)


# =============================================================================
# B. 物理约束正则化器 (可用于任何sklearn兼容模型)
# =============================================================================

class PhysicsConstrainedClassifier(BaseEstimator, ClassifierMixin):
    """
    物理约束包装器: 用物理规则调整任意分类器的训练

    方法:
      1. 用原始模型训练得到基线预测
      2. 识别物理上"可疑"的预测
      3. 对可疑样本进行样本权重调整后重训练

    这类似于 "物理引导的样本重加权" (Physics-Guided Sample Reweighting)
    """

    def __init__(self, base_model, lambda_physics=0.3,
                 muon_threshold=0.3, asym_threshold=0.3,
                 n_refine_iterations=2):
        """
        参数:
            base_model:          基础分类器 (任意sklearn兼容模型)
            lambda_physics:      物理约束强度
            muon_threshold:      μ子含量阈值
            asym_threshold:      不对称性阈值
            n_refine_iterations: 细化迭代次数
        """
        self.base_model = base_model
        self.lambda_physics = lambda_physics
        self.muon_threshold = muon_threshold
        self.asym_threshold = asym_threshold
        self.n_refine_iterations = n_refine_iterations

    def _compute_physics_penalty(self, X_df, y_pred):
        """
        计算每个样本的物理不一致惩罚权重

        返回:
            weights: 样本权重 (物理不一致的样本权重降低)
        """
        n_samples = len(X_df)
        weights = np.ones(n_samples)

        # 只在有物理特征列的情况下计算
        if 'muon_hit_ratio' in X_df.columns and 'total_asymmetry' in X_df.columns:
            for i in range(n_samples):
                penalty = 0.0

                # 预测为伽马(1) 但 μ子含量高 → 物理不一致
                if y_pred[i] == 1:
                    if X_df['muon_hit_ratio'].iloc[i] > self.muon_threshold:
                        penalty += self.lambda_physics * (
                            X_df['muon_hit_ratio'].iloc[i] - self.muon_threshold
                        )
                    if X_df['total_asymmetry'].iloc[i] > self.asym_threshold:
                        penalty += self.lambda_physics * (
                            X_df['total_asymmetry'].iloc[i] - self.asym_threshold
                        )

                # 预测为质子(0) 但核心集中度高 → 物理不一致
                if y_pred[i] == 0:
                    if X_df['core_concentration'].iloc[i] > 0.5:
                        penalty += self.lambda_physics * (
                            X_df['core_concentration'].iloc[i] - 0.5
                        )

                weights[i] = np.exp(-penalty)

        return weights

    def fit(self, X, y):
        # X可能是numpy array或DataFrame
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
        else:
            X_df = X.copy()

        # 第一次训练
        self.base_model.fit(X, y)
        y_pred = self.base_model.predict(X)

        # 迭代细化
        for it in range(self.n_refine_iterations):
            sample_weights = self._compute_physics_penalty(X_df, y_pred)
            self.base_model.fit(X, y, sample_weight=sample_weights)
            y_pred = self.base_model.predict(X)

        return self

    def predict(self, X):
        return self.base_model.predict(X)

    def predict_proba(self, X):
        return self.base_model.predict_proba(X)


# =============================================================================
# C. 物理辅助训练样例生成
# =============================================================================

def generate_physics_augmented_samples(X_df, y, n_synthetic=200):
    """
    基于物理对称性生成增强样本

    原理: 簇射的方位角对称性意味着旋转探测器图像不应改变分类结果。
    这里通过交换等效的对称特征来近似这种不变性。

    对环形采样特征(NuM1~NuM4)，扰动其径向分布来模拟
    探测器噪声和簇射涨落，同时保持物理合理性。
    """
    np.random.seed(42)
    n_samples = len(X_df)
    indices = np.random.choice(n_samples, size=min(n_synthetic, n_samples), replace=False)

    X_aug = X_df.iloc[indices].copy()
    y_aug = y[indices].copy() if isinstance(y, np.ndarray) else np.array(y)[indices].copy()

    # 对NuM类特征做小幅度扰动 (模拟探测器涨落)
    for col in ['NuM1', 'NuM2', 'NuM3', 'NuM4']:
        if col in X_aug.columns:
            noise_factor = np.random.uniform(0.9, 1.1, size=len(X_aug))
            X_aug[col] = X_aug[col] * noise_factor

    # 重新计算衍生物理特征
    # (如果X_df已包含这些列, 这里会更新它们)
    from physics_features import compute_physics_features
    X_aug = compute_physics_features(X_aug)

    # 确保与原始数据列一致
    common_cols = [c for c in X_df.columns if c in X_aug.columns]
    X_aug = X_aug[common_cols]

    return X_aug, y_aug


# =============================================================================
# D. 演示代码
# =============================================================================

if __name__ == "__main__":
    import pandas as pd
    from physics_features import compute_physics_features, ORIGINAL_FEATURES

    # 生成模拟数据
    np.random.seed(42)
    n = 500
    gamma_df = pd.DataFrame({
        'NuM1': np.random.poisson(80, n), 'NuM2': np.random.poisson(30, n),
        'NuM3': np.random.poisson(10, n), 'NuM4': np.random.poisson(3, n),
        'NfiltM': np.random.poisson(200, n), 'NhitM': np.random.poisson(250, n),
        'NuW1': np.random.poisson(50, n), 'base': np.abs(np.random.normal(100, 20, n)),
        'rec_Eage': np.abs(np.random.normal(10, 3, n)), 'lable': 1
    })
    proton_df = pd.DataFrame({
        'NuM1': np.random.poisson(40, n), 'NuM2': np.random.poisson(35, n),
        'NuM3': np.random.poisson(30, n), 'NuM4': np.random.poisson(25, n),
        'NfiltM': np.random.poisson(300, n), 'NhitM': np.random.poisson(350, n),
        'NuW1': np.random.poisson(40, n), 'base': np.abs(np.random.normal(100, 25, n)),
        'rec_Eage': np.abs(np.random.normal(8, 3, n)), 'lable': 0
    })
    df = pd.concat([gamma_df, proton_df], ignore_index=True)

    # 计算物理特征
    df_feats = compute_physics_features(df)

    # === 演示 PyTorch 物理损失 ===
    print("=" * 70)
    print("物理信息损失函数演示")
    print("=" * 70)

    # 模拟一批数据
    batch_size = 32
    idx = np.random.choice(len(df_feats), batch_size)
    batch_df = df_feats.iloc[idx]

    # 模拟模型输出和标签
    logits = torch.randn(batch_size, 2)
    targets = torch.tensor(batch_df['lable'].values, dtype=torch.float32)

    physics_feats = {
        'muon_hit_ratio': torch.tensor(batch_df['muon_hit_ratio'].values, dtype=torch.float32),
        'total_asymmetry': torch.tensor(batch_df['total_asymmetry'].values, dtype=torch.float32),
        'core_concentration': torch.tensor(batch_df['core_concentration'].values, dtype=torch.float32),
    }

    # 计算损失
    criterion = PhysicsInformedLoss(lambda_muon=0.1, lambda_asym=0.1)
    ce_loss = F.cross_entropy(logits, targets.long())
    total_loss = criterion(logits, targets, physics_feats)

    print(f"\n  交叉熵损失:      {ce_loss.item():.4f}")
    print(f"  物理约束损失:    {(total_loss - ce_loss).item():.4f}")
    print(f"  总损失:          {total_loss.item():.4f}")
    print(f"  物理约束贡献:    {((total_loss - ce_loss) / total_loss * 100).item():.1f}%")

    # === 演示自适应衰减 ===
    print(f"\n{'='*70}")
    print("自适应物理损失 - λ衰减曲线")
    print("=" * 70)
    adaptive = AdaptivePhysicsLoss(lambda_muon=0.5, lambda_asym=0.5,
                                    decay_rate=0.9, decay_steps=5)
    print(f"\n  {'Epoch':<8s} {'λ_muon':<12s} {'λ_asym':<12s}")
    print("  " + "-" * 32)
    for ep in range(26):
        lambda_m = adaptive.base_loss.lambda_muon
        lambda_a = adaptive.base_loss.lambda_asym
        if ep % 5 == 0:
            print(f"  {ep:<8d} {lambda_m:<12.4f} {lambda_a:<12.4f}")
        adaptive.step()
