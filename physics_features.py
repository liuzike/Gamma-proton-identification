"""
=============================================================================
Step 1: 物理引导的特征工程 (Physics-Guided Feature Engineering)
=============================================================================
基于伽马/质子簇射物理差异构建区分性特征。

核心物理原理：
  - 伽马簇射(电磁级联): 横向分布窄、轴对称、μ子贫乏
  - 质子簇射(强子级联): 横向分布宽、不对称、μ子丰富

特征来源: LHAASO KM2A + WCDA 探测器重建参数
  NuM1~NuM4 : 不同径向环的N数(可能对应muon探测器计数)
  NfiltM    : 滤波后的探测器命中数
  NhitM     : 总命中数
  NuW1      : WCDA相关参数
  base      : 基线参数
  rec_Eage  : 重建能量/年龄
=============================================================================
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def compute_physics_features(df):
    """
    从原始特征计算物理引导的衍生特征。

    参数:
        df: DataFrame, 包含原始特征列
    返回:
        DataFrame, 包含原始特征 + 物理衍生特征
    """
    eps = 1e-8  # 防止除零

    feats = df.copy()

    # =========================================================================
    # 类别A: 横向分布宽度特征 (Lateral Distribution)
    # 原理: 质子簇射横向分布更宽，伽马簇射更窄
    # =========================================================================

    # 总N数(所有环带之和)
    feats['NuM_total'] = feats['NuM1'] + feats['NuM2'] + feats['NuM3'] + feats['NuM4']

    # 加权平均半径——衡量簇射横向扩展的"质心"位置
    # 质子值更大(更分散)，伽马值更小(更集中)
    feats['lateral_spread'] = (
        1 * feats['NuM1'] + 2 * feats['NuM2'] +
        3 * feats['NuM3'] + 4 * feats['NuM4']
    ) / (feats['NuM_total'] + eps)

    # 半径方差——衡量信号能量的分散程度
    r_mean = feats['lateral_spread']
    feats['lateral_variance'] = (
        1 * feats['NuM1'] * (1 - r_mean)**2 +
        2 * feats['NuM2'] * (2 - r_mean)**2 +
        3 * feats['NuM3'] * (3 - r_mean)**2 +
        4 * feats['NuM4'] * (4 - r_mean)**2
    ) / (feats['NuM_total'] + eps)

    # =========================================================================
    # 类别B: 浓度/陡峭度特征 (Concentration / Steepness)
    # 原理: 伽马簇射能量集中在核心，质子簇射能量分散
    # =========================================================================

    # 核心集中度: 最内环占总信号的比重
    # 伽马高(~0.7-0.9), 质子低(~0.3-0.6)
    feats['core_concentration'] = feats['NuM1'] / (feats['NuM_total'] + eps)

    # 外围比例: 最外环占总信号的比重
    feats['outer_fraction'] = feats['NuM4'] / (feats['NuM_total'] + eps)

    # 核心-外围比: 综合衡量簇射形状
    feats['core_to_outer_ratio'] = (feats['NuM1'] + eps) / (feats['NuM4'] + eps)

    # 径向对数梯度: log(N(r1)/N(r3))——衡量横向剖面陡峭程度
    feats['radial_gradient_1_3'] = np.log(feats['NuM1'] + eps) - np.log(feats['NuM3'] + eps)

    # 逐环比(环间递减速度)
    feats['ring_ratio_21'] = feats['NuM2'] / (feats['NuM1'] + eps)
    feats['ring_ratio_32'] = feats['NuM3'] / (feats['NuM2'] + eps)
    feats['ring_ratio_43'] = feats['NuM4'] / (feats['NuM3'] + eps)

    # =========================================================================
    # 类别C: 不对称性特征 (Asymmetry)
    # 原理: 质子簇射由于强子相互作用的不规则性，表现出更大的不对称性
    #       伽马簇射更轴对称
    # =========================================================================

    # 相邻环不对称性指标
    # 对称的簇射应有 NuM1≈NuM3, NuM2≈NuM4 等
    feats['asymmetry_1_3'] = np.abs(feats['NuM1'] - feats['NuM3']) / (feats['NuM1'] + feats['NuM3'] + eps)
    feats['asymmetry_2_4'] = np.abs(feats['NuM2'] - feats['NuM4']) / (feats['NuM2'] + feats['NuM4'] + eps)

    # 偶奇不对称: 奇数环(NuM1+NuM3) vs 偶数环(NuM2+NuM4)
    odd_sum = feats['NuM1'] + feats['NuM3']
    even_sum = feats['NuM2'] + feats['NuM4']
    feats['odd_even_asymmetry'] = np.abs(odd_sum - even_sum) / (odd_sum + even_sum + eps)

    # 总不对称性(结合多个不对称指标)
    feats['total_asymmetry'] = np.sqrt(
        feats['asymmetry_1_3']**2 + feats['asymmetry_2_4']**2
    )

    # NfiltM与NhitM的不对称: 滤波后命中分布的偏斜程度
    feats['filter_hit_ratio'] = feats['NfiltM'] / (feats['NhitM'] + eps)

    # =========================================================================
    # 类别D: μ子含量代理特征 (Muon Content Proxy)
    # 原理: 伽马簇射μ子极少(电磁级联), 质子簇射μ子多(π→μ衰变链)
    #       NuM系列来自μ子探测器，可作为μ子含量代理
    # =========================================================================

    # μ子-命中比: 这是最强的单变量区分特征之一
    feats['muon_hit_ratio'] = feats['NuM_total'] / (feats['NhitM'] + eps)

    # 外围μ子比例: 强子簇射外围μ子更多
    feats['outer_muon_ratio'] = (feats['NuM3'] + feats['NuM4']) / (feats['NuM1'] + feats['NuM2'] + eps)

    # 归一化μ子含量(用base归一化)
    feats['normalized_muon_content'] = feats['NuM_total'] / (feats['base'] + eps)

    # =========================================================================
    # 类别E: 复合判别特征 (Composite Discriminants)
    # 原理: 多个物理量的非线性组合，模仿传统Cut-Based方法的判别变量
    # =========================================================================

    # 类Hillas宽度-长度乘积(用环计数类比)
    # width ∝ lateral_spread, length ∝ sqrt(lateral_variance)
    feats['hillas_like_size'] = feats['lateral_spread'] * np.sqrt(feats['lateral_variance'] + eps)

    # 簇射"紧密度": 核心集中度 × (1 - 总不对称性)
    feats['compactness'] = feats['core_concentration'] * (1 - feats['total_asymmetry'])

    # 伽马似然度: 结合三个物理约束
    # 1. 高核心集中度  2. 低μ子含量  3. 低不对称性
    feats['gamma_likeness'] = (
        feats['core_concentration'] *
        (1 / (feats['muon_hit_ratio'] + eps)) *
        (1 - feats['total_asymmetry'])
    )

    # 传统Cut-Based中的判别参数 (类似A.D. Supanitsky et al.方法)
    feats['discriminant_C'] = (
        feats['core_concentration'] *
        np.log(1 / (feats['muon_hit_ratio'] + eps))
    )

    # =========================================================================
    # 类别F: 能量/年龄相关交叉特征
    # =========================================================================

    # 能量相关的横向扩展: 相同能量下质子更分散
    feats['spread_per_energy'] = feats['lateral_spread'] / (feats['rec_Eage'] + eps)

    # μ子含量随能量的变化趋势
    feats['muon_energy_ratio'] = feats['NuM_total'] / (feats['rec_Eage'] + eps)

    return feats


# =============================================================================
# 完整特征列表
# =============================================================================
ORIGINAL_FEATURES = ['NuM4', 'NfiltM', 'base', 'NuM2', 'NuW1', 'NuM3', 'NhitM', 'NuM1']

PHYSICS_FEATURES = [
    # A: 横向分布宽度
    'NuM_total', 'lateral_spread', 'lateral_variance',
    # B: 浓度/陡峭度
    'core_concentration', 'outer_fraction', 'core_to_outer_ratio',
    'radial_gradient_1_3', 'ring_ratio_21', 'ring_ratio_32', 'ring_ratio_43',
    # C: 不对称性
    'asymmetry_1_3', 'asymmetry_2_4', 'odd_even_asymmetry',
    'total_asymmetry', 'filter_hit_ratio',
    # D: μ子含量代理
    'muon_hit_ratio', 'outer_muon_ratio', 'normalized_muon_content',
    # E: 复合判别特征
    'hillas_like_size', 'compactness', 'gamma_likeness', 'discriminant_C',
    # F: 能量/年龄交叉特征
    'spread_per_energy', 'muon_energy_ratio',
]


def build_feature_matrix(df):
    """
    构建特征矩阵: 原始特征 + 物理衍生特征

    参数:
        df: 原始DataFrame
    返回:
        X: 特征矩阵
        all_features: 所有特征名列表
    """
    df_enhanced = compute_physics_features(df)
    all_features = ORIGINAL_FEATURES + PHYSICS_FEATURES
    for f in all_features:
        if f not in df_enhanced.columns:
            raise ValueError(f"特征 '{f}' 不存在于增强后的DataFrame中")
    return df_enhanced[all_features], all_features


if __name__ == "__main__":
    # 生成模拟数据进行演示
    np.random.seed(42)
    n_samples = 1000

    # 模拟伽马样本
    gamma = pd.DataFrame({
        'NuM1': np.random.poisson(80, n_samples),
        'NuM2': np.random.poisson(30, n_samples),
        'NuM3': np.random.poisson(10, n_samples),
        'NuM4': np.random.poisson(3, n_samples),
        'NfiltM': np.random.poisson(200, n_samples),
        'NhitM': np.random.poisson(250, n_samples),
        'NuW1': np.random.poisson(50, n_samples),
        'base': np.abs(np.random.normal(100, 20, n_samples)),
        'rec_Eage': np.abs(np.random.normal(10, 3, n_samples)),
        'lable': 1
    })

    # 模拟质子样本 (更分散, μ子更多)
    proton = pd.DataFrame({
        'NuM1': np.random.poisson(40, n_samples),
        'NuM2': np.random.poisson(35, n_samples),
        'NuM3': np.random.poisson(30, n_samples),
        'NuM4': np.random.poisson(25, n_samples),
        'NfiltM': np.random.poisson(300, n_samples),
        'NhitM': np.random.poisson(350, n_samples),
        'NuW1': np.random.poisson(40, n_samples),
        'base': np.abs(np.random.normal(100, 25, n_samples)),
        'rec_Eage': np.abs(np.random.normal(8, 3, n_samples)),
        'lable': 0
    })

    df_demo = pd.concat([gamma, proton], ignore_index=True)
    df_feats = compute_physics_features(df_demo)

    print("=" * 70)
    print("物理特征工程演示")
    print("=" * 70)
    print(f"\n原始特征数: {len(ORIGINAL_FEATURES)}")
    print(f"新增物理特征数: {len(PHYSICS_FEATURES)}")
    print(f"总特征数: {len(ORIGINAL_FEATURES) + len(PHYSICS_FEATURES)}")
    print(f"样本数: {len(df_feats)}")
    print(f"\n{'='*70}")
    print("新增物理特征列表:")
    print("=" * 70)
    for i, feat in enumerate(PHYSICS_FEATURES, 1):
        print(f"  {i:2d}. {feat}")

    print(f"\n{'='*70}")
    print("伽马 vs 质子均值对比 (部分关键特征):")
    print("=" * 70)
    print(f"{'特征':<30s} {'伽马(均值)':<15s} {'质子(均值)':<15s} {'物理含义'}")
    print("-" * 70)
    key_feats = {
        'core_concentration': '核心集中度(伽马>质子)',
        'muon_hit_ratio': 'μ子占比(伽马<<质子)',
        'total_asymmetry': '总不对称性(伽马<质子)',
        'lateral_spread': '横向扩展(伽马<质子)',
        'compactness': '紧密度(伽马>质子)',
    }
    for feat, desc in key_feats.items():
        g_val = df_feats[df_feats['lable']==1][feat].mean()
        p_val = df_feats[df_feats['lable']==0][feat].mean()
        print(f"  {feat:<30s} {g_val:<15.4f} {p_val:<15.4f} {desc}")
