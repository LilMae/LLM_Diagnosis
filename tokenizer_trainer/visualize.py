import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from data.dataset import _channel_labels_for_mode


# 1) 임베딩/라벨/예측/데이터셋 수집 (배치 돌며 쌓아서 concat)
# embeds: (N, D), y_true: (N,), y_pred: (N,), ds_name: (N,)  ex) ["A","B","C","D","Val"]
# 예: y_pred = logits.argmax(1).cpu().numpy()


def pca_2d(embeds):
    return PCA(n_components=2).fit_transform(embeds)

def plot_all(Z, y_true, y_pred, ds_name, class_names=None, title="Embeddings (all datasets)"):
    # --- 팔레트/마커 정의 (클래스=색, 데이터셋=마커)
    classes = np.array(sorted(np.unique(y_true)))               # 5 classes
    datasets = np.array(sorted(np.unique(ds_name)))             # 5 datasets
    K = len(classes)
    cmap = plt.get_cmap("tab10")                                # 10색 팔레트
    color_of = {c: cmap(i % 10) for i, c in enumerate(classes)}
    marker_list = ['o', 's', '^', 'D', 'P']                     # 5개
    marker_of = {d: marker_list[i % len(marker_list)] for i, d in enumerate(datasets)}

    # 정/오분류 테두리
    correct = (y_true == y_pred)
    edgecolor = np.where(correct, "#1a7f37", "#d73a49")         # green / red

    # --- 산점도 (하나의 축에 전부)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for d in datasets:
        m = marker_of[d]
        idx = (ds_name == d)
        # 각 데이터셋 묶음에서 실제 클래스별로 색 부여
        colors = [color_of[c] for c in y_true[idx]]
        ax.scatter(Z[idx, 0], Z[idx, 1],
                   c=colors, marker=m, s=28, alpha=0.9,
                   edgecolors=edgecolor[idx], linewidths=0.7, label=d)

    ax.set_title(title); ax.set_xlabel("dim1"); ax.set_ylabel("dim2")

    # --- 범례 1: 실제 클래스(색)
    if class_names is None:
        class_names = {c: f"class {c}" for c in classes}
    handles_true = [Line2D([0],[0], marker='o', color='w',
                           markerfacecolor=color_of[c], markersize=8, label=class_names[c])
                    for c in classes]
    leg1 = ax.legend(handles_true, [h.get_label() for h in handles_true],
                     title="True class (color)", loc="upper right", frameon=True)
    ax.add_artist(leg1)

    # --- 범례 2: 데이터셋(마커)
    handles_ds = [Line2D([0],[0], marker=marker_of[d], color='k',
                         linestyle='None', markersize=7, markerfacecolor='none', label=d)
                  for d in datasets]
    leg2 = ax.legend(handles_ds, [h.get_label() for h in handles_ds],
                     title="Dataset (marker)", loc="lower right", frameon=True)

    # --- 범례 3: 정/오분류(테두리)
    handles_acc = [Line2D([0],[0], marker='o', color='#1a7f37', linestyle='-',
                          markerfacecolor='w', markersize=7, label='correct'),
                   Line2D([0],[0], marker='o', color='#d73a49', linestyle='-',
                          markerfacecolor='w', markersize=7, label='wrong')]
    ax.legend(handles_acc, [h.get_label() for h in handles_acc],
              title="Edge = prediction", loc="lower left", frameon=True)

    plt.tight_layout()
    plt.show()

# === 사용 예 ===
# Z = umap_2d(embeds, n_neighbors=15, min_dist=0.05, metric="cosine", seed=0)
# (비교용) Z_pca = pca_2d(embeds)
# class_names = {0:"C0",1:"C1",2:"C2",3:"C3",4:"C4"}  # 필요시
# plot_all(Z, y_true, y_pred, ds_name, class_names, title="ViT CLS embeddings (UMAP)")

def create_reconstruction_figure(
    orig_tensor,              # (C,H,W) 원본 이미지 텐서
    rec_tensor,               # (C,H,W) 재구성 이미지 텐서
    mode: str,
    max_order: float,
    window_sec: float,
    figsize=(12, 18)
):
    """
    원본과 재구성 이미지를 채널별로 나란히 비교하는 matplotlib Figure를 생성합니다.
    이 Figure 객체는 wandb.Image()로 변환되어 로깅될 수 있습니다.
    """
    orig_arr = orig_tensor.detach().to(dtype=torch.float32).cpu().numpy()
    rec_arr  = rec_tensor.detach().to(dtype=torch.float32).cpu().numpy()
    
    assert orig_arr.ndim == 3, "Input must be (C,H,W)"
    C, H, W = orig_arr.shape

    ch_labels = _channel_labels_for_mode(mode)
    if len(ch_labels) < C:
        ch_labels += [f"ch{i}" for i in range(len(ch_labels), C)]

    # 각 채널을 한 행으로, (원본, 재구성)을 두 열으로 배치
    fig, axes = plt.subplots(C, 2, figsize=figsize, squeeze=False)
    extent = [0.0, float(window_sec), 0.0, float(max_order)]

    for i in range(C):
        # --- 원본 이미지 ---
        ax_orig = axes[i, 0]
        img_orig = orig_arr[i]
        im_orig = ax_orig.imshow(img_orig, origin="lower", aspect="auto", extent=extent, cmap="magma")
        ax_orig.set_title(f"Original: {ch_labels[i]}", fontsize=10)
        ax_orig.set_xlabel("Time (s)")
        ax_orig.set_ylabel("Order")
        fig.colorbar(im_orig, ax=ax_orig, fraction=0.046, pad=0.04)

        # --- 재구성 이미지 ---
        ax_rec = axes[i, 1]
        img_rec = rec_arr[i]
        # 원본과 동일한 스케일로 보기 위해 vmin/vmax 공유
        im_rec = ax_rec.imshow(img_rec, origin="lower", aspect="auto", extent=extent, cmap="magma", 
                               vmin=im_orig.get_clim()[0], vmax=im_orig.get_clim()[1])
        ax_rec.set_title(f"Reconstructed: {ch_labels[i]}", fontsize=10)
        ax_rec.set_xlabel("Time (s)")
        ax_rec.set_ylabel("Order")
        fig.colorbar(im_rec, ax=ax_rec, fraction=0.046, pad=0.04)

    fig.suptitle(f"Reconstruction Comparison (Mode: {mode})", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # suptitle과 겹치지 않게 조정
    
    # 생성된 Figure 객체를 반환 (plt.show()나 savefig()는 호출하지 않음)
    return fig