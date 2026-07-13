from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "gradcam_3d_visual_walkthrough.ipynb"


def clean(source: str) -> str:
    return dedent(source).strip() + "\n"


def markdown(source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": clean(source),
    }


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": clean(source),
    }


cells = [
    markdown(
        r"""
        # Grad-CAM 3D 可视化逐步拆解

        配套 PPT：`artifacts/deck/gradcam-roadmap-v18.pptx`

        这个 notebook 固定对齐当前 PPT 的方法顺序：

        1. Experimental setup：3D toy volume、1-voxel 3-axis cross、3D box distractors、Conv3D classifier。
        2. Method 01：notGradCAM / activation map。
        3. Method 02：Grad-CAM。
        4. Method 03：Guided Grad-CAM。
        5. Method 04：LayerCAM。
        6. Method 05：Occlusion sensitivity。
        7. Method 06：Integrated Gradients。
        8. Method 07：Integrated Grad-CAM。

        前半部分只快速复现数据、模型和 checkpoint；后半部分按 PPT 逐步拆开每个可视化方法。视觉部分的代码行都带注释，方便直接搬到讲稿里解释。
        """
    ),
    markdown(
        r"""
        ## 运行方式

        在仓库根目录运行：

        ```bash
        cd /Users/baiduli/ProgramProject/gradcam-repro
        uv sync
        ```

        然后用 VS Code / Jupyter 打开本 notebook。默认直接读取已经训练好的 checkpoint 和已经生成的 PPT 图；如果要完整重跑实验，用：

        ```bash
        uv run gradcam-repro all
        ```
        """
    ),
    markdown(
        r"""
        ## Part 1 - 实验设置快速略过

        PPT 的实验目标不是证明一个大模型，而是制造一个可控的 3D 场景：

        - 输入体素：$x \in \mathbb{R}^{1 \times 24 \times 24 \times 24}$
        - 目标：1-voxel-wide 的 3D 十字，沿 $x,y,z$ 三个轴都有 7-voxel arms
        - 干扰：随机 3D box / cuboid distractors
        - 标签：十字在左半边为 class 0，在右半边为 class 1
        - CAM tap point：`stage2`，特征大小为 $A \in \mathbb{R}^{16 \times 6 \times 6 \times 6}$

        训练和数据生成这一段讲 PPT 时可以快速带过；后面的 visualization 才是主要内容。
        """
    ),
    code(
        r"""
        from pathlib import Path  # 用 Path 统一处理本地仓库、artifact 和 notebook 路径。
        import json  # 用 json 读取 figure manifest、训练指标和 attribution 分数。
        import os  # 用 os 设置 Matplotlib cache，避免受 home 目录权限影响。
        import sys  # 用 sys.path 暴露本地 src 包，避免必须先 pip install。
        import torch  # 用 torch 执行 3D CNN forward、gradient 和 attribution。
        os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "gradcam-repro-matplotlib"))  # 把 Matplotlib cache 放到可写临时目录。
        import matplotlib.pyplot as plt  # 用 matplotlib 展示 2D slice 和 heatmap overlay。

        start_path = Path.cwd().resolve()  # 记录当前 Jupyter 工作目录，可能是仓库根目录也可能是 notebooks/。
        repo = start_path  # 先假设当前目录就是仓库根目录。
        for candidate in [start_path, *start_path.parents]:  # 从当前目录向上搜索真正的 gradcam-repro 根目录。
            marker = candidate / "src" / "gradcam_repro"  # 用源码包目录作为仓库根目录标记。
            if marker.exists():  # 如果这个候选目录包含源码包，就认为找到了仓库根目录。
                repo = candidate  # 保存仓库根目录，后面所有相对路径都基于它。
                break  # 找到以后退出搜索，避免继续向上误匹配。

        src_path = repo / "src"  # 计算本地源码目录。
        if str(src_path) not in sys.path:  # 如果源码目录还没进 Python import path，就补进去。
            sys.path.insert(0, str(src_path))  # 把本地源码放在最前，确保导入当前仓库代码。

        from gradcam_repro.attribution import METHODS  # 导入 PPT 中已经固定的 attribution 方法表。
        from gradcam_repro.attribution import _cam_forward  # 导入 CAM forward helper，用来拿 activation 并 retain gradient。
        from gradcam_repro.attribution import gaussian_blur3d  # 导入 3D Gaussian smoothing，用于 IG 后处理。
        from gradcam_repro.attribution import guided_backprop_raw  # 导入 Guided Backprop 原始梯度，用于 Guided Grad-CAM。
        from gradcam_repro.attribution import normalize_map  # 导入逐样本 min-max normalize，方便显示 heatmap。
        from gradcam_repro.attribution import occlusion_sensitivity  # 导入 3D occlusion sensitivity 实现。
        from gradcam_repro.attribution import score_for_target  # 导入 target logit 选择函数，统一 class c 的得分。
        from gradcam_repro.attribution import upsample_to_input  # 导入 3D trilinear upsample，把 feature CAM 放回输入大小。
        from gradcam_repro.data import CrossHalfDataset  # 导入合成 3D 十字数据集。
        from gradcam_repro.data import ToyDataConfig  # 导入数据配置，展示 target 和 distractor 参数。
        from gradcam_repro.train import load_model  # 导入 checkpoint loader。
        from gradcam_repro.train import resolve_device  # 导入设备解析函数；本 notebook 默认用 CPU 保持一致。
        from gradcam_repro.visualize import feature_z  # 导入输入 z 到 feature z 的映射函数。
        from gradcam_repro.visualize import overlay_heatmap  # 导入 heatmap overlay helper。
        from gradcam_repro.visualize import sample_z  # 导入样本中心 z 切片 helper。
        from gradcam_repro.visualize import volume_slice  # 导入 3D volume 到 2D slice 的 helper。
        """
    ),
    code(
        r"""
        checkpoint_path = repo / "artifacts" / "checkpoints" / "toy_cnn.pt"  # 当前 PPT 使用的训练 checkpoint。
        metrics_path = repo / "artifacts" / "checkpoints" / "toy_cnn.metrics.json"  # 训练曲线和测试指标。
        scores_path = repo / "artifacts" / "scores" / "attribution_scores.json"  # 各 attribution 方法的量化指标。
        manifest_path = repo / "artifacts" / "figures" / "manifest.json"  # PPT figure provenance manifest。
        figures_dir = repo / "artifacts" / "figures"  # 当前 PPT 嵌入的实验图目录。

        if not checkpoint_path.exists():  # 如果 checkpoint 不存在，说明还没有跑过实验。
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")  # 提醒先运行 uv run gradcam-repro all。
        if not manifest_path.exists():  # 如果 manifest 不存在，说明 PPT 图还没有由实验代码生成。
            raise FileNotFoundError(f"Missing figure manifest: {manifest_path}")  # 提醒先生成 figures。

        with metrics_path.open() as handle:  # 打开训练指标 JSON。
            metrics = json.load(handle)  # 读取训练指标到 Python dict。
        with scores_path.open() as handle:  # 打开 attribution score JSON。
            scores = json.load(handle)  # 读取 attribution 指标到 Python dict。
        with manifest_path.open() as handle:  # 打开 figure provenance manifest。
            manifest = json.load(handle)  # 读取 manifest 到 Python dict。
        """
    ),
    code(
        r"""
        print("repo:", repo)  # 打印仓库根目录，确认 notebook 的路径解析正确。
        print("checkpoint:", checkpoint_path)  # 打印 checkpoint 路径，确认复现使用的模型。
        print("experiment:", manifest["experiment"])  # 打印当前 figure manifest 记录的实验名。
        print("test metrics:", metrics["test_metrics"])  # 打印测试集指标，PPT 里可简单略过。
        print("stopped reason:", metrics["stopped_reason"])  # 打印训练早停原因，说明模型确实学会任务。
        print("figure generated at:", manifest["generated_at"])  # 打印实验图生成时间，方便追踪缓存。
        """
    ),
    code(
        r"""
        device = resolve_device("cpu")  # 使用 CPU，和当前 PPT manifest 的 device=cpu 保持一致。
        model = load_model(checkpoint_path, device)  # 从 checkpoint 加载已经训练好的 3D CNN。
        model.eval()  # 设置 eval 模式，保证 inference 和 visualization 稳定。
        config = ToyDataConfig()  # 使用默认数据配置，匹配 PPT 中的 24^3 volume 和 3D cross。
        viz_ds = CrossHalfDataset(n_samples=12, seed=27, config=config)  # 用 manifest 里的 seed=27 生成可视化样本。
        sample_index = next(i for i in range(len(viz_ds)) if int(viz_ds[i]["label"]) == 1)  # 选一个 class 1 样本，展示右半边目标。
        sample = viz_ds[sample_index]  # 从数据集中取出这个样本。
        image = sample["image"].unsqueeze(0).to(device)  # 增加 batch 维度，得到形状 [1, 1, 24, 24, 24]。
        target = sample["label"].view(1).to(device)  # 把标签变成 batch 形状，后面统一作为 class c。
        center_z, center_y, center_x = [int(value.item()) for value in sample["center"]]  # 读取 3D 十字中心坐标。
        print("sample_index:", sample_index)  # 打印样本 index，方便复现同一张图。
        print("label:", int(sample["label"].item()))  # 打印 class label；1 表示目标在右半边。
        print("center (z, y, x):", (center_z, center_y, center_x))  # 打印 target center，后面切片都围绕它展示。
        """
    ),
    code(
        r"""
        with torch.no_grad():  # 只做 forward 形状检查，不需要梯度。
            logits_shape_check, features_shape_check = model(image, return_features=True)  # 取 logits 和中间 feature volumes。

        print("input:", tuple(image.shape))  # 输入形状应为 [1, 1, 24, 24, 24]。
        print("stage1:", tuple(features_shape_check["stage1"].shape))  # stage1 形状应为 [1, 8, 12, 12, 12]。
        print("stage2:", tuple(features_shape_check["stage2"].shape))  # stage2 形状应为 [1, 16, 6, 6, 6]，是 PPT 的 CAM tap point。
        print("stage3:", tuple(features_shape_check["stage3"].shape))  # stage3 形状应为 [1, 32, 6, 6, 6]。
        print("logits:", logits_shape_check.detach().cpu().numpy())  # 打印两个 class logit，确认模型更偏向 true class。
        """
    ),
    markdown(
        r"""
        ### 当前 PPT 总览图

        这张图来自 `uv run gradcam-repro demo`，也是 PPT 里 DEMO I 使用的实验图。

        ![method grid](../artifacts/figures/method_grid.png)
        """
    ),
    markdown(
        r"""
        ## Part 2 - 视觉拆解 helper

        从这里开始是讲 PPT 时真正需要逐行解释的部分。下面 helper 只负责把 3D tensor 切成可读的 2D slice 或 overlay；每个方法的数学逻辑会在各自小节里单独拆开。
        """
    ),
    code(
        r"""
        def as_volume(tensor):  # 把输入 tensor 规整成 [D, H, W]，方便做 axial/coronal/sagittal slice。
            data = tensor.detach().float().cpu().squeeze()  # 去掉 batch/channel 维度，并移动到 CPU。
            if data.ndim != 3:  # 如果 squeeze 以后不是 3D volume，说明传入对象不适合切片。
                raise ValueError(f"Expected a 3D volume after squeeze, got shape {tuple(data.shape)}")  # 报错时显示实际形状。
            return data  # 返回 [D, H, W] 的 3D volume。

        def show_panels(panel_items, title, columns=4):  # 统一展示若干 2D panel，保持后续方法页面视觉一致。
            rows = (len(panel_items) + columns - 1) // columns  # 根据 panel 数量计算需要几行。
            fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.0 * rows), squeeze=False)  # 创建固定大小的 subplot grid。
            flat_axes = axes.ravel()  # 把二维 axes 展平成一维，方便逐个填图。
            for axis, (panel_title, panel_tensor, cmap_name, low, high) in zip(flat_axes, panel_items, strict=False):  # 遍历每个 panel。
                data = panel_tensor.detach().cpu() if torch.is_tensor(panel_tensor) else panel_tensor  # tensor 转 CPU，RGB numpy/array 直接保留。
                if cmap_name is None:  # cmap=None 表示这是 RGB overlay，不需要 colormap。
                    axis.imshow(data)  # 直接展示 RGB 图。
                else:  # 有 cmap 时表示这是单通道 heatmap 或 grayscale slice。
                    axis.imshow(data, cmap=cmap_name, vmin=low, vmax=high)  # 用指定 cmap 和取值范围展示。
                axis.set_title(panel_title, fontsize=10)  # 给每个 panel 加短标题，和 PPT 页面对应。
                axis.set_xticks([])  # 去掉 x 轴刻度，减少视觉噪声。
                axis.set_yticks([])  # 去掉 y 轴刻度，减少视觉噪声。
            for axis in flat_axes[len(panel_items):]:  # 遍历没有用到的空 subplot。
                axis.axis("off")  # 关闭空 subplot，保持画面干净。
            fig.suptitle(title, fontsize=13, fontweight="bold")  # 给整组 panel 加标题。
            fig.tight_layout(pad=0.35)  # 压紧布局，让图像尽量大。
            plt.show()  # 在 notebook 中显示这组图。

        def show_three_planes(volume_tensor, title, cmap_name="gray", low=0, high=1):  # 展示同一 3D volume 的三个正交切面。
            volume = as_volume(volume_tensor)  # 把输入规整为 [D, H, W]。
            axial = volume[center_z]  # axial slice 固定 z=center_z，显示 x-y 平面。
            coronal = volume[:, center_y, :]  # coronal slice 固定 y=center_y，显示 z-x 平面。
            sagittal = volume[:, :, center_x]  # sagittal slice 固定 x=center_x，显示 z-y 平面。
            panels = [  # 组织三个正交视图。
                (f"axial z={center_z}", axial, cmap_name, low, high),  # axial 图应该看到 x/y 两条 arm。
                (f"coronal y={center_y}", coronal, cmap_name, low, high),  # coronal 图应该看到 z/x 两条 arm。
                (f"sagittal x={center_x}", sagittal, cmap_name, low, high),  # sagittal 图应该看到 z/y 两条 arm。
            ]  # 完成 panel 列表。
            show_panels(panels, title, columns=3)  # 用统一 helper 展示三个正交切面。

        def overlay_panel(heatmap_tensor, title):  # 把 3D heatmap overlay 到输入 volume 的 target z slice 上。
            z_index = sample_z(sample)  # 使用样本记录的 target center z 作为展示切片。
            overlay = overlay_heatmap(sample["image"], heatmap_tensor.detach().cpu(), z_index)  # 生成灰度输入加彩色 heatmap 的 overlay。
            return (title, overlay, None, None, None)  # 返回 show_panels 可消费的 RGB panel。
        """
    ),
    markdown(
        r"""
        ## Part 3 - 先确认 3D target 和干扰项

        这里要在讲 PPT 时明确说：这不是 2D 十字，也不是只在一张 axial slice 上画了 target。它是一个 3D 十字，沿三个轴都有 1-voxel-wide arm；同时 volume 里有 3D box distractors。
        """
    ),
    code(
        r"""
        show_three_planes(sample["image"], "Input volume: 3-axis 1-voxel cross + 3D box distractors")  # 展示输入 volume 的三个正交切面。
        show_three_planes(sample["mask"], "Ground-truth 7x7x7 target window around the cross", cmap_name="Greens", low=0, high=1)  # 展示 GT mask 的三个正交切面。
        """
    ),
    markdown(
        r"""
        ## Method 01 - notGradCAM / Activation Map

        PPT 对应页：`METHOD 01`

        公式：

        $$
        M(x)=\operatorname{Norm}\left(\operatorname{Up}\left(\frac{1}{K}\sum_{k=1}^{K} A^k(x)\right)\right)
        $$

        符号：

        - $A^k(x)$：目标层 `stage2` 的第 $k$ 个 3D activation volume
        - $K$：channel 数，这里是 16
        - $\sum_k A^k$：把所有 channel 的 activation 加起来
        - $\frac{1}{K}\sum_k A^k$：activation 的 channel mean
        - $\operatorname{Up}$：把 $6^3$ feature volume 放大回 $24^3$
        - $\operatorname{Norm}$：min-max normalize 到 $[0,1]$

        它解决的最早问题：只看模型内部激活，不使用 class gradient；因此简单，但不一定 class-discriminative。

        论文/背景：activation visualization 是 CAM/Grad-CAM 之前最基础的模型内部特征查看方式；PPT 中把它作为梯度类方法的起点。

        ![notgradcam decomposition](../artifacts/figures/notgradcam_decomposition.png)
        """
    ),
    code(
        r"""
        model.eval()  # 确保模型处于 inference 模式。
        with torch.no_grad():  # notGradCAM 不使用梯度，所以关闭 autograd。
            logits_ng, features_ng = model(image, return_features=True)  # 前向传播，同时取出所有 stage feature。
            activation_ng = features_ng["stage2"]  # 取 PPT 指定的 stage2 activation A。

        channel_scores_ng = activation_ng.flatten(2).mean(dim=2).squeeze(0)  # 对每个 channel 的 3D activation 求平均强度。
        top_channels_ng = torch.topk(channel_scores_ng, k=4).indices.tolist()  # 选出最活跃的 4 个 A^k 方便展示。
        sum_activation_ng = activation_ng.sum(dim=1, keepdim=True)  # 计算 Σ_k A^k。
        mean_activation_ng = activation_ng.mean(dim=1, keepdim=True)  # 计算 (1/K)Σ_k A^k。
        notgradcam_heatmap = normalize_map(upsample_to_input(mean_activation_ng, image))  # 上采样到输入大小并 normalize。
        feature_z_ng = feature_z(sample, activation_ng)  # 把输入中心 z 映射到 stage2 的 feature z。
        """
    ),
    code(
        r"""
        channel_panels_ng = []  # 准备存放 A^k 的可视化 panel。
        for channel_id in top_channels_ng:  # 遍历前面选出的 top activation channels。
            channel_slice_ng = volume_slice(activation_ng[0, channel_id], feature_z_ng)  # 取该 channel 在 feature_z 上的 2D slice。
            channel_panels_ng.append((f"A^{channel_id}[z={feature_z_ng}]", channel_slice_ng, "viridis", None, None))  # 添加 A^k panel。

        panels_ng = [(f"input x[z={center_z}]", volume_slice(sample["image"], center_z), "gray", 0, 1)]  # 第一张图显示输入 axial slice。
        panels_ng = panels_ng + channel_panels_ng  # 接着展示多个 A^k。
        panels_ng.append(("Σ_k A^k", volume_slice(sum_activation_ng, feature_z_ng), "viridis", None, None))  # 展示 channel summation。
        panels_ng.append(("(1/K)Σ_k A^k", volume_slice(mean_activation_ng, feature_z_ng), "viridis", None, None))  # 展示 channel mean。
        panels_ng.append(("activation output", volume_slice(notgradcam_heatmap, center_z), "turbo", 0, 1))  # 展示上采样后的 activation heatmap。
        panels_ng.append(overlay_panel(notgradcam_heatmap, "overlay output"))  # 展示 overlay 结果。
        show_panels(panels_ng, "Method 01: A^k -> Σ A^k -> mean A^k -> activation output", columns=4)  # 显示完整拆解。
        """
    ),
    markdown(
        r"""
        ## Method 02 - Grad-CAM

        PPT 对应页：`METHOD 02`

        公式：

        $$
        \alpha_k^c=\frac{1}{Z}\sum_i\sum_j\sum_l\frac{\partial y^c}{\partial A^k_{ijl}},
        \qquad
        L_{\mathrm{GradCAM}}^c=\operatorname{ReLU}\left(\sum_k\alpha_k^c A^k\right)
        $$

        符号：

        - $y^c$：class $c$ 的 logit
        - $A^k_{ijl}$：第 $k$ 个 feature volume 在 3D 位置 $(i,j,l)$ 的 activation
        - $Z$：feature volume 的空间位置数量，这里是 $6 \times 6 \times 6$
        - $\alpha_k^c$：用 gradient spatial average 得到的 channel weight
        - $\operatorname{ReLU}$：只保留支持 class $c$ 的正证据

        它解决了 activation map 的痛点：activation 不是 class-specific；Grad-CAM 用 $\partial y^c/\partial A^k$ 把 heatmap 变成 target-class 相关。

        论文：Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017.

        ![gradcam decomposition](../artifacts/figures/gradcam_decomposition.png)
        """
    ),
    code(
        r"""
        model.eval()  # 确保模型处于 inference 模式。
        model.zero_grad(set_to_none=True)  # 清空旧梯度，避免污染当前解释。
        logits_gc, activation_gc = _cam_forward(model, image, "stage2")  # 前向传播并保留 stage2 activation 的梯度。
        score_gc = score_for_target(logits_gc, target)  # 取目标 class c 的 logit y^c。
        score_gc.backward()  # 从 y^c 反向传播到 stage2 activation。
        gradients_gc = activation_gc.grad  # 读取 ∂y^c/∂A。
        if gradients_gc is None:  # 如果没有梯度，说明 retain_grad 或 backward 出错。
            raise RuntimeError("Grad-CAM did not receive activation gradients")  # 显式报错，避免展示错误图。

        weights_gc = gradients_gc.mean(dim=(2, 3, 4), keepdim=True)  # 对 D,H,W 求平均，得到 α_k^c。
        weighted_activation_gc = weights_gc * activation_gc  # 计算每个 channel 的 α_k^c A^k。
        raw_cam_gc = weighted_activation_gc.sum(dim=1, keepdim=True)  # 对 channel 求和，得到 Σ_k α_k^c A^k。
        relu_cam_gc = raw_cam_gc.clamp_min(0)  # 应用 ReLU，只保留正向 class evidence。
        gradcam_heatmap = normalize_map(upsample_to_input(relu_cam_gc, image))  # 上采样到 24^3 并 normalize。
        feature_z_gc = feature_z(sample, activation_gc)  # 把输入中心 z 映射到 stage2 的 feature z。
        selected_channel_gc = int(weights_gc.abs().flatten().argmax().item())  # 选一个权重绝对值最大的 channel 做解释。
        """
    ),
    code(
        r"""
        panels_gc = [  # 组织 Grad-CAM 拆解图。
            (f"input x[z={center_z}]", volume_slice(sample["image"], center_z), "gray", 0, 1),  # 输入 slice。
            (f"A^{selected_channel_gc}", volume_slice(activation_gc[0, selected_channel_gc], feature_z_gc), "viridis", None, None),  # 单个 activation channel。
            (f"d y^c / d A^{selected_channel_gc}", volume_slice(gradients_gc[0, selected_channel_gc], feature_z_gc), "coolwarm", None, None),  # 对应梯度。
            (f"alpha*A^{selected_channel_gc}", volume_slice(weighted_activation_gc[0, selected_channel_gc], feature_z_gc), "viridis", None, None),  # channel 加权结果。
            ("Σ_k alpha_k A^k", volume_slice(raw_cam_gc, feature_z_gc), "viridis", None, None),  # channel summation。
            ("ReLU + Up", volume_slice(gradcam_heatmap, center_z), "turbo", 0, 1),  # 最终 heatmap。
            overlay_panel(gradcam_heatmap, "Grad-CAM overlay"),  # overlay 输出。
            ("target mask", volume_slice(sample["mask"], center_z), "Greens", 0, 1),  # GT mask 对照。
        ]  # 完成 Grad-CAM panel 列表。
        show_panels(panels_gc, "Method 02: gradients weight feature volumes", columns=4)  # 显示 Grad-CAM 拆解。
        """
    ),
    markdown(
        r"""
        ## Method 03 - Guided Grad-CAM

        PPT 对应页：`METHOD 03`

        公式：

        $$
        L_{\mathrm{GuidedGradCAM}}^c
        =
        \operatorname{GuidedBP}^c(x)\odot \operatorname{Up}(L_{\mathrm{GradCAM}}^c)
        $$

        符号：

        - $\operatorname{GuidedBP}^c(x)$：只允许正向 ReLU activation 和正向 gradient 通过的 input gradient
        - $L_{\mathrm{GradCAM}}^c$：低分辨率但 class-discriminative 的 Grad-CAM heatmap
        - $\odot$：逐体素相乘

        它解决 Grad-CAM 的痛点：Grad-CAM localization 好，但分辨率粗；Guided Grad-CAM 用 input-level gradient 恢复边缘细节。

        论文：Selvaraju et al., ICCV 2017；Guided Backprop 来自 Springenberg et al., "Striving for Simplicity", ICLR Workshop 2015.

        ![guided gradcam decomposition](../artifacts/figures/guided_gradcam_decomposition.png)
        """
    ),
    code(
        r"""
        model.eval()  # 确保模型处于 inference 模式。
        cam_for_guided = METHODS["gradcam"](model, image, target)  # 先计算 class-specific Grad-CAM heatmap。
        guided_raw = guided_backprop_raw(model, image, target)  # 计算 Guided Backprop 的原始 input gradient。
        guided_abs = normalize_map(guided_raw.abs().sum(dim=1, keepdim=True))  # 对 input channel 求绝对值和并 normalize。
        guided_product = guided_raw * cam_for_guided  # 用 Grad-CAM 空间 mask 过滤 Guided Backprop 细节。
        guided_gradcam_heatmap = normalize_map(guided_product.abs().sum(dim=1, keepdim=True))  # 得到 Guided Grad-CAM heatmap。
        """
    ),
    code(
        r"""
        panels_guided = [  # 组织 Guided Grad-CAM 拆解图。
            (f"input x[z={center_z}]", volume_slice(sample["image"], center_z), "gray", 0, 1),  # 输入 slice。
            ("GuidedBP", volume_slice(guided_abs, center_z), "magma", 0, 1),  # high-resolution input gradient。
            ("Grad-CAM", volume_slice(cam_for_guided, center_z), "turbo", 0, 1),  # low-resolution class localization。
            ("GuidedBP * Grad-CAM", volume_slice(guided_gradcam_heatmap, center_z), "turbo", 0, 1),  # 逐体素相乘后的结果。
            overlay_panel(guided_gradcam_heatmap, "Guided Grad-CAM overlay"),  # overlay 输出。
            ("target mask", volume_slice(sample["mask"], center_z), "Greens", 0, 1),  # GT mask 对照。
        ]  # 完成 Guided Grad-CAM panel 列表。
        show_panels(panels_guided, "Method 03: high-resolution gradient details gated by Grad-CAM", columns=3)  # 显示 Guided Grad-CAM 拆解。
        """
    ),
    markdown(
        r"""
        ## Method 04 - LayerCAM

        PPT 对应页：`METHOD 04`

        公式：

        $$
        L_{\mathrm{LayerCAM}}^c
        =
        \operatorname{ReLU}\left(\sum_k \operatorname{ReLU}\left(\frac{\partial y^c}{\partial A^k}\right)\odot A^k\right)
        $$

        符号：

        - $\operatorname{ReLU}(\partial y^c/\partial A^k)$：每个空间位置自己的正梯度权重
        - $\odot A^k$：不是对整个 channel 求一个全局 $\alpha_k^c$，而是在每个 voxel 保留 local weight

        它解决 Grad-CAM 的痛点：Grad-CAM 的 $\alpha_k^c$ 是全局 channel weight，可能丢掉局部空间差异；LayerCAM 保留 spatially local gradient。

        论文：Jiang et al., "LayerCAM: Exploring Hierarchical Class Activation Maps for Localization", IEEE TIP 2021.

        ![layercam decomposition](../artifacts/figures/layercam_decomposition.png)
        """
    ),
    code(
        r"""
        model.eval()  # 确保模型处于 inference 模式。
        model.zero_grad(set_to_none=True)  # 清空旧梯度。
        logits_lc, activation_lc = _cam_forward(model, image, "stage2")  # 前向传播并保留 stage2 activation 梯度。
        score_lc = score_for_target(logits_lc, target)  # 选择目标 class 的 logit。
        score_lc.backward()  # 从目标 logit 反向传播到 stage2。
        gradients_lc = activation_lc.grad  # 读取 ∂y^c/∂A。
        if gradients_lc is None:  # 检查梯度是否存在。
            raise RuntimeError("LayerCAM did not receive activation gradients")  # 没有梯度时停止，避免展示错误图。

        positive_grad_lc = gradients_lc.clamp_min(0)  # 只保留正梯度，表示支持 class c 的位置。
        local_evidence_lc = positive_grad_lc * activation_lc  # 在每个 voxel 计算 local gradient × activation。
        summed_lc = local_evidence_lc.sum(dim=1, keepdim=True).clamp_min(0)  # 对 channel 求和并保留正 evidence。
        layercam_heatmap = normalize_map(upsample_to_input(summed_lc, image))  # 上采样并 normalize。
        feature_z_lc = feature_z(sample, activation_lc)  # 映射 feature z。
        selected_channel_lc = int(local_evidence_lc.flatten(2).mean(dim=2).squeeze(0).argmax().item())  # 选平均 local evidence 最大的 channel。
        """
    ),
    code(
        r"""
        panels_lc = [  # 组织 LayerCAM 拆解图。
            (f"input x[z={center_z}]", volume_slice(sample["image"], center_z), "gray", 0, 1),  # 输入 slice。
            (f"A^{selected_channel_lc}", volume_slice(activation_lc[0, selected_channel_lc], feature_z_lc), "viridis", None, None),  # activation channel。
            (f"ReLU(dy/dA^{selected_channel_lc})", volume_slice(positive_grad_lc[0, selected_channel_lc], feature_z_lc), "magma", None, None),  # local 正梯度。
            ("local grad*A", volume_slice(local_evidence_lc[0, selected_channel_lc], feature_z_lc), "viridis", None, None),  # 单 channel local product。
            ("Σ_k local evidence", volume_slice(summed_lc, feature_z_lc), "viridis", None, None),  # 所有 channel 的 local evidence 和。
            ("LayerCAM heatmap", volume_slice(layercam_heatmap, center_z), "turbo", 0, 1),  # 最终 heatmap。
            overlay_panel(layercam_heatmap, "LayerCAM overlay"),  # overlay 输出。
            ("target mask", volume_slice(sample["mask"], center_z), "Greens", 0, 1),  # GT mask 对照。
        ]  # 完成 LayerCAM panel 列表。
        show_panels(panels_lc, "Method 04: keep gradient weights local instead of global", columns=4)  # 显示 LayerCAM 拆解。
        """
    ),
    markdown(
        r"""
        ## Method 05 - Occlusion Sensitivity

        PPT 对应页：`METHOD 05`

        公式：

        $$
        H(p)=s_c(x)-s_c(x_{\setminus p})
        $$

        符号：

        - $p$：被遮挡的 3D cube 位置
        - $x_{\setminus p}$：把位置 $p$ 的 cube 用 per-volume mean 替换后的输入
        - $s_c(x)$：模型对 class $c$ 的 softmax probability
        - $H(p)$：遮挡这个位置以后 target class probability 的下降量

        它解决梯度方法的痛点：不用依赖 gradient；直接问 “遮掉这里会不会让模型信心下降”。代价是计算更慢、分辨率受 mask size / stride 影响。

        背景：Zeiler and Fergus, "Visualizing and Understanding Convolutional Networks", ECCV 2014.

        ![occlusion decomposition](../artifacts/figures/occlusion_decomposition.png)
        """
    ),
    code(
        r"""
        mask_size_occ = 4  # 每次遮挡一个 4x4x4 cube，和当前代码默认设置一致。
        stride_occ = 4  # 遮挡窗口每次移动 4 个 voxel。
        fill_value_occ = float(image.mean().item())  # 用 per-volume mean 填充，避免制造黑色边缘 artifact。
        model.eval()  # 确保模型处于 inference 模式。
        with torch.no_grad():  # occlusion score 只需要 forward，不需要梯度。
            logits_occ = model(image)  # 原始输入的 logits。
            probs_occ = logits_occ.softmax(dim=1)  # 把 logits 转成 class probability。
            base_prob_occ = probs_occ.gather(1, target.view(1, 1)).view(1)  # 取 true class 的原始 probability。

        center_position_occ = (max(0, center_z - 2), max(0, center_y - 2), max(0, center_x - 2))  # 选择覆盖 target center 的遮挡位置。
        distractor_position_occ = (center_z, 4, 4)  # 选择同一 z slice 上的一个非 target 区域作为对照。
        background_position_occ = (center_z, image.shape[-2] - mask_size_occ - 4, image.shape[-1] - mask_size_occ - 4)  # 选择另一处背景区域。
        positions_occ = [center_position_occ, distractor_position_occ, background_position_occ]  # 把三个示例遮挡位置放在一起。
        """
    ),
    code(
        r"""
        occluded_panels = []  # 准备存放示例 occlusion 输入。
        for panel_id, (z0, y0, x0) in enumerate(positions_occ, start=1):  # 遍历三个示例遮挡位置。
            occluded = image.detach().clone()  # 复制原始输入，避免修改 sample 本身。
            occluded[:, :, z0:z0 + mask_size_occ, y0:y0 + mask_size_occ, x0:x0 + mask_size_occ] = fill_value_occ  # 用 mean value 替换 3D cube。
            with torch.no_grad():  # 对被遮挡输入只做 forward。
                occluded_prob = model(occluded).softmax(dim=1).gather(1, target.view(1, 1)).view(1)  # 取被遮挡后的 target probability。
            score_drop = float((base_prob_occ - occluded_prob).item())  # 计算 H(p)=原始概率-遮挡后概率。
            occluded_slice = volume_slice(occluded, center_z)  # 展示 target z slice 上的遮挡效果。
            occluded_panels.append((f"mask p{panel_id}, drop={score_drop:.3f}", occluded_slice, "gray", 0, 1))  # 保存示例 panel。

        show_panels(occluded_panels, "Method 05: example 3D mask cubes before full occlusion sweep", columns=3)  # 显示三个示例遮挡。
        """
    ),
    code(
        r"""
        occlusion_heatmap = occlusion_sensitivity(model, image, target, mask_size=mask_size_occ, stride=stride_occ)  # 扫描全 volume 得到 occlusion heatmap。
        panels_occ = [  # 组织 occlusion 输出图。
            (f"input x[z={center_z}]", volume_slice(sample["image"], center_z), "gray", 0, 1),  # 输入 slice。
            ("H(p) score drop", volume_slice(occlusion_heatmap, center_z), "turbo", 0, 1),  # 遮挡导致的 class probability 下降。
            overlay_panel(occlusion_heatmap, "Occlusion overlay"),  # overlay 输出。
            ("target mask", volume_slice(sample["mask"], center_z), "Greens", 0, 1),  # GT mask 对照。
        ]  # 完成 occlusion panel 列表。
        show_panels(panels_occ, "Method 05: full 3D occlusion sensitivity map", columns=4)  # 显示完整 occlusion heatmap。
        """
    ),
    markdown(
        r"""
        ## Method 06 - Integrated Gradients

        PPT 对应页：`METHOD 06`

        公式：

        $$
        \operatorname{IG}_i^c(x)
        =
        (x_i-x'_i)
        \int_0^1
        \frac{\partial y^c(x'+\alpha(x-x'))}{\partial x_i}
        d\alpha
        $$

        符号：

        - $x'$：baseline，本实验实现里对 Integrated Gradients 使用 zero baseline
        - $\alpha$：从 baseline 到输入的路径位置
        - $i$：input voxel index
        - $\frac{\partial y^c}{\partial x_i}$：路径上每个输入 voxel 的 gradient

        它解决单点 gradient 的痛点：只在当前输入点看梯度容易饱和；Integrated Gradients 沿 baseline 到 input 的路径积分。

        论文：Sundararajan et al., "Axiomatic Attribution for Deep Networks", ICML 2017.

        ![integrated gradients decomposition](../artifacts/figures/integrated_gradients_decomposition.png)
        """
    ),
    code(
        r"""
        steps_ig = 32  # 用 32 个离散点近似积分，和 detail 图保持轻量。
        baseline_ig = torch.zeros_like(image)  # Integrated Gradients 当前实现使用 zero baseline。
        total_grad_ig = torch.zeros_like(image)  # 初始化路径上 input gradient 的累加器。
        alphas_ig = torch.linspace(0, 1, steps_ig + 1, device=image.device, dtype=image.dtype)[1:]  # 构造从 0 到 1 的积分路径，跳过 alpha=0。

        for alpha_ig in alphas_ig:  # 遍历 baseline 到 input 的每个路径点。
            model.zero_grad(set_to_none=True)  # 清空旧梯度。
            x_step_ig = (baseline_ig + alpha_ig * (image - baseline_ig)).detach().requires_grad_(True)  # 构造路径点 x'+alpha(x-x')。
            logits_ig = model(x_step_ig)  # 在路径点做 forward。
            grad_ig = torch.autograd.grad(score_for_target(logits_ig, target), x_step_ig)[0]  # 计算 dy^c/dx_step。
            total_grad_ig += grad_ig  # 累加路径梯度。

        avg_grad_ig = total_grad_ig / steps_ig  # 用离散平均近似积分。
        attribution_ig = (image - baseline_ig) * avg_grad_ig  # 乘以输入和 baseline 的差值。
        raw_ig_heatmap = normalize_map(attribution_ig.abs().sum(dim=1, keepdim=True))  # 对 channel 聚合并 normalize。
        smoothed_ig_heatmap = gaussian_blur3d(raw_ig_heatmap, sigma=0.8)  # 轻微 3D smoothing，让 1-voxel target 更可读。
        integrated_gradients_heatmap = normalize_map(smoothed_ig_heatmap)  # smoothing 后重新 normalize 到 [0,1]。
        midpoint_ig = baseline_ig + 0.5 * (image - baseline_ig)  # 取路径中点，帮助解释 baseline-to-input path。
        """
    ),
    code(
        r"""
        panels_ig = [  # 组织 Integrated Gradients 拆解图。
            ("baseline x'", volume_slice(baseline_ig, center_z), "gray", 0, 1),  # baseline slice。
            (f"input x[z={center_z}]", volume_slice(sample["image"], center_z), "gray", 0, 1),  # 输入 slice。
            ("path alpha=0.5", volume_slice(midpoint_ig, center_z), "gray", 0, 1),  # 路径中点。
            ("avg |dy/dx|", volume_slice(avg_grad_ig.abs().sum(dim=1, keepdim=True), center_z), "magma", None, None),  # 平均 input gradient。
            ("(x-x')*avg grad", volume_slice(raw_ig_heatmap, center_z), "turbo", 0, 1),  # attribution heatmap before smoothing。
            ("IG heatmap", volume_slice(integrated_gradients_heatmap, center_z), "turbo", 0, 1),  # smoothing 后输出。
            overlay_panel(integrated_gradients_heatmap, "IG overlay"),  # overlay 输出。
            ("target mask", volume_slice(sample["mask"], center_z), "Greens", 0, 1),  # GT mask 对照。
        ]  # 完成 IG panel 列表。
        show_panels(panels_ig, "Method 06: integrate input gradients along the baseline path", columns=4)  # 显示 IG 拆解。
        """
    ),
    markdown(
        r"""
        ## Method 07 - Integrated Grad-CAM

        PPT 对应页：`METHOD 07`

        公式：

        $$
        L_{\mathrm{IGC}}^c
        =
        \operatorname{Norm}
        \left(
        \operatorname{Up}
        \left(
        \sum_k
        \left|
        \Delta A^k
        \cdot
        \frac{1}{m}\sum_{t=1}^{m}
        \frac{\partial y_t^c}{\partial A_t^k}
        \right|
        \right)
        \right)
        $$

        符号：

        - $x'$：baseline；这里用 per-volume mean baseline，避免纯黑 baseline 引入不自然边界
        - $A^k$：输入 $x$ 的 stage2 feature volume
        - $A'^k$：baseline $x'$ 的 stage2 feature volume
        - $\Delta A^k=A^k-A'^k$：feature space 的输入差异
        - $t$：baseline-to-input path 上的第 $t$ 个插值点
        - $m$：积分路径采样步数

        它解决 Grad-CAM 和 IG 的折中问题：Grad-CAM 在 feature space 做 class localization，但只看单点梯度；IG 沿路径积分但在 input space 可能很细碎。IGC 在 feature space 上做路径积分。

        背景：Integrated Grad-CAM / path-integrated CAM 系列方法沿用 Integrated Gradients 的路径积分思想，把 attribution 放到 convolutional feature map 上。

        ![integrated gradcam decomposition](../artifacts/figures/integrated_gradcam_decomposition.png)
        """
    ),
    code(
        r"""
        steps_igc = 20  # Integrated Grad-CAM 当前实现使用 20 个路径采样点。
        baseline_igc = torch.ones_like(image) * image.detach().mean(dim=(1, 2, 3, 4), keepdim=True)  # 用 per-volume mean baseline。
        model.eval()  # 确保模型处于 inference 模式。
        with torch.no_grad():  # 计算 baseline/input feature 不需要梯度。
            _, input_features_igc = model(image, return_features=True)  # 取真实输入的 feature maps。
            _, baseline_features_igc = model(baseline_igc, return_features=True)  # 取 baseline 的 feature maps。
            feature_delta_igc = input_features_igc["stage2"] - baseline_features_igc["stage2"]  # 计算 ΔA = A(x)-A(x')。

        total_grad_igc = torch.zeros_like(feature_delta_igc)  # 初始化 feature-gradient 累加器。
        alphas_igc = torch.linspace(0, 1, steps_igc + 1, device=image.device, dtype=image.dtype)[1:]  # 构造 baseline 到 input 的路径采样点。

        for alpha_igc in alphas_igc:  # 遍历路径上的每个采样点。
            model.zero_grad(set_to_none=True)  # 清空旧梯度。
            x_step_igc = (baseline_igc + alpha_igc * (image - baseline_igc)).detach()  # 构造路径输入 x_t。
            logits_igc, activation_igc = _cam_forward(model, x_step_igc, "stage2")  # 前向并保留 stage2 activation gradient。
            score_for_target(logits_igc, target).backward()  # 反向传播 class c 的 logit。
            gradients_igc = activation_igc.grad  # 读取路径点上的 ∂y_t^c/∂A_t。
            if gradients_igc is None:  # 检查梯度是否存在。
                raise RuntimeError("Integrated Grad-CAM did not receive activation gradients")  # 无梯度时立即停止。
            total_grad_igc += gradients_igc  # 累加路径上的 feature gradients。

        avg_feature_grad_igc = total_grad_igc / steps_igc  # 对路径梯度求平均，近似积分。
        contribution_igc = avg_feature_grad_igc * feature_delta_igc  # 计算 ΔA^k · avg_grad^k。
        cam_igc = contribution_igc.abs().sum(dim=1, keepdim=True)  # 对 channel 求绝对贡献和。
        integrated_gradcam_heatmap = normalize_map(upsample_to_input(cam_igc, image))  # 上采样到输入大小并 normalize。
        feature_z_igc = feature_z(sample, feature_delta_igc)  # 把输入中心 z 映射到 feature z。
        """
    ),
    code(
        r"""
        panels_igc = [  # 组织 Integrated Grad-CAM 拆解图。
            ("baseline x'", volume_slice(baseline_igc, center_z), "gray", 0, 1),  # mean baseline slice。
            (f"input x[z={center_z}]", volume_slice(sample["image"], center_z), "gray", 0, 1),  # 输入 slice。
            ("mean ΔA", volume_slice(feature_delta_igc.mean(dim=1, keepdim=True), feature_z_igc), "viridis", None, None),  # feature delta 的 channel mean。
            ("avg dy/dA", volume_slice(avg_feature_grad_igc.mean(dim=1, keepdim=True), feature_z_igc), "magma", None, None),  # 路径平均 feature gradient。
            ("|ΔA*avg grad|", volume_slice(cam_igc, feature_z_igc), "viridis", None, None),  # feature-space contribution。
            ("IGC heatmap", volume_slice(integrated_gradcam_heatmap, center_z), "turbo", 0, 1),  # 最终 heatmap。
            overlay_panel(integrated_gradcam_heatmap, "IGC overlay"),  # overlay 输出。
            ("target mask", volume_slice(sample["mask"], center_z), "Greens", 0, 1),  # GT mask 对照。
        ]  # 完成 IGC panel 列表。
        show_panels(panels_igc, "Method 07: path-integrated feature-volume contribution", columns=4)  # 显示 IGC 拆解。
        """
    ),
    markdown(
        r"""
        ## 最后对齐 PPT 的方法对比

        下面只使用当前 PPT 已经包含的七个方法，不加入新方法。这个对比 cell 可以作为 DEMO 页的 notebook 版。

        ![class discriminability](../artifacts/figures/class_discriminability.png)
        """
    ),
    code(
        r"""
        method_order = ["notgradcam", "gradcam", "guided_gradcam", "layercam", "occlusion", "integrated_gradients", "integrated_gradcam"]  # 固定为 PPT 当前七个方法。
        comparison_panels = [(f"input y={int(sample['label'])}", volume_slice(sample["image"], center_z), "gray", 0, 1)]  # 第一列展示输入。
        comparison_panels.append(("target mask", volume_slice(sample["mask"], center_z), "Greens", 0, 1))  # 第二列展示 GT。
        for method_name in method_order:  # 逐个运行 PPT 中的 attribution 方法。
            heatmap = METHODS[method_name](model, image, target)  # 计算该方法的 3D heatmap。
            comparison_panels.append(overlay_panel(heatmap, method_name))  # 把 heatmap overlay 到输入 slice 上。

        show_panels(comparison_panels, "PPT-matched method comparison on the same 3D sample", columns=3)  # 展示同一样本的七方法对比。
        """
    ),
    code(
        r"""
        for method_name in method_order:  # 按 PPT 顺序打印已有量化指标。
            method_score = scores[method_name]  # 取出该方法的 score dict。
            mass = method_score["mass_in_gt"]  # 读取 heatmap mass inside GT。
            ratio = method_score["inside_outside_ratio"]  # 读取 GT 内外平均热度比。
            pointing = method_score["pointing_acc"]  # 读取 pointing accuracy。
            print(f"{method_name:22s} mass_in_gt={mass:.3f} inside/outside={ratio:.2f} pointing={pointing:.1f}")  # 打印一行可讲述的指标摘要。
        """
    ),
    markdown(
        r"""
        ## Notebook 和 PPT 的对应关系

        - `Experimental setup` 页：对应 Part 1、模型 shape check、3-plane target visualization。
        - `METHOD 01` 页：对应 notGradCAM 的 $A^k \rightarrow \sum A^k \rightarrow \frac{1}{K}\sum A^k$。
        - `METHOD 02` 页：对应 Grad-CAM 的 gradient average weight $\alpha_k^c$。
        - `METHOD 03` 页：对应 GuidedBP 与 Grad-CAM 的逐体素乘法。
        - `METHOD 04` 页：对应 LayerCAM 的 local positive gradient。
        - `METHOD 05` 页：对应 3D cube occlusion 和 score drop。
        - `METHOD 06` 页：对应 Integrated Gradients 的 input-space path integral。
        - `METHOD 07` 页：对应 Integrated Grad-CAM 的 feature-space path integral。

        后续如果 PPT 继续改，我们优先改实验代码和这个 notebook，再重新导出 PPT 图，保证每张图都有代码来源。
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
