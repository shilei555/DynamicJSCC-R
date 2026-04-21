import os
import torch
import torchvision
from datetime import datetime
import base64


def save_image_pairs(images, x_rec, epoch, batch_idx, save_dir, max_samples=8):
    """
    保存原始图像和重建图像的对比图

    Args:
        images: 原始图像 [B, C, H, W]
        x_rec: 重建图像 [B, C, H, W]
        epoch: 当前 epoch
        batch_idx: 当前 batch 索引
        save_dir: 保存目录
        max_samples: 最多保存多少对图像

    Returns:
        saved_paths: 保存的图像路径列表
    """
    os.makedirs(save_dir, exist_ok=True)

    # 限制保存数量
    num_samples = min(images.size(0), max_samples)
    images = images[:num_samples].cpu()
    x_rec = x_rec[:num_samples].cpu()

    # 将图像 clamp 到 [0, 1] 范围
    images = torch.clamp(images, 0, 1)
    x_rec = torch.clamp(x_rec, 0, 1)

    saved_paths = []
    for i in range(num_samples):
        # 创建对比图：左边原图，右边重建图
        combined = torch.cat([images[i], x_rec[i]], dim=2)  # [C, H, W*2]

        # 保存图像
        save_path = os.path.join(save_dir, f'epoch_{epoch:03d}_batch_{batch_idx:04d}_sample_{i:02d}.png')
        torchvision.utils.save_image(combined, save_path)
        saved_paths.append(save_path)

    return saved_paths


def generate_batch_html(epoch, batch_idx, image_paths, html_path, loss_info=None):
    """
    生成单个 batch 的 HTML 报告

    Args:
        epoch: 当前 epoch
        batch_idx: 当前 batch 索引
        image_paths: 图像路径列表
        html_path: HTML 文件保存路径
        loss_info: 损失信息字典 (可选)
    """
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Epoch {epoch:03d} Batch {batch_idx:04d} - DynamicJSCC-R</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .info {{
            text-align: center;
            margin-bottom: 20px;
            color: #666;
        }}
        .loss-info {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin: 20px auto;
            max-width: 600px;
            text-align: center;
        }}
        .loss-item {{
            display: inline-block;
            margin: 0 20px;
        }}
        .loss-value {{
            font-size: 24px;
            font-weight: bold;
        }}
        .loss-label {{
            font-size: 14px;
            opacity: 0.9;
        }}
        .grid-container {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(450px, 1fr));
            gap: 20px;
            padding: 20px;
        }}
        .image-card {{
            background: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.2s;
        }}
        .image-card:hover {{
            transform: scale(1.02);
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        }}
        .image-card img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            border: 1px solid #ddd;
        }}
        .image-label {{
            margin-top: 10px;
            font-weight: bold;
            color: #555;
        }}
        .legend {{
            display: flex;
            justify-content: center;
            gap: 40px;
            margin: 20px 0;
            background: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .legend-left {{
            width: 30px;
            height: 20px;
            background: linear-gradient(90deg, #4CAF50 50%, #2196F3 50%);
            border-radius: 3px;
        }}
        .legend-text {{
            color: #666;
        }}
        .nav-links {{
            text-align: center;
            margin: 20px;
        }}
        .nav-links a {{
            display: inline-block;
            padding: 10px 20px;
            margin: 0 10px;
            background: #2196F3;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            transition: background 0.2s;
        }}
        .nav-links a:hover {{
            background: #1976D2;
        }}
        .timestamp {{
            text-align: center;
            color: #999;
            margin-top: 30px;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <h1>🖼️ DynamicJSCC-R Training Visualization</h1>
    <div class="info">
        <h2>📊 Epoch: {epoch:03d} | Batch: {batch_idx:04d}</h2>
    </div>
"""

    # 添加损失信息（如果提供）
    if loss_info:
        html_content += f"""
    <div class="loss-info">
        <div class="loss-item">
            <div class="loss-value">{loss_info.get('total_loss', 0):.4f}</div>
            <div class="loss-label">Total Loss</div>
        </div>
        <div class="loss-item">
            <div class="loss-value">{loss_info.get('mse_loss', 0):.4f}</div>
            <div class="loss-label">MSE Loss</div>
        </div>
        <div class="loss-item">
            <div class="loss-value">{loss_info.get('ce_loss', 0):.4f}</div>
            <div class="loss-label">CE Loss</div>
        </div>
        <div class="loss-item">
            <div class="loss-value">{loss_info.get('accuracy', 0):.2f}%</div>
            <div class="loss-label">Accuracy</div>
        </div>
    </div>
"""

    html_content += """
    <div class="legend">
        <div class="legend-item">
            <div class="legend-left"></div>
            <span class="legend-text">⬅️ Original Image &nbsp;&nbsp;|&nbsp;&nbsp; Reconstructed Image ➡️</span>
        </div>
    </div>
    <div class="grid-container">
"""

    for i, img_path in enumerate(image_paths):
        # 将图像转换为 base64 嵌入 HTML
        with open(img_path, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode('utf-8')

        html_content += f"""
        <div class="image-card">
            <img src="data:image/png;base64,{img_base64}" alt="Sample {i}">
            <div class="image-label">📌 Sample {i:02d}</div>
        </div>
"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 导航链接
    html_content += f"""
    </div>
    <div class="nav-links">
        <a href="../index.html">📁 Back to Index</a>
        <a href="epoch_{epoch:03d}_summary.html">📋 Epoch Summary</a>
    </div>
    <div class="timestamp">
        Generated at: {timestamp}
    </div>
</body>
</html>
"""

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def generate_epoch_summary_html(epoch, batch_records, summary_path):
    """
    生成单个 epoch 的汇总页面，显示该 epoch 所有保存的 batch

    Args:
        epoch: 当前 epoch
        batch_records: 该 epoch 保存的所有 batch 记录列表
            每个元素为 dict: {'batch_idx': int, 'html_path': str, 'loss_info': dict, 'timestamp': str}
        summary_path: 汇总 HTML 保存路径
    """
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Epoch {epoch:03d} Summary - DynamicJSCC-R</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .batch-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }}
        .batch-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
            text-align: center;
        }}
        .batch-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        }}
        .batch-card a {{
            text-decoration: none;
            color: #333;
        }}
        .batch-number {{
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #2196F3;
        }}
        .loss-preview {{
            font-size: 14px;
            color: #666;
            margin: 10px 0;
        }}
        .loss-value {{
            font-size: 20px;
            font-weight: bold;
            color: #e74c3c;
        }}
        .timestamp {{
            font-size: 12px;
            color: #999;
            margin-top: 15px;
        }}
        .nav-links {{
            text-align: center;
            margin: 30px 0 10px;
        }}
        .nav-links a {{
            display: inline-block;
            padding: 12px 25px;
            margin: 0 10px;
            background: #2196F3;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            font-size: 16px;
        }}
        .nav-links a:hover {{
            background: #1976D2;
        }}
        .stats {{
            display: flex;
            justify-content: center;
            gap: 40px;
            margin: 20px 0;
            padding: 20px;
            background: #f0f0f0;
            border-radius: 10px;
        }}
        .stat-item {{
            text-align: center;
        }}
        .stat-number {{
            font-size: 32px;
            font-weight: bold;
            color: #2196F3;
        }}
        .stat-label {{
            font-size: 14px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Epoch {epoch:03d} Summary</h1>

        <div class="stats">
            <div class="stat-item">
                <div class="stat-number">{len(batch_records)}</div>
                <div class="stat-label">Batches Saved</div>
            </div>
"""

    # 计算平均损失
    if batch_records:
        avg_total = sum(r.get('loss_info', {}).get('total_loss', 0) for r in batch_records) / len(batch_records)
        avg_mse = sum(r.get('loss_info', {}).get('mse_loss', 0) for r in batch_records) / len(batch_records)
        avg_ce = sum(r.get('loss_info', {}).get('ce_loss', 0) for r in batch_records) / len(batch_records)

        html_content += f"""
            <div class="stat-item">
                <div class="stat-number">{avg_total:.4f}</div>
                <div class="stat-label">Avg Total Loss</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{avg_mse:.4f}</div>
                <div class="stat-label">Avg MSE</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{avg_ce:.4f}</div>
                <div class="stat-label">Avg CE</div>
            </div>
"""

    html_content += """
        </div>

        <div class="batch-grid">
"""

    # 按 batch_idx 排序
    batch_records_sorted = sorted(batch_records, key=lambda x: x['batch_idx'])

    for record in batch_records_sorted:
        batch_idx = record['batch_idx']
        html_rel_path = os.path.basename(record['html_path'])
        loss_info = record.get('loss_info', {})
        timestamp = record.get('timestamp', '')

        html_content += f"""
            <div class="batch-card">
                <a href="{html_rel_path}">
                    <div class="batch-number">Batch {batch_idx:04d}</div>
                    <div class="loss-preview">
                        Loss: <span class="loss-value">{loss_info.get('total_loss', 0):.4f}</span>
                    </div>
                    <div class="loss-preview">
                        MSE: {loss_info.get('mse_loss', 0):.4f} | CE: {loss_info.get('ce_loss', 0):.4f}
                    </div>
                    <div class="loss-preview">
                        Acc: {loss_info.get('accuracy', 0):.2f}%
                    </div>
                    <div class="timestamp">{timestamp}</div>
                </a>
            </div>
"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_content += f"""
        </div>

        <div class="nav-links">
            <a href="../index.html">📁 Back to Index</a>
        </div>

        <div class="timestamp">
            Generated at: {timestamp}
        </div>
    </div>
</body>
</html>
"""

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def update_index_html(vis_dir, epoch, batch_idx, html_path, loss_info=None):
    """
    更新总索引页面，列出所有 epoch 和 batch
    """
    index_path = os.path.join(vis_dir, 'index.html')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 读取或创建 epoch_records
    epoch_records = {}
    records_file = os.path.join(vis_dir, 'records.txt')

    if os.path.exists(records_file):
        with open(records_file, 'r') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 5:
                    ep = int(parts[0])
                    bt = int(parts[1])
                    if ep not in epoch_records:
                        epoch_records[ep] = []
                    epoch_records[ep].append({
                        'batch_idx': bt,
                        'html_path': parts[2],
                        'total_loss': float(parts[3]),
                        'timestamp': parts[4]
                    })

    # 添加新记录
    if epoch not in epoch_records:
        epoch_records[epoch] = []

    epoch_records[epoch].append({
        'batch_idx': batch_idx,
        'html_path': os.path.relpath(html_path, vis_dir),
        'total_loss': loss_info.get('total_loss', 0) if loss_info else 0,
        'timestamp': timestamp
    })

    # 保存记录
    with open(records_file, 'w') as f:
        for ep, batches in epoch_records.items():
            for bt in batches:
                f.write(f"{ep}|{bt['batch_idx']}|{bt['html_path']}|{bt['total_loss']}|{bt['timestamp']}\n")

    # 生成 HTML
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>DynamicJSCC-R Visualization Index</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .epoch-section {{
            margin-bottom: 30px;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
        }}
        .epoch-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            font-size: 20px;
            font-weight: bold;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .epoch-header a {{
            color: white;
            text-decoration: none;
            padding: 5px 15px;
            background: rgba(255,255,255,0.2);
            border-radius: 20px;
            font-size: 14px;
        }}
        .epoch-header a:hover {{
            background: rgba(255,255,255,0.3);
        }}
        .batch-list {{
            padding: 15px;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .batch-link {{
            display: inline-block;
            padding: 10px 18px;
            background: #f0f0f0;
            border-radius: 8px;
            text-decoration: none;
            color: #333;
            transition: all 0.2s;
            border-left: 4px solid #2196F3;
        }}
        .batch-link:hover {{
            background: #2196F3;
            color: white;
            transform: translateY(-2px);
        }}
        .batch-loss {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}
        .batch-link:hover .batch-loss {{
            color: #e0e0e0;
        }}
        .total-batches {{
            color: #999;
            font-size: 14px;
            margin-left: 10px;
        }}
        .timestamp {{
            text-align: center;
            color: #999;
            margin-top: 30px;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📁 DynamicJSCC-R Visualization Index</h1>
"""

    # 按 epoch 降序排列
    for epoch_num in sorted(epoch_records.keys(), reverse=True):
        batches = sorted(epoch_records[epoch_num], key=lambda x: x['batch_idx'])

        html_content += f"""
        <div class="epoch-section">
            <div class="epoch-header">
                <span>📊 Epoch {epoch_num:03d} <span class="total-batches">({len(batches)} batches)</span></span>
                <a href="epoch_{epoch_num:03d}/epoch_{epoch_num:03d}_summary.html">View Summary →</a>
            </div>
            <div class="batch-list">
"""
        for bt in batches:
            html_content += f"""
                <a href="{bt['html_path']}" class="batch-link">
                    Batch {bt['batch_idx']:04d}
                    <div class="batch-loss">Loss: {bt['total_loss']:.4f}</div>
                </a>
"""
        html_content += """
            </div>
        </div>
"""

    html_content += f"""
        <div class="timestamp">
            Last updated: {timestamp}
        </div>
    </div>
</body>
</html>
"""

    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    # 更新 epoch 汇总页面
    for epoch_num, batches in epoch_records.items():
        summary_path = os.path.join(vis_dir, f'epoch_{epoch_num:03d}', f'epoch_{epoch_num:03d}_summary.html')
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)

        # 构建完整的 loss_info
        batch_records = []
        for bt in batches:
            batch_records.append({
                'batch_idx': bt['batch_idx'],
                'html_path': bt['html_path'],
                'loss_info': {'total_loss': bt['total_loss']},
                'timestamp': bt['timestamp']
            })

        generate_epoch_summary_html(epoch_num, batch_records, summary_path)


def save_checkpoint_with_images(model, optimizer, epoch, batch_idx, images, x_rec, config, loss_dict=None):
    """
    保存图像对并生成 HTML 报告
    """
    # 创建保存目录
    vis_dir = os.path.join(config.save_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # 子目录：按 epoch 组织
    epoch_dir = os.path.join(vis_dir, f'epoch_{epoch:03d}')
    os.makedirs(epoch_dir, exist_ok=True)

    # 保存图像对
    image_paths = save_image_pairs(
        images, x_rec, epoch, batch_idx,
        epoch_dir, max_samples=config.vis_max_samples
    )

    # 准备损失信息
    loss_info = None
    if loss_dict:
        loss_info = {
            'total_loss': loss_dict.get('total_loss', 0),
            'mse_loss': loss_dict.get('mse_loss', 0),
            'ce_loss': loss_dict.get('ce_loss', 0),
            'accuracy': loss_dict.get('accuracy', 0)
        }

    # 生成单个 batch 的 HTML 报告
    html_path = os.path.join(epoch_dir, f'epoch_{epoch:03d}_batch_{batch_idx:04d}.html')
    generate_batch_html(epoch, batch_idx, image_paths, html_path, loss_info)

    # 更新索引页面（累加所有保存的 batch）
    update_index_html(vis_dir, epoch, batch_idx, html_path, loss_info)

    return html_path