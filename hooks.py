
import os
import json
import torch
import numpy as np
from PIL import Image
import folder_paths
import execution
import time

# オリジナルのexecute関数を保持
original_execute = execution.execute

# ベースの保存先ディレクトリ
HOOK_BASE_DIR = os.path.join(folder_paths.get_output_directory(), "hooked_outputs")

def get_save_dir(prompt_id):
    # prompt_id (実行ID) ごとにフォルダを分ける
    # タイムスタンプは実行開始時ではなくノード実行時になってしまうが、prompt_id自体がユニークなIDとして使える
    # 分かりやすさのためにディレクトリ名に日付などは入れず prompt_id だけにするか、
    # どこかで開始時刻を取れれば良いが、単純化のため prompt_id をフォルダ名とする
    save_dir = os.path.join(HOOK_BASE_DIR, prompt_id)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def save_image(images, filename_prefix, save_dir):
    results = []
    # Tensor以外（リストなど）が来る場合もあるので再帰的に処理するか、Tensorのみ抽出
    if isinstance(images, list):
        for img in images:
            results.extend(save_image(img, filename_prefix, save_dir))
        return results

    if not isinstance(images, torch.Tensor):
        return []

    # 画像として保存可能なTensorかチェック (B, H, W, C)
    if images.ndim == 4 and images.shape[-1] in [1, 3, 4]:
        for batch_idx, img_tensor in enumerate(images):
            try:
                img_array = 255. * img_tensor.cpu().numpy()
                img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
                
                filename = f"{filename_prefix}_{batch_idx}.png"
                file_path = os.path.join(save_dir, filename)
                img.save(file_path, optimize=True)
                results.append(filename)
            except Exception as e:
                # 稀に形状が合わない等のエラーがあっても全体を止めない
                print(f"[Hook Image Save Error] {e}")
    
    return results

def serialize_value(value):
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    elif isinstance(value, torch.Tensor):
        return f"Tensor shape={value.shape} dtype={value.dtype} device={value.device}"
    elif isinstance(value, list):
        return [serialize_value(x) for x in value]
    elif isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    else:
        return str(type(value))

async def new_execute(server, dynprompt, caches, current_item, extra_data, executed, prompt_id, execution_list, pending_subgraph_results, pending_async_nodes, ui_outputs):
    # オリジナルの実行
    result = await original_execute(server, dynprompt, caches, current_item, extra_data, executed, prompt_id, execution_list, pending_subgraph_results, pending_async_nodes, ui_outputs)
    
    # 実行成功時のみ処理
    if result[0] == execution.ExecutionResult.SUCCESS:
        try:
            # キャッシュから出力を取得
            cached = caches.outputs.get(current_item)
            if cached is not None and cached.outputs:
                timestamp = int(time.time())
                node_info = dynprompt.get_node(current_item)
                class_type = node_info.get("class_type", "Unknown")
                
                # 保存ディレクトリの決定
                save_dir = get_save_dir(prompt_id)
                
                # ファイル名のプレフィックス (タイムスタンプ_ノードID_タイプ)
                filename_base = f"{timestamp}_{current_item}_{class_type}"
                
                saved_files = []
                outputs_summary = []

                # 全ての出力を走査して保存
                for out_idx, output_data in enumerate(cached.outputs):
                    try:
                        # save_dir を渡すように変更
                        files = save_image(output_data, f"{filename_base}_out{out_idx}", save_dir)
                        if files:
                            saved_files.extend(files)
                            outputs_summary.append({"index": out_idx, "type": "image", "files": files})
                        else:
                            val_str = serialize_value(output_data)
                            outputs_summary.append({"index": out_idx, "type": "other", "value": val_str})
                    except Exception as e:
                        outputs_summary.append({"index": out_idx, "type": "error", "error": str(e)})

                # メタデータ(実行ログ)の保存
                metadata = {
                    "node_id": current_item,
                    "class_type": class_type,
                    "prompt_id": prompt_id,
                    "timestamp": timestamp,
                    "outputs": outputs_summary,
                    "inputs": node_info.get("inputs"),
                    "workflow": dynprompt.get_original_prompt()
                }
                
                json_path = os.path.join(save_dir, f"{filename_base}_meta.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=4, ensure_ascii=False)
                            
        except Exception as e:
            print(f"[Hook Error] Failed to save output for node {current_item}: {e}")
            import traceback
            traceback.print_exc()

    return result

def apply_hook():
    execution.execute = new_execute
    print(f"ComfyUI Execution Hook Applied: All intermediate outputs will be logged to {HOOK_BASE_DIR}/<prompt_id>/")
