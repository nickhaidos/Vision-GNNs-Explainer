import torch
import numpy as np
import logging
import argparse
import json
from pathlib import Path
import vig_2
from utils import *

# --- Global Mappings (loaded if paths are provided) ---
CLASS_IDX_TO_NAME_MAPPING = {} # Populated by imagenet1000_clsidx_to_labels.txt

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.StreamHandler() # Log to console
    ]
)


# --- Main Orchestration ---
def main(args):
    """
    Main function to run the Vision GNN explanation analysis for a single image.
    """
    logging.info(f"Starting analysis for image: {args.im_path}")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logging.info(f"Using device: {device}")

    global CLASS_IDX_TO_NAME_MAPPING
    if args.cls_to_labels_path:
        try:
            with open(args.cls_to_labels_path, 'r') as f:
                CLASS_IDX_TO_NAME_MAPPING = eval(f.read()) 
            logging.info(f"Loaded class index to name mapping from: {args.cls_to_labels_path}")
        except Exception as e:
            logging.error(f"Could not load class mapping from {args.cls_to_labels_path}: {e}")

    logging.info(f"Loading ViG model: {args.model_variant} from {args.model_weights}")
    try:
        model_constructor = getattr(vig_2, args.model_variant)
        model = model_constructor().to(device)
    except AttributeError:
        logging.error(f"Model variant '{args.model_variant}' not found in vig_2.py.")
        return
    except Exception as e:
        logging.error(f"Error initializing model {args.model_variant}: {e}")
        return

    try:
        model.load_state_dict(torch.load(args.model_weights, map_location=device, weights_only=True))
        logging.info("Model weights loaded successfully.")
    except FileNotFoundError:
        logging.error(f"Model weights file not found: {args.model_weights}")
        return
    except Exception as e:
        logging.error(f"Error loading model weights: {e}")
        return
    
    try:
        image_tensor = image_to_tensor(args.im_path, device)
    except Exception:
        return 

    try:
        logging.info("Running model inference...")
        final_logits, edge_indexes_per_layer, block_features_per_layer = run_model_inference(model, image_tensor)
        logging.info("Model inference complete.")
        logging.debug(f"Final logits shape: {final_logits.shape if final_logits is not None else 'None'}")
        if edge_indexes_per_layer:
            logging.debug(f"Num edge index tensors: {len(edge_indexes_per_layer)}, first shape: {edge_indexes_per_layer[0].shape if edge_indexes_per_layer[0] is not None else 'None'}")
        if block_features_per_layer:
            logging.debug(f"Num block feature tensors: {len(block_features_per_layer)}, first shape: {block_features_per_layer[0].shape if block_features_per_layer[0] is not None else 'None'}")

    except Exception as e:
        logging.error(f"Error during model inference: {e}", exc_info=True)
        return

    results = {}
    results["visual_similarity"] = calculate_visual_similarity(edge_indexes_per_layer, image_tensor)
    results["spatial_distance"] = calculate_spatial_distance(edge_indexes_per_layer)
    results["embedding_similarity"] = calculate_embedding_similarity(edge_indexes_per_layer, block_features_per_layer)

    target_class_idx_for_prob = args.gt_label_idx
    if target_class_idx_for_prob is not None:
        probs, correct = calculate_layer_probabilities(model, block_features_per_layer, target_class_idx_for_prob)
        results["layer_probabilities_gt"] = probs
        results["layer_is_correct_gt"] = correct
    else:
        results["layer_probabilities_gt"] = None
        results["layer_is_correct_gt"] = None
        logging.info("Ground truth label index not provided, skipping probability metrics.")

    object_mask_224 = None
    prompt_for_dino = args.gt_label_name
    if not prompt_for_dino and args.gt_label_idx is not None and CLASS_IDX_TO_NAME_MAPPING:
        raw_mapping = CLASS_IDX_TO_NAME_MAPPING.get(args.gt_label_idx)
        if raw_mapping:
            if isinstance(raw_mapping, (list, tuple)):
                prompt_for_dino = raw_mapping[-1] 
            elif isinstance(raw_mapping, str) and ',' in raw_mapping:
                 prompt_for_dino = raw_mapping.split(',')[-1].strip().replace("'", "")
            else: # Assume it's already the class name string
                prompt_for_dino = raw_mapping

            logging.info(f"Derived DINO prompt from gt_label_idx: '{prompt_for_dino}'")
        else:
            logging.warning(f"Could not find class name for index {args.gt_label_idx} in mapping for DINO prompt.")
            
    if prompt_for_dino and DINO_SAM_AVAILABLE and args.dino_config and args.dino_weights and args.sam_checkpoint:
        object_mask_224 = generate_segmentation_mask(
            image_path_str=args.im_path,
            text_prompt=prompt_for_dino,
            dino_config_path=args.dino_config,
            dino_weights_path=args.dino_weights,
            sam_checkpoint_path=args.sam_checkpoint,
            sam_model_type=args.sam_model_type,
            device=str(device) # Pass device as string
        )
        if object_mask_224 is None:
            logging.warning("Failed to obtain object mask for modularity via integrated DINO/SAM.")
    elif prompt_for_dino:
        logging.info("Prompt for DINO available, but DINO/SAM libraries or model paths not configured. Skipping mask generation.")
    else:
        logging.info("No prompt for DINO. Skipping mask generation and modularity.")
        
    results["modularity"] = calculate_all_layers_modularity(edge_indexes_per_layer, object_mask_224)
    
    if final_logits is not None:
        predicted_class_idx = torch.argmax(final_logits, dim=1).item()
        results["predicted_class_idx_final"] = predicted_class_idx
        if CLASS_IDX_TO_NAME_MAPPING and predicted_class_idx in CLASS_IDX_TO_NAME_MAPPING:
            # Handle potential tuple/list in mapping
            pred_name_raw = CLASS_IDX_TO_NAME_MAPPING[predicted_class_idx]
            if isinstance(pred_name_raw, (list, tuple)):
                results["predicted_class_name_final"] = pred_name_raw[-1]
            else:
                results["predicted_class_name_final"] = pred_name_raw

        logging.info(f"Model final prediction index: {predicted_class_idx}, Name: {results.get('predicted_class_name_final', 'N/A')}")

    # Convert NumPy arrays to lists for JSON serialization
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results[key] = value.tolist()
        elif isinstance(value, list):
            results[key] = [item.tolist() if isinstance(item, np.ndarray) else item for item in value]

    output_filename_base = Path(args.im_path).stem
    output_file_path = Path(args.out_dir) / f"metrics_{output_filename_base}.json"
    try:
        with open(output_file_path, "w") as f:
            json.dump(results, f, indent=4)
        logging.info(f"Successfully saved metrics to: {output_file_path}")
    except Exception as e:
        logging.error(f"Failed to save results to {output_file_path}: {e}")

    logging.info("Analysis finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Vision GNN Explanation Analysis for a single image with integrated DINO/SAM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Image and Model ---
    parser.add_argument('--im_path', type=str, required=True, help='Path to the input image file.')
    parser.add_argument('--model_weights', type=str, required=True, help='Path to the pre-trained ViG model weights (.pth file).')
    parser.add_argument('--model_variant', type=str, default='vig_s_224_gelu', help='Name of the ViG model variant function in vig_2.py.')
    parser.add_argument('--out_dir', type=str, default='./outputs', help='Directory to save the analysis outputs.')
    parser.add_argument('--device', type=str, default='auto', choices=['cuda', 'cpu', 'auto'], help="Device to use.")

    # --- Ground Truth and Prompting ---
    parser.add_argument('--gt_label_idx', type=int, default=None, help='(Optional) Ground truth ImageNet class index (0-999).')
    parser.add_argument('--gt_label_name', type=str, default=None, help='(Optional) Ground truth object name for DINO prompt (e.g., "dog").')
    parser.add_argument('--cls_to_labels_path', type=str, default='imagenet1000_clsidx_to_labels.txt', help='(Optional) Path to imagenet1000_clsidx_to_labels.txt file.')

    # --- DINO and SAM Configuration (Required if using modularity with gt_label_name or derived name) ---
    parser.add_argument('--dino_config', type=str, default=None, help='Path to GroundingDINO config file (e.g., GroundingDINO_SwinT_OGC.py).')
    parser.add_argument('--dino_weights', type=str, default=None, help='Path to GroundingDINO weights file (e.g., groundingdino_swint_ogc.pth).')
    parser.add_argument('--sam_checkpoint', type=str, default=None, help='Path to SAM checkpoint file (e.g., sam_vit_h_4b8939.pth).')
    parser.add_argument('--sam_model_type', type=str, default='vit_h', choices=['vit_h', 'vit_l', 'vit_b'], help='SAM model type.')
    
    args = parser.parse_args()
    main(args)
