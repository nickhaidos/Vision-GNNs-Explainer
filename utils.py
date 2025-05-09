import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
import networkx as nx
import logging
import cv2
import vig_2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# --- Optional Imports for GroundingDINO and SAM ---
try:
    from groundingdino.util.inference import load_model as dino_load_model, load_image as dino_load_image, predict as dino_predict
    from segment_anything import SamPredictor, sam_model_registry 
    DINO_SAM_AVAILABLE = True
    logging.info("Successfully imported GroundingDINO and SAM libraries.")
except ImportError as e:
    DINO_SAM_AVAILABLE = False
    logging.warning(f"GroundingDINO or SAM libraries not found. Mask generation will be unavailable. Error: {e}")


# --- Image Preprocessing ---
def image_to_tensor(image_path, device):
    """
    Loads an image, applies transformations, and converts it to a tensor.
    """
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    try:
        image = Image.open(image_path).convert('RGB')
        tensor = transform(image).unsqueeze(0)
        return tensor.to(device)
    except FileNotFoundError:
        logging.error(f"Image file not found: {image_path}")
        raise
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        raise

# --- Vision-GNN Inference ---
def run_model_inference(model, image_tensor):
    """
    Runs the image tensor through the model and returns intermediate outputs.
    """
    model.eval()
    with torch.no_grad():
        final_logits, edge_indexes, _, block_features = model(image_tensor)
    return final_logits, edge_indexes, block_features

# --- Integrated DINO + SAM Mask Generation ---
def generate_segmentation_mask(
    image_path_str: str,
    text_prompt: str,
    dino_config_path: str,
    dino_weights_path: str,
    sam_checkpoint_path: str,
    sam_model_type: str = "vit_h",
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    device: str = "cuda"
):
    """
    Generates a segmentation mask using GroundingDINO and SAM.
    Returns a 224x224 boolean numpy array or None if failed.
    """
    if not DINO_SAM_AVAILABLE:
        logging.warning("GroundingDINO/SAM libraries not available. Cannot generate mask.")
        return None

    logging.info(f"Attempting to generate mask for prompt: '{text_prompt}' on image: {image_path_str}")

    try:
        # 1. Load GroundingDINO model
        dino_model = dino_load_model(dino_config_path, dino_weights_path, device=device)
        logging.info("GroundingDINO model loaded.")

        # 2. Load image for DINO
        image_source_rgb, image_for_dino = dino_load_image(image_path_str)
        logging.info("Image loaded for GroundingDINO.")

        # 3. GroundingDINO Inference
        boxes, logits, phrases = dino_predict(
            model=dino_model,
            image=image_for_dino,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device
        )
        logging.info(f"GroundingDINO found {len(boxes)} boxes for prompt '{text_prompt}'.")

        if boxes.nelement() == 0: # No boxes found
            logging.warning(f"No boxes detected by GroundingDINO for prompt: '{text_prompt}'")
            return None

        # 4. Set up SAM model
        sam_model = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_path)
        sam_model.to(device=device)
        sam_predictor = SamPredictor(sam_model)
        logging.info("SAM model loaded.")

        # 5. SAM Inference for Masks
        # image_source_rgb is a NumPy array (H, W, C) in RGB format
        sam_predictor.set_image(image_source_rgb)
        logging.info("Image set for SAM predictor.")

        # Get image dimensions for mask combination
        h_img, w_img, _ = image_source_rgb.shape # Original image dimensions
        combined_mask_original_size = np.zeros((h_img, w_img), dtype=bool)

        logging.info(f"Processing {len(boxes)} boxes with SAM...")
        for i, box_normalized in enumerate(boxes):
            # GroundingDINO boxes are [cx, cy, w, h] in normalized coordinates.
            # SAM expects [x1, y1, x2, y2] in absolute image coordinates.
            cx, cy, w, h = box_normalized.tolist()
            x1_abs = (cx - w / 2) * w_img
            y1_abs = (cy - h / 2) * h_img
            x2_abs = (cx + w / 2) * w_img
            y2_abs = (cy + h / 2) * h_img
            sam_input_box = np.array([x1_abs, y1_abs, x2_abs, y2_abs])
            
            logging.debug(f"Processing box {i+1}/{len(boxes)} with SAM: {sam_input_box}")

            masks_sam, _, _ = sam_predictor.predict(
                box=sam_input_box,
                multimask_output=False # Get a single mask for the box
            )
            
            # masks_sam is (1, H_orig, W_orig) boolean numpy array
            if masks_sam is not None and masks_sam.size > 0:
                current_mask_original_size = masks_sam[0] # Get (H_orig, W_orig)
                combined_mask_original_size = np.logical_or(combined_mask_original_size, current_mask_original_size)
            else:
                logging.warning(f"SAM did not return a mask for box {i+1}.")
        
        if not np.any(combined_mask_original_size): # Check if any part of the combined mask is True
            logging.warning("SAM did not return any valid masks for the detected boxes.")
            return None

        final_mask_original_size = combined_mask_original_size
        
        # 6. Resize mask to 224x224
        # Convert boolean to uint8 for resizing
        mask_uint8 = final_mask_original_size.astype(np.uint8) * 255
        resized_mask_uint8 = cv2.resize(mask_uint8, (224, 224), interpolation=cv2.INTER_NEAREST)
        
        # Convert back to boolean
        resized_mask_boolean = resized_mask_uint8.astype(bool)
        
        logging.info("Successfully generated and resized segmentation mask.")
        return resized_mask_boolean

    except Exception as e:
        logging.error(f"Error during integrated DINO-SAM mask generation: {e}", exc_info=True)
        return None


# --- Metric Calculation Functions (adapted from your script) ---

def calculate_visual_similarity(edge_indexes_per_layer, image_tensor_normalized):
    """
    Calculates average visual similarity (cosine similarity of raw patch pixels)
    between connected patches for each layer.
    """
    logging.info("Calculating Visual Similarity...")    
    patch_size = 16 
    num_patches_dim = 14 
    
    similarities_all_layers = []
    if not isinstance(edge_indexes_per_layer, list) or not edge_indexes_per_layer:
        logging.warning("edge_indexes_per_layer is not a list or is empty. Skipping visual similarity.")
        return [0.0] * 16 

    patches_data = []
    _, C, H, W = image_tensor_normalized.shape
    for i in range(num_patches_dim * num_patches_dim):
        row, col = i // num_patches_dim, i % num_patches_dim
        patch = image_tensor_normalized[0, :, row * patch_size:(row + 1) * patch_size, col * patch_size:(col + 1) * patch_size]
        patches_data.append(patch.flatten())

    for layer_idx, layer_edge_index_data in enumerate(edge_indexes_per_layer):
        if layer_edge_index_data is None or layer_edge_index_data.numel() == 0:
            logging.warning(f"Edge index data for layer {layer_idx} is empty. Appending 0 similarity.")
            similarities_all_layers.append(0.0)
            continue

        current_layer_similarities = []
        num_nodes = layer_edge_index_data.shape[2] 
        num_neighbors = layer_edge_index_data.shape[3]

        for node_i in range(num_nodes):
            patch1_flat = patches_data[node_i]
            for k_neighbor in range(1, num_neighbors): 
                node_j_idx = layer_edge_index_data[0, 0, node_i, k_neighbor].item()
                if 0 <= node_j_idx < len(patches_data):
                    patch2_flat = patches_data[node_j_idx]
                    sim = F.cosine_similarity(patch1_flat, patch2_flat, dim=0).item()
                    current_layer_similarities.append(sim)
                else:
                    logging.warning(f"Node index {node_j_idx} out of bounds for patches_data (len {len(patches_data)}) in layer {layer_idx}.")

        if current_layer_similarities:
            similarities_all_layers.append(np.mean(current_layer_similarities))
        else:
            similarities_all_layers.append(0.0) 
            logging.warning(f"No valid connections found for visual similarity in layer {layer_idx}.")
    logging.info(f"Visual Similarities: {similarities_all_layers}")
    return similarities_all_layers


def calculate_spatial_distance(edge_indexes_per_layer):
    """
    Calculates average Manhattan distance between connected patches for each layer.
    """
    logging.info("Calculating Spatial Distance...")
    distances_all_layers = []
    num_patches_dim = 14 
    if not isinstance(edge_indexes_per_layer, list) or not edge_indexes_per_layer:
        logging.warning("edge_indexes_per_layer is not a list or is empty. Skipping spatial distance.")
        return [0.0] * 16

    for layer_idx, layer_edge_index_data in enumerate(edge_indexes_per_layer):
        if layer_edge_index_data is None or layer_edge_index_data.numel() == 0:
            logging.warning(f"Edge index data for layer {layer_idx} is empty. Appending 0 distance.")
            distances_all_layers.append(0.0)
            continue

        current_layer_distances = []
        num_nodes = layer_edge_index_data.shape[2]
        num_neighbors = layer_edge_index_data.shape[3]

        for node_i in range(num_nodes):
            row_i, col_i = node_i // num_patches_dim, node_i % num_patches_dim
            for k_neighbor in range(1, num_neighbors): 
                node_j_idx = layer_edge_index_data[0, 0, node_i, k_neighbor].item()
                row_j, col_j = node_j_idx // num_patches_dim, node_j_idx % num_patches_dim
                dist = abs(row_i - row_j) + abs(col_i - col_j)
                current_layer_distances.append(dist)
        
        if current_layer_distances:
            distances_all_layers.append(np.mean(current_layer_distances))
        else:
            distances_all_layers.append(0.0)
            logging.warning(f"No valid connections found for spatial distance in layer {layer_idx}.")
    logging.info(f"Spatial Distances: {distances_all_layers}")
    return distances_all_layers

def calculate_embedding_similarity(edge_indexes_per_layer, block_features_per_layer):
    """
    Calculates average cosine similarity between embeddings of connected patches for each layer.
    """
    logging.info("Calculating Embedding Similarity...")
    similarities_all_layers = []
    if not isinstance(edge_indexes_per_layer, list) or not edge_indexes_per_layer or \
       not isinstance(block_features_per_layer, list) or not block_features_per_layer or \
       len(edge_indexes_per_layer) != len(block_features_per_layer):
        logging.warning("Input data for embedding similarity is invalid. Skipping.")
        return [0.0] * 16

    for layer_idx in range(len(edge_indexes_per_layer)):
        layer_edge_index_data = edge_indexes_per_layer[layer_idx]
        layer_block_features = block_features_per_layer[layer_idx] 

        if layer_edge_index_data is None or layer_edge_index_data.numel() == 0 or \
           layer_block_features is None or layer_block_features.numel() == 0:
            logging.warning(f"Data for layer {layer_idx} is empty. Appending 0 embedding similarity.")
            similarities_all_layers.append(0.0)
            continue
        
        if layer_block_features.ndim == 4:
            batch_size, D_emb, H_feat, W_feat = layer_block_features.shape
            layer_block_features_reshaped = layer_block_features.permute(0, 2, 3, 1).reshape(batch_size, -1, D_emb)
        else: 
            layer_block_features_reshaped = layer_block_features
        
        node_embeddings = layer_block_features_reshaped[0] 

        current_layer_similarities = []
        num_nodes = layer_edge_index_data.shape[2]
        num_neighbors = layer_edge_index_data.shape[3]

        for node_i in range(num_nodes):
            emb1 = node_embeddings[node_i]
            for k_neighbor in range(1, num_neighbors): 
                node_j_idx = layer_edge_index_data[0, 0, node_i, k_neighbor].item()
                if 0 <= node_j_idx < num_nodes:
                    emb2 = node_embeddings[node_j_idx]
                    sim = F.cosine_similarity(emb1, emb2, dim=0).item()
                    current_layer_similarities.append(sim)
                else:
                    logging.warning(f"Node index {node_j_idx} out of bounds for embeddings (num_nodes {num_nodes}) in layer {layer_idx}.")

        if current_layer_similarities:
            similarities_all_layers.append(np.mean(current_layer_similarities))
        else:
            similarities_all_layers.append(0.0)
            logging.warning(f"No valid connections found for embedding similarity in layer {layer_idx}.")
    logging.info(f"Embedding Similarities: {similarities_all_layers}")
    return similarities_all_layers


def calculate_layer_probabilities(model, block_features_per_layer, target_class_idx):
    """
    Calculates the probability of the target_class_idx for each layer's features.
    """
    if target_class_idx is None:
        logging.info("Target class index not provided. Skipping layer probability calculation.")
        return None, None
    
    logging.info(f"Calculating Layer Probabilities for target index: {target_class_idx}...")
    probabilities = []
    is_correct_list = []

    if not isinstance(block_features_per_layer, list) or not block_features_per_layer:
        logging.warning("block_features_per_layer is not a list or is empty. Skipping probability calculation.")
        return [0.0] * 16, [False] * 16

    model.eval() 
    with torch.no_grad():
        for layer_idx, layer_features in enumerate(block_features_per_layer):
            
            if layer_features is None or layer_features.numel() == 0:
                logging.warning(f"Features for layer {layer_idx} are empty. Appending 0 probability.")
                probabilities.append(0.0)
                is_correct_list.append(False)
                continue

            layer_logits = model.prediction(F.adaptive_avg_pool2d(layer_features, 1)).squeeze(-1).squeeze(-1)
            
            layer_probs_softmax = F.softmax(layer_logits, dim=1).squeeze() 
            if target_class_idx < len(layer_probs_softmax):
                 probabilities.append(layer_probs_softmax[target_class_idx].item())
            else:
                logging.warning(f"Target class index {target_class_idx} out of bounds for layer {layer_idx} logits ({len(layer_probs_softmax)} classes). Appending 0.")
                probabilities.append(0.0)

            predicted_class_idx = torch.argmax(layer_logits, dim=1).item()
            is_correct_list.append(predicted_class_idx == target_class_idx)
            
    logging.info(f"Layer Probabilities (for target): {probabilities}")
    logging.info(f"Layer Is Correct (for target): {is_correct_list}")
    return probabilities, is_correct_list


def find_object_patches_from_mask(object_mask_224):
    """
    Identifies which 16x16 patches overlap with the 224x224 object mask.
    """
    if object_mask_224 is None:
        return []
    
    object_patch_indices = []
    patch_size = 16
    num_patches_dim = 14 

    for i in range(num_patches_dim * num_patches_dim): 
        row, col = i // num_patches_dim, i % num_patches_dim
        patch_mask_region = object_mask_224[row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size]
        if np.sum(patch_mask_region) > 0: 
            object_patch_indices.append(i)
    logging.info(f"Found {len(object_patch_indices)} object patches from mask.")
    return object_patch_indices

def calculate_modularity_for_layer(layer_edge_index_data, object_patch_indices):
    """
    Calculates graph modularity for a single layer.
    """
    if not object_patch_indices: 
        logging.warning("No object patches provided, cannot calculate modularity.")
        return 0.0 

    G = nx.Graph()
    num_nodes = layer_edge_index_data.shape[2]
    num_neighbors = layer_edge_index_data.shape[3]
    
    edges = []
    for node_i in range(num_nodes):
        G.add_node(node_i) 
        for k_neighbor in range(num_neighbors): 
            node_j_idx = layer_edge_index_data[0, 0, node_i, k_neighbor].item()
            if node_i != node_j_idx : 
                 if 0 <= node_j_idx < num_nodes: 
                    edges.append((node_i, node_j_idx))
    
    G.add_edges_from(edges)

    communities = [set(), set()] 
    all_patches_set = set(range(num_nodes))
    object_patches_set = set(object_patch_indices)

    communities[1] = object_patches_set
    communities[0] = all_patches_set - object_patches_set
    
    communities = [c for c in communities if c]
    if len(communities) < 2: 
        logging.warning("Less than two communities found, modularity is ill-defined. Returning 0.")
        return 0.0

    try:
        modularity_score = nx.community.modularity(G, communities)
    except Exception as e: 
        logging.error(f"Error calculating modularity with NetworkX: {e}")
        modularity_score = 0.0 
    return modularity_score


def calculate_all_layers_modularity(edge_indexes_per_layer, object_mask_224):
    """
    Calculates modularity for all layers.
    """
    if object_mask_224 is None:
        logging.info("Object mask not available. Skipping modularity calculation.")
        return None # Return None if mask is not available

    logging.info("Calculating Graph Modularity for all layers...")
    object_patch_indices = find_object_patches_from_mask(object_mask_224)
    if not object_patch_indices:
        logging.warning("No object patches identified from the mask. Modularity will be 0 for all layers.")
        # Return a list of zeros if no object patches, matching expected output length
        return [0.0] * (len(edge_indexes_per_layer) if isinstance(edge_indexes_per_layer, list) else 16)


    modularity_scores_all_layers = []
    if not isinstance(edge_indexes_per_layer, list) or not edge_indexes_per_layer:
        logging.warning("edge_indexes_per_layer is not a list or is empty. Skipping modularity.")
        return [0.0] * 16


    for layer_idx, layer_edge_index_data in enumerate(edge_indexes_per_layer):
        if layer_edge_index_data is None or layer_edge_index_data.numel() == 0:
            logging.warning(f"Edge index data for layer {layer_idx} is empty. Appending 0 modularity.")
            modularity_scores_all_layers.append(0.0)
            continue
        score = calculate_modularity_for_layer(layer_edge_index_data, object_patch_indices)
        modularity_scores_all_layers.append(score)
    
    logging.info(f"Graph Modularities: {modularity_scores_all_layers}")
    return modularity_scores_all_layers


def load_model_and_dict(model_weights_path, model_variant='vig_s_224_gelu', cls_to_labels_path=None, device='auto'):
    if device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    cls_idx_to_name = {}
    if cls_to_labels_path:
        try:
            with open(cls_to_labels_path, 'r') as f:
                cls_idx_to_name = eval(f.read()) 
        except Exception as e:
            logging.error(f"Could not load class mapping from {cls_to_labels_path}: {e}")

    try:
        model_constructor = getattr(vig_2, model_variant)
        model = model_constructor().to(device)
    except AttributeError:
        logging.error(f"Model variant '{model_variant}' not found in vig_2.py.")
        return
    except Exception as e:
        logging.error(f"Error initializing model {model_variant}: {e}")
        return

    try:
        model.load_state_dict(torch.load(model_weights_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        logging.error(f"Model weights file not found: {model_weights_path}")
        return
    except Exception as e:
        logging.error(f"Error loading model weights: {e}")
        return
    
    return model, cls_idx_to_name


def normalize_similarities(sims):
    sim_values = list(sims.values())
    sim_min = min(sim_values)
    sim_max = max(sim_values)
    # Scale to [0,1] first, then scale to [0.1,1.0]
    return {k: 0.4 + 0.6 * ((v - sim_min) / (sim_max - sim_min)) for k, v in sims.items()}


def visualize_gradient_incoming_edges(model, image_path, layer, patch_coords, imagenet_dict=None, figsize=(10,10)):
    image_tensor = image_to_tensor(image_path, next(model.parameters()).device)
    temp_out, edge_indexes, block_features = run_model_inference(model, image_tensor)
    answer = torch.argmax(temp_out, dim=1).item()
    if imagenet_dict is not None:
        print("Predicted class: ", imagenet_dict[answer])
    
    edge_index = edge_indexes[layer]

    i, j = patch_coords
    patch_idx = i * 14 + j  # Convert 2D coordinates to flattened index

    # Find incoming and outgoing connections
    incoming_indices = edge_index[0, 0, patch_idx].cpu().numpy()

    # Create masks for different types of connections
    source_mask = np.zeros((14, 14), dtype=bool)
    source_mask[i, j] = True

    incoming_mask = np.zeros((14, 14), dtype=bool)

    # Mark incoming and outgoing connections
    for idx in incoming_indices:
        if idx == patch_idx:
            continue
        incoming_mask[idx // 14, idx % 14] = True

    current_sims = {}
    block_features[layer].shape

    for idx in incoming_indices:
        if idx == patch_idx:
            continue
        patch1 = block_features[layer][0, :, idx // 14, idx % 14]
        patch2 = block_features[layer][0, :, patch_idx // 14, patch_idx % 14]
        current_sims[(idx//14, idx%14)] = torch.nn.functional.cosine_similarity(patch1, patch2, dim=0).item()

    normalized_sims = normalize_similarities(current_sims)

    # Upscale masks to 224x224
    source_mask = np.kron(source_mask, np.ones((16,16), dtype=bool))
    incoming_mask = np.kron(incoming_mask, np.ones((16,16), dtype=bool))

    print(f"Patch ({i},{j}) (green) [index {patch_idx}]: {len(normalized_sims)} incoming (red, brighter have more similar embeddings)")

    plt.figure(figsize=figsize)

    # Load and resize image
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0

    plt.imshow(img_array)

    # Create colored overlays (RGBA)
    green_overlay = np.zeros((224, 224, 4))
    red_overlay = np.zeros((224, 224, 4))

    green_overlay[source_mask] = [0, 1, 0, 0.6]      # Green for source

    # Variable red intensity for incoming connections
    for (i, j), sim in normalized_sims.items():
        # Create mask for this specific patch
        patch_mask = np.zeros((14, 14), dtype=bool)
        patch_mask[i, j] = True
        patch_mask = np.kron(patch_mask, np.ones((16,16), dtype=bool))
        
        # Alpha could also vary with similarity if desired
        red_overlay[patch_mask] = [1, 0, 0, sim * 0.6]

    plt.imshow(green_overlay)
    plt.imshow(red_overlay)

    plt.axis('off')
    plt.show()