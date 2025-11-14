"""
HCP-DETR Utility Functions for Hierarchical Category Prototype Learning.

This module provides critical functions for:
1. Hungarian matching between predictions and ground truth
2. Subcategory label mapping for comprehensive prototype training

Author: Ultralytics Team
Date: 2025-11-14
"""

from typing import Tuple, List, Optional
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def hungarian_match_hcp_detr(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    query_embed: torch.Tensor,
    cost_class: float = 2.0,
    cost_bbox: float = 5.0,
    cost_giou: float = 2.0,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Perform Hungarian matching between predictions and ground truth for HCP-DETR.

    This function correctly aligns predicted features with ground truth labels,
    which is CRITICAL for prototype learning. Without matching, features and labels
    are randomly paired, causing the network to learn meaningless patterns.

    Args:
        pred_boxes (torch.Tensor): Predicted bounding boxes [bs, nq, 4] in format (cx, cy, w, h)
        pred_scores (torch.Tensor): Predicted class scores [bs, nq, total_nc]
        gt_boxes (torch.Tensor): Ground truth bounding boxes [bs, max_gt, 4]
        gt_labels (torch.Tensor): Ground truth labels [bs, max_gt], -1 for padding
        query_embed (torch.Tensor): Query embeddings from decoder [bs, nq, hidden_dim]
        cost_class (float): Weight for classification cost (default: 2.0)
        cost_bbox (float): Weight for L1 box distance cost (default: 5.0)
        cost_giou (float): Weight for GIoU cost (default: 2.0)

    Returns:
        matched_features (torch.Tensor): Matched query features [N_matched, hidden_dim]
        matched_labels (torch.Tensor): Matched ground truth labels [N_matched]

        Returns (None, None) if no valid matches found

    Example:
        >>> pred_boxes = torch.rand(2, 300, 4)
        >>> pred_scores = torch.rand(2, 300, 6)
        >>> gt_boxes = torch.rand(2, 10, 4)
        >>> gt_labels = torch.randint(0, 2, (2, 10))
        >>> query_embed = torch.rand(2, 300, 256)
        >>> features, labels = hungarian_match_hcp_detr(
        ...     pred_boxes, pred_scores, gt_boxes, gt_labels, query_embed
        ... )
    """
    try:
        from torchvision.ops import box_convert, generalized_box_iou
    except ImportError:
        raise ImportError(
            "torchvision is required for Hungarian matching. "
            "Install it with: pip install torchvision"
        )

    bs, nq, _ = pred_boxes.shape
    device = pred_boxes.device

    matched_features_list = []
    matched_labels_list = []

    for b in range(bs):
        # Get valid GT (filter out padding with label = -1)
        valid_mask = gt_labels[b] >= 0
        num_gt = valid_mask.sum().item()

        if num_gt == 0:
            # No ground truth objects in this image
            continue

        valid_gt_boxes = gt_boxes[b, valid_mask]  # [num_gt, 4]
        valid_gt_labels = gt_labels[b, valid_mask]  # [num_gt]

        # === Cost Matrix Computation ===
        # Following DETR paper: C = λ_class * C_class + λ_bbox * C_bbox + λ_giou * C_giou

        # 1. Classification Cost
        # Negative log-likelihood of predicted class matching ground truth
        pred_probs = F.softmax(pred_scores[b], dim=-1)  # [nq, total_nc]

        # Gather probabilities for ground truth classes
        # Create index tensor [nq, num_gt] where each column is the GT class index
        gt_class_indices = valid_gt_labels.unsqueeze(0).expand(nq, -1).long()  # [nq, num_gt]

        # Gather: for each query, get probability of each GT class
        cls_cost = -pred_probs.gather(dim=1, index=gt_class_indices)  # [nq, num_gt]

        # 2. Bounding Box L1 Cost
        # L1 distance between predicted and GT boxes
        # Expand for broadcasting: [nq, 1, 4] vs [1, num_gt, 4]
        bbox_cost = torch.cdist(
            pred_boxes[b],  # [nq, 4]
            valid_gt_boxes,  # [num_gt, 4]
            p=1  # L1 distance
        )  # [nq, num_gt]

        # 3. GIoU Cost
        # Generalized IoU is better for object detection than standard IoU
        # Convert from (cx, cy, w, h) to (x1, y1, x2, y2) for IoU computation
        pred_boxes_xyxy = box_convert(pred_boxes[b], in_fmt='cxcywh', out_fmt='xyxy')  # [nq, 4]
        gt_boxes_xyxy = box_convert(valid_gt_boxes, in_fmt='cxcywh', out_fmt='xyxy')  # [num_gt, 4]

        giou = generalized_box_iou(pred_boxes_xyxy, gt_boxes_xyxy)  # [nq, num_gt]
        giou_cost = -giou  # Negative because we want to minimize cost

        # Total Cost Matrix
        C = cost_class * cls_cost + cost_bbox * bbox_cost + cost_giou * giou_cost  # [nq, num_gt]

        # === Hungarian Algorithm ===
        # Find optimal assignment minimizing total cost
        C_np = C.detach().cpu().numpy()
        pred_idx, gt_idx = linear_sum_assignment(C_np)

        # Convert to tensors
        pred_idx = torch.tensor(pred_idx, dtype=torch.long, device=device)
        gt_idx = torch.tensor(gt_idx, dtype=torch.long, device=device)

        # Extract matched features and labels
        matched_features_list.append(query_embed[b, pred_idx])  # [num_matched, hidden_dim]
        matched_labels_list.append(valid_gt_labels[gt_idx])  # [num_matched]

    # Concatenate all batches
    if len(matched_features_list) == 0:
        return None, None

    matched_features = torch.cat(matched_features_list, dim=0)  # [N_matched, hidden_dim]
    matched_labels = torch.cat(matched_labels_list, dim=0)  # [N_matched]

    return matched_features, matched_labels


def map_to_subcategories_hcp_detr(
    labels: torch.Tensor,
    sub_categories: dict,
    strategy: str = 'uniform',
) -> torch.Tensor:
    """
    Map main category labels to subcategory labels for HCP-DETR.

    This function ensures ALL prototypes receive gradients during training.
    Without this, subcategory prototypes (e.g., young_fruit, flower, occluded, malformed)
    remain untrained (Xavier initialized), wasting 67% of the prototype capacity!

    Args:
        labels (torch.Tensor): Main category labels [N], e.g., [0, 1, 1, 0, 1]
                              where 0=harvestable, 1=no_harvestable
        sub_categories (dict): Mapping from main category to subcategory names
                              Example: {1: ['young_fruit', 'flower', 'occluded', 'malformed']}
        strategy (str): Subcategory assignment strategy
                       - 'uniform': Distribute evenly across subcategories
                       - 'random': Randomly assign to subcategories
                       Default: 'uniform'

    Returns:
        mapped_labels (torch.Tensor): Subcategory labels [N]
                                     Example: [0, 2, 3, 0, 4]

    Example:
        >>> labels = torch.tensor([0, 1, 1, 0, 1])  # Main categories
        >>> sub_categories = {1: ['young_fruit', 'flower', 'occluded', 'malformed']}
        >>> mapped = map_to_subcategories_hcp_detr(labels, sub_categories, strategy='uniform')
        >>> print(mapped)
        tensor([0, 2, 3, 0, 4])  # Now uses subcategories 2, 3, 4 instead of just 1

    Notes:
        - Main categories without subcategories keep their original labels
        - Subcategory indices are computed as: nc + cumulative_sub_count + sub_index
        - For cucumber dataset:
            * Label 0 (harvestable): No subcategories, remains 0
            * Label 1 (no_harvestable): Maps to [2, 3, 4, 5]
              - 2: young_fruit
              - 3: flower
              - 4: occluded
              - 5: malformed
    """
    if not sub_categories:
        # No subcategories defined, return original labels
        return labels

    device = labels.device
    mapped_labels = labels.clone()

    # Compute number of main categories (max main category + 1)
    num_main_categories = max(sub_categories.keys()) + 1

    for main_cls, sub_list in sub_categories.items():
        # Find samples belonging to this main category
        mask = (labels == main_cls)
        num_samples = mask.sum().item()

        if num_samples == 0:
            continue

        # Compute subcategory index offset
        # Subcategories start after all main categories
        # Example: nc=2 (harvestable, no_harvestable), main_cls=1
        #          subcategories for class 1 start at index 2
        sub_start_idx = num_main_categories

        # Add offset for subcategories of previous main classes
        for prev_cls in sorted(sub_categories.keys()):
            if prev_cls < main_cls:
                sub_start_idx += len(sub_categories[prev_cls])

        num_subcategories = len(sub_list)

        # Assign subcategory indices based on strategy
        if strategy == 'uniform':
            # Distribute evenly: [0, 1, 2, 3, 0, 1, ...] cycling through subcategories
            indices = torch.arange(num_samples, device=device) % num_subcategories
            sub_indices = sub_start_idx + indices

        elif strategy == 'random':
            # Random assignment
            sub_indices = torch.randint(
                sub_start_idx,
                sub_start_idx + num_subcategories,
                (num_samples,),
                device=device
            )
        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Use 'uniform' or 'random'."
            )

        # Apply mapping
        mapped_labels[mask] = sub_indices

    return mapped_labels


def validate_hcp_detr_config(
    num_classes: int,
    sub_categories: dict,
    total_prototypes: int,
) -> bool:
    """
    Validate HCP-DETR configuration for correctness.

    Args:
        num_classes (int): Number of main categories
        sub_categories (dict): Subcategory mapping
        total_prototypes (int): Total number of prototypes

    Returns:
        bool: True if configuration is valid

    Raises:
        ValueError: If configuration is invalid with detailed error message

    Example:
        >>> validate_hcp_detr_config(
        ...     num_classes=2,
        ...     sub_categories={1: ['young_fruit', 'flower', 'occluded', 'malformed']},
        ...     total_prototypes=6
        ... )
        True
    """
    # Count expected total prototypes
    expected_prototypes = num_classes
    for main_cls, sub_list in sub_categories.items():
        expected_prototypes += len(sub_list)

    if total_prototypes != expected_prototypes:
        raise ValueError(
            f"Prototype count mismatch!\n"
            f"  num_classes: {num_classes}\n"
            f"  subcategories: {sub_categories}\n"
            f"  expected_prototypes: {expected_prototypes}\n"
            f"  actual total_prototypes: {total_prototypes}\n"
            f"  → Set total_nc={expected_prototypes} in model config"
        )

    # Validate subcategory keys are within num_classes range
    for main_cls in sub_categories.keys():
        if main_cls >= num_classes:
            raise ValueError(
                f"Invalid subcategory key {main_cls} (must be < num_classes={num_classes})"
            )

    return True


# Example usage and testing
if __name__ == '__main__':
    print("HCP-DETR Utility Functions")
    print("=" * 60)

    # Test Hungarian matching
    print("\n1. Testing Hungarian Matching...")
    pred_boxes = torch.rand(2, 300, 4)
    pred_scores = torch.rand(2, 300, 6)
    gt_boxes = torch.rand(2, 10, 4)
    gt_labels = torch.randint(0, 2, (2, 10))
    query_embed = torch.rand(2, 300, 256)

    features, labels = hungarian_match_hcp_detr(
        pred_boxes, pred_scores, gt_boxes, gt_labels, query_embed
    )

    if features is not None:
        print(f"  ✅ Matched {features.shape[0]} predictions to ground truth")
        print(f"  Features shape: {features.shape}")
        print(f"  Labels shape: {labels.shape}")
    else:
        print("  ⚠️ No matches found (possibly no GT in batch)")

    # Test subcategory mapping
    print("\n2. Testing Subcategory Mapping...")
    test_labels = torch.tensor([0, 1, 1, 0, 1, 1, 0, 1])
    sub_categories = {1: ['young_fruit', 'flower', 'occluded', 'malformed']}

    mapped = map_to_subcategories_hcp_detr(test_labels, sub_categories, strategy='uniform')
    print(f"  Original labels: {test_labels.tolist()}")
    print(f"  Mapped labels:   {mapped.tolist()}")

    # Count usage of each subcategory
    unique, counts = torch.unique(mapped, return_counts=True)
    print(f"  Label distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    # Test configuration validation
    print("\n3. Testing Configuration Validation...")
    try:
        validate_hcp_detr_config(
            num_classes=2,
            sub_categories={1: ['young_fruit', 'flower', 'occluded', 'malformed']},
            total_prototypes=6
        )
        print("  ✅ Configuration is valid")
    except ValueError as e:
        print(f"  ❌ Configuration error: {e}")

    print("\n" + "=" * 60)
    print("All tests completed!")
