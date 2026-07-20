import numpy as np
import torch

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio

def compute_auc(pred_mask, true_mask):
    """
    pred_mask: (B, 3, H, W) - soft predictions replicated over channels
    true_mask: (B, 3, H, W) - binary labels (0 or 1), replicated
    Returns: scalar AUC
    """
    # 채널 하나만 사용 (복제된 거니까)
    pred = pred_mask[:, 0].flatten().detach().cpu().numpy()  # (B*H*W,)
    true = true_mask[:, 0].flatten().detach().cpu().numpy()

    
    auc = roc_auc_score(true, pred)
    return auc

def compute_binary_f1(pred_mask, true_mask, eps=1e-6):
    """
    pred_mask: (B, 3, H, W) - 3 identical binary masks
    true_mask: (B, 3, H, W) - 3 identical binary masks
    returns: scalar F1 score
    """
    pred = pred_mask[:, 0]  # shape: (B, H, W)
    true = true_mask[:, 0]

    intersection = (pred.bool() & true.bool()).sum().float()
    total = pred.sum().float() + true.sum().float()

    f1 = 2 * intersection / (total + eps)
    return f1

def compute_binary_iou(pred_mask, true_mask, eps=1e-6):
    """
    pred_mask: (B, 3, H, W) - 3 identical binary masks
    true_mask: (B, 3, H, W) - 3 identical binary masks
    returns: scalar IoU
    """
    # 채널 하나만 사용
    pred = pred_mask[:, 0]  # shape: (B, H, W)
    true = true_mask[:, 0]

    intersection = (pred.bool() & true.bool()).sum().float()
    union = (pred.bool() | true.bool()).sum().float()       

    iou = intersection / (union + eps)
    return iou

def calculate_masked_psnr(img1, img2, mask):
    """
    img1, img2: Tensor of shape (C, H, W) and values in [0, 1]
    mask: Tensor of shape (1, H, W) or (H, W) with 1 for valid regions, 0 for ignored
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)  # (1, H, W)

    diff = (img1 - img2) ** 2


    mse = (diff * mask).sum() / (mask.sum() + 1e-8 )

    if mse == 0:
        return 36

    psnr = 10 * torch.log10(1.0 / mse)
    return psnr.item()

def calculate_snr(signal, noise):
    # 신호의 파워 계산 (평균 제곱 값)
    signal_power = torch.mean(signal ** 2)
    noise_power = torch.mean((signal-noise) ** 2)
    
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr

def calculate_psnr(img1, img2):
    # 채널별로 PSNR 계산
    return peak_signal_noise_ratio(img1, img2, data_range=1.0)

def calculate_ssim(img1, img2):
    ssim_index, _ = ssim(img1, img2, full=True, channel_axis=-1,data_range=1.0)
    return ssim_index

def calculate_pesq(ref_signal, deg_signal):
    ref_signal = ref_signal.cpu().detach().numpy()
    deg_signal = deg_signal.cpu().detach().numpy()
    try:
        return pesq(16000, ref_signal, deg_signal, 'wb')
    except:
        return 1

def calculate_stoi(clean_signal, noisy_signal):
    return stoi(clean_signal, noisy_signal, 16000, extended = False)

def compute_AP_metric_l2(tensor_A, tensor_B, nt):
    if tensor_A.ndim == 4:
        B, H, W, C = tensor_A.shape
        N = B * H * W

        # 텐서를 (N, C) 형태로 변환
        tensor_A = tensor_A.view(N, C)
        tensor_B = tensor_B.view(N, C)

    elif tensor_A.ndim == 2:
        N, C = tensor_A.shape

    
    total_recall_n = []
    total_precision_n = []
    total_AP = []
    # 각 벡터에 대해 L2 거리 계산
    dists = torch.cdist(tensor_A.unsqueeze(0), tensor_B.unsqueeze(0), p=2).squeeze(0).cuda()  # (N, N) 형태

    # 거리 순서대로 정렬
    sorted_dists, sorted_indices = torch.sort(dists, dim=1)

    # 각 앵커 픽셀에 대한 positive 매치의 랭크 찾기
    positive_indices = torch.arange(N).unsqueeze(1).cuda()  # (N, 1)
    positive_ranks = (sorted_indices == positive_indices).nonzero()[:,1]

    # Recall@n 계산
    for n in nt:
        recall_n = (positive_ranks < n).float().mean()

        # Precision@n 계산
        # 전체 Top-n 결과에서 Positive 매치의 총합을 계산
        total_positive_in_top_n = (positive_ranks < n).float().sum()
        total_elements_in_top_n = N * n
        precision_n = total_positive_in_top_n / total_elements_in_top_n

        # AP 계산
        # 각 쿼리에 대한 Precision@k에서 Positive 매치의 Precision 값을 평균
        ap_per_query = 1.0 / (positive_ranks + 1).float()
        AP = ap_per_query.mean()
        total_recall_n.append(recall_n.item())
        total_precision_n.append(precision_n.item())
        total_AP.append(AP.item())

    return total_recall_n, total_precision_n, total_AP

def compute_AP_metric_cos(tensor_A, tensor_B, nt):

    N, T = tensor_A.shape

    dists = torch.nn.functional.cosine_similarity(tensor_A.unsqueeze(1), tensor_B.unsqueeze(0), dim=2).cuda()

    sorted_dists, sorted_indices = torch.sort(dists, dim=1, descending=True)

    positive_indices = torch.arange(N).unsqueeze(1).cuda()  # (N, 1)
    positive_ranks = (sorted_indices == positive_indices).nonzero()[:,1]

    total_recall_n = []
    total_precision_n = []
    total_AP = []

    for n in nt:
        # Recall@n 계산
        recall_n = (positive_ranks < n).float().mean()

        # Precision@n 계산
        # 전체 Top-n 결과에서 Positive 매치의 총합을 계산
        total_positive_in_top_n = (positive_ranks < n).float().sum()
        total_elements_in_top_n = N * n
        precision_n = total_positive_in_top_n / total_elements_in_top_n

        # AP 계산
        # 각 쿼리에 대한 Precision@k에서 Positive 매치의 Precision 값을 평균
        ap_per_query = 1.0 / (positive_ranks + 1).float()
        AP = ap_per_query.mean()

        total_recall_n.append(recall_n.item())
        total_precision_n.append(precision_n.item())
        total_AP.append(AP.item())
    
    return total_recall_n, total_precision_n, total_AP

import torch

def compute_l2_distance(image_a, image_b):
    distances = torch.norm(image_a - image_b, p=2, dim=2)
    return distances

def compute_precision_recall(top_n_mask, total_positives):
    top_n_mask = top_n_mask.float()
    # True Positive의 누적 합계
    tp_cumulative = torch.cumsum(top_n_mask, dim=0)
    # 위치 인덱스 (1부터 n까지)
    positions = torch.arange(1, len(top_n_mask) + 1, device=top_n_mask.device).float()
    # 각 위치에서의 Precision
    precision_at_k = tp_cumulative / positions
    # 각 위치에서의 Recall
    recall_at_k = tp_cumulative / total_positives
    return precision_at_k, recall_at_k

def compute_average_precision(sorted_labels):

    if not isinstance(sorted_labels, torch.Tensor):
        sorted_labels = torch.tensor(sorted_labels)
    
    sorted_labels = sorted_labels.float()
    total_positives = sorted_labels.sum()
    if total_positives == 0:
        return 0.0  # 양성 예제가 없으면 AP는 0
    
    cumulative_relevance = sorted_labels.cumsum(dim=0)
    k_indices = torch.arange(1, len(sorted_labels) + 1).cuda()
    precision_at_k = cumulative_relevance / k_indices
    precision_at_k = precision_at_k * sorted_labels
    # AP 계산
    ap = precision_at_k.sum() / total_positives
    return ap.item()


def compute_AP_metric_l2_for_inference(tensor_A, tensor_B, mask, nt):
    if tensor_A.ndim == 5:
        B, T, H, W, C = tensor_A.shape
        N = B * H * W

        # 텐서를 (T, N, C) 형태로 변환
        tensor_A = tensor_A.permute(1, 0, 2, 3, 4).reshape(T, N, C)
        tensor_B = tensor_B.permute(1, 0, 2, 3, 4).reshape(T, N, C)
        mask = mask.permute(1, 0, 2, 3).reshape(T, N)

    elif tensor_A.ndim == 3:
        T, N, C = tensor_A.shape
        mask = mask.view(T, N)

    total_recall_n = [0] * len(nt)
    total_precision_n = [0] * len(nt)
    total_AP = [0] * len(nt)

    distances = torch.norm(tensor_A - tensor_B, p=2, dim=2)  # distances shape: (T, N)

    # 시간 단계별로 루프
    for t in range(T):
        distances_t = distances[t]  # (N,)
        mask_t = mask[t]            # (N,)

        # 거리를 내림차순으로 정렬 (거리가 큰 것이 긍정 사례)
        sorted_distances_t, sorted_indices_t = torch.sort(distances_t, descending=True)
        sorted_mask_t = mask_t[sorted_indices_t]

        total_positives = mask_t.sum().float()
        for idx, n in enumerate(nt):
            n = min(n, len(sorted_mask_t))
            top_n_mask = sorted_mask_t[:n]
            num_correct = top_n_mask.sum().float()

            # Precision@n, Recall@n 계산
            precision_n = num_correct / n if n > 0 else 0.0
            recall_n = num_correct / total_positives if total_positives > 0 else 0.0

            total_precision_n[idx] += precision_n
            total_recall_n[idx] += recall_n

            # 각 n에 대한 AP 계산
            ap_n = compute_average_precision(top_n_mask)
            total_AP[idx] += ap_n

    

    return total_precision_n, total_recall_n, total_AP, T


def compute_AP_metric_cosine_for_inference(tensor, mask):

    if tensor_A.ndim == 5:
        B, T, H, W, C = tensor_A.shape
        N = B * H * W

        # 텐서를 (T, N, C) 형태로 변환
        tensor_A = tensor_A.permute(1, 0, 2, 3, 4).reshape(T, N, C)
        tensor_B = tensor_B.permute(1, 0, 2, 3, 4).reshape(T, N, C)
        mask = mask.permute(1, 0, 2, 3).reshape(T, N)

    elif tensor_A.ndim == 3:
        T, N, C = tensor_A.shape
        mask = mask.view(T, N)

    total_recall_n = [0] * len(nt)
    total_precision_n = [0] * len(nt)
    total_AP = [0] * len(nt)

    # 시간 단계별로 루프
    for t in range(T):
        tensor_A_t = tensor_A[t]  # (N, C)
        tensor_B_t = tensor_B[t]  # (N, C)
        mask_t = mask[t]          # (N,)

        # 코사인 유사도 계산 (높을수록 유사)
        similarities_t = F.cosine_similarity(tensor_A_t, tensor_B_t, dim=1)  # (N,)

        # 유사도를 오름차순으로 정렬 (낮은 것이 긍정 사례)
        sorted_similarities_t, sorted_indices_t = torch.sort(similarities_t, descending=False)
        sorted_mask_t = torch.gather(mask_t, 0, sorted_indices_t)

        # Average Precision 계산
        total_positives = mask_t.sum().float()
        for idx, n in enumerate(nt):
            n = min(n, len(sorted_mask_t))
            top_n_mask = sorted_mask_t[:n]
            num_correct = top_n_mask.sum().float()

            # Precision@n, Recall@n 계산
            precision_n = num_correct / n if n > 0 else 0.0
            recall_n = num_correct / total_positives if total_positives > 0 else 0.0

            total_precision_n[idx] += precision_n
            total_recall_n[idx] += recall_n

            # 각 n에 대한 AP 계산
            ap_n = compute_average_precision(top_n_mask)
            total_AP[idx] += ap_n
            
    return total_precision_n, total_recall_n, total_AP, T


import torch
import torch.nn.functional as F

def compute_average_precision(sorted_mask):
    # 누적 합을 통해 각 위치까지의 정확한 예측 수를 계산
    cum_correct = torch.cumsum(sorted_mask, dim=0).float()
    total_correct = sorted_mask.sum().float()

    # 각 위치에서의 Precision 계산
    precisions = cum_correct / torch.arange(1, len(sorted_mask) + 1, device=sorted_mask.device).float()
    
    # 모든 positive 인스턴스에 대해 계산된 precision의 평균을 AP로 사용
    if total_correct == 0:
        return 0.0
    return (precisions * sorted_mask).sum() / total_correct

def compute_average_precision(sorted_mask):
    # 누적 합을 통해 각 위치까지의 정확한 예측 수를 계산
    cum_correct = torch.cumsum(sorted_mask, dim=0).float()
    total_correct = sorted_mask.sum().float()

    # 각 위치에서의 Precision 계산
    precisions = cum_correct / torch.arange(1, len(sorted_mask) + 1, device=sorted_mask.device).float()
    
    # 모든 positive 인스턴스에 대해 계산된 precision의 평균을 AP로 사용
    if total_correct == 0:
        return 0.0
    return (precisions * sorted_mask).sum() / total_correct

def interpolate_N(N_known, known_indices, T):
    # N_known: (N_batches, t, H, W)
    N_batches, t, H, W = N_known.shape
    device = N_known.device
    dtype = N_known.dtype

    # Reshape the tensor to (N_batches, H * W, t)
    N_known = N_known.permute(0, 2, 3, 1).reshape(N_batches, H * W, t)

    # Prepare scale factor
    scale_factor = T / t

    # Perform interpolation along the last dimension (time dimension)
    N_interpolated = F.interpolate(N_known, size=T, mode='linear', align_corners=False)

    # Reshape back to (N_batches, T, H, W)
    N_interpolated = N_interpolated.reshape(N_batches, H, W, T).permute(0, 3, 1, 2).permute(1, 0, 2, 3)


    return N_interpolated

def compute_APall_metric_cosine_for_inference(tensor, mask):
    T = tensor.shape[0]


       
    similarities_t = tensor
    mask_t = mask
    k = 0

    if mask_t.sum() == 0:
        k += 1
        
    

    sorted_similarities_t, sorted_indices_t = torch.sort(similarities_t, descending=False)
    sorted_mask_t = torch.gather(mask_t, 0, sorted_indices_t)

    ap_t = compute_average_precision(sorted_mask_t)
    print(ap_t)

    return ap_t

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

def compute_AUC_metric_cosine_for_inference(tensor, mask):
    tensor = 1 - tensor

    try:
        # roc_auc_score 함수를 사용하여 AUC 계산
        auc_t = roc_auc_score(mask.cpu().numpy(), tensor.cpu().detach().numpy())
    except ValueError as e:
        # 모든 레이블이 같거나 샘플 수가 충분하지 않을 때 예외 처리
        print(f"Error calculating AUC for timestep {t}: {e}")
        

    return auc_t

