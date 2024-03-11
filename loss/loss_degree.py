import numpy as np
import math
import torch


def best_pos_cosine_similarity(query, pos_vecs):

    query_norm = query / query.norm(dim=2, keepdim=True)
    pos_vecs_norm = pos_vecs / pos_vecs.norm(dim=2, keepdim=True)
 
    cosine_sim = torch.bmm(query_norm, pos_vecs_norm.transpose(1, 2)).squeeze(1)
   
    epsilon = 1e-6
    radians = 180/math.pi*torch.acos(cosine_sim.clamp(-1 + epsilon, 1 - epsilon))  
   
    min_radians, _ = radians.min(1)
    max_radians, _ = radians.max(1)

    return min_radians, max_radians

def quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2, use_min=False, lazy=False, ignore_zero_loss=False):
    min_pos, max_pos = best_pos_cosine_similarity(q_vec, pos_vecs)
    
    if use_min:
        positive = min_pos
    else:
        positive = max_pos

    positive = positive.view(-1, 1)
 
    query_norm = q_vec / q_vec.norm(dim=2, keepdim=True)
    neg_vecs_norm = neg_vecs / neg_vecs.norm(dim=2, keepdim=True)
    cosine_sim_qn = torch.bmm(query_norm, neg_vecs_norm.transpose(1, 2)).squeeze(1)

    epsilon = 1e-6
    radians_qn = 180/math.pi*torch.acos(cosine_sim_qn.clamp(-1 + epsilon, 1 - epsilon))   

    loss = m1+ positive - radians_qn  
    loss = loss.clamp(min=0.0)

    if lazy:
        triplet_loss = loss.max(1)[0]
    else:
        triplet_loss = loss.sum(1)
    if ignore_zero_loss:
        hard_triplets = torch.gt(triplet_loss, 1e-16).float()
        num_hard_triplets = torch.sum(hard_triplets)
        triplet_loss = triplet_loss.sum() / (num_hard_triplets + 1e-16)
    else:
        triplet_loss = triplet_loss.mean()

    other_neg_vecs_norm = other_neg / other_neg.norm(dim=2, keepdim=True)
    cosine_sim_no = torch.bmm(other_neg_vecs_norm,neg_vecs_norm.transpose(1, 2)).squeeze(1)
    radians_no = 180/math.pi*torch.acos(cosine_sim_no.clamp(-1 + epsilon, 1 - epsilon)) 

    second_loss = m2 + positive - radians_no
    second_loss = second_loss.clamp(min=0.0)


    if lazy:
        second_loss = second_loss.max(1)[0]
    else:
        second_loss = second_loss.sum(1)

    if ignore_zero_loss:
        hard_second = torch.gt(second_loss, 1e-16).float()
        num_hard_second = torch.sum(hard_second)
        second_loss = second_loss.sum() / (num_hard_second + 1e-16)
    else:
        second_loss = second_loss.mean()

    total_loss = triplet_loss + second_loss

    return total_loss
