import torch
import time
import collections
import numpy as np

def dist(emb_dict, min_gallery=6, batch_size=80, use_gpu=False, print_time=False):
    """
    Complete MAP torch
    """
    # Convert embedding dictionary to torch tensors
    probe_emb_list = []
    gallery_emb_list = []
    probe_class_list = []
    gallery_class_list = []
    for i,c in enumerate(sorted(emb_dict)):
        if len(emb_dict[c]) > min_gallery:
            for emb in emb_dict[c]:
                probe_emb_list.append(emb.unsqueeze(0))
                probe_class_list.append(i)
        for emb in emb_dict[c]:
            gallery_emb_list.append(emb.unsqueeze(0))
            gallery_class_list.append(i)

    # Convert the embedding list to a torch tensor
    probe_emb_tsr = torch.cat(probe_emb_list)
    gallery_emb_tsr = torch.cat(gallery_emb_list)
    m, _ = probe_emb_tsr.size()
    n, _ = gallery_emb_tsr.size()

    # Convert the list of classes corresponding to the embedding list
    # to a torch tensor
    probe_class_tsr = torch.LongTensor(probe_class_list)
    gallery_class_tsr = torch.LongTensor(gallery_class_list)

    # Keep track of the average precision for each probe/gallery
    avg_precision_tsr = torch.zeros(m)

    # Precomputed torch.range tensor to speed up the computation
    range_tsr = torch.arange(0, n-1)

    # Instantiate dict for distances
    dist_dict = collections.defaultdict(lambda : collections.defaultdict(list))

    # The class tensor, full embedding tensor, and a precomputed torch.range
    # tensor are converted to CUDA tensors which use GPU.
    if use_gpu:
        probe_class_tsr = probe_class_tsr.cuda()
        gallery_class_tsr = gallery_class_tsr.cuda()
        probe_emb_tsr = probe_emb_tsr.cuda()
        gallery_emb_tsr = gallery_emb_tsr.cuda()
        range_tsr = range_tsr.cuda()

    # Batchify the computation
    for i in range(0, m, batch_size):
        # Carve out the mini-batch from the full embedding tensor
        probe_emb_tsr_part = probe_emb_tsr[i:i+batch_size, :]
        if use_gpu:
            probe_emb_tsr_part = probe_emb_tsr_part.cuda()

        # Compute squared differences for all embedding pairs
        dist_tsr = ((gallery_emb_tsr.unsqueeze(0) - probe_emb_tsr_part.unsqueeze(1))**2).sum(dim=2)
        # Assimilate distances
        for j1 in range(dist_tsr.size(0)):
            for j2 in range(dist_tsr.size(1)):
                dist_dict[probe_class_tsr[i+j1]][gallery_class_tsr[j2]].append(dist_tsr[j1, j2])

    return dist_dict

def map(emb_dict, min_gallery=6, batch_size=80, use_gpu=False, print_time=False, per_class=False):
    """
    Complete MAP torch
    """
    # Convert embedding dictionary to torch tensors
    probe_emb_list = []
    gallery_emb_list = []
    probe_class_list = []
    gallery_class_list = []
    for i,c in enumerate(sorted(emb_dict)):
        if len(emb_dict[c]) > min_gallery:
            for emb in emb_dict[c]:
                probe_emb_list.append(emb.unsqueeze(0))
                probe_class_list.append(i)
        for emb in emb_dict[c]:
            gallery_emb_list.append(emb.unsqueeze(0))
            gallery_class_list.append(i)

    # Convert the embedding list to a torch tensor
    probe_emb_tsr = torch.cat(probe_emb_list)
    gallery_emb_tsr = torch.cat(gallery_emb_list)
    m, _ = probe_emb_tsr.size()
    n, _ = gallery_emb_tsr.size()
    # Convert the list of classes corresponding to the embedding list
    # to a torch tensor
    probe_class_tsr = torch.LongTensor(probe_class_list)
    gallery_class_tsr = torch.LongTensor(gallery_class_list)

    # Keep track of the average precision for each probe/gallery
    avg_precision_tsr = torch.zeros(m)

    # Precomputed torch.range tensor to speed up the computation
    range_tsr = torch.arange(0, n-1, dtype=torch.float)

    # Keep track of time spent in different parts of function for profiling
    dist_mat_time = 0.0
    comp_map_time = 0.0

    # Index for the current probe/gallery
    j = 0

    # The class tensor, full embedding tensor, and a precomputed torch.range
    # tensor are converted to CUDA tensors which use GPU.
    if use_gpu:
        probe_class_tsr = probe_class_tsr.cuda()
        gallery_class_tsr = gallery_class_tsr.cuda()
        probe_emb_tsr = probe_emb_tsr.cuda()
        gallery_emb_tsr = gallery_emb_tsr.cuda()
        range_tsr = range_tsr.cuda()

    # Batchify the computation
    for i in range(0, m, batch_size):
        st1 = time.time()

        # Carve out the mini-batch from the full embedding tensor
        probe_emb_tsr_part = probe_emb_tsr[i:i+batch_size, :]
        if use_gpu:
            probe_emb_tsr_part = probe_emb_tsr_part.cuda()

        # Compute squared differences for all embedding pairs
        dist_tsr = ((gallery_emb_tsr.unsqueeze(0) - probe_emb_tsr_part.unsqueeze(1))**2).sum(dim=2)

        dt1 = time.time()
        dist_mat_time += (dt1 - st1)

        # Vectorize MAP
        st2 = time.time()
        for dist_row in dist_tsr:
            # Sort dist row, take all but first element (the identity)
            _, sorted_idx = torch.sort(dist_row)
            gallery_idx = sorted_idx[1:]
            # Get the class indeces corresponding to the sorted distances
            probe_class = probe_class_tsr[j]
            sorted_class_row = gallery_class_tsr[gallery_idx]
            # Produce binary array by comparing the sorted class row to the probe class
            binary_class_row = sorted_class_row==probe_class
            # Get indeces of matches
            match_idx_row = range_tsr[binary_class_row == True]
            # Get match counts
            match_count_row = range_tsr[:len(match_idx_row)]+1
            # Get total counts up to each match
            tot_count_row = match_idx_row+1
            # Divide element-wise to get precision
            precision_arr = match_count_row/tot_count_row
            # Take mean of precision array to get average precision
            avg_precision = torch.mean(precision_arr)
            # Accumulate average precision
            avg_precision_tsr[j] = avg_precision
            # Increment index for probe/gallery
            j += 1
        dt2 = time.time()
        comp_map_time += (dt2 - st2)
    # Take the mean of the average precision tensor to get the 'complete' MAP
    amap = torch.mean(avg_precision_tsr)
    # Print time profiling info
    if print_time:
        print('Total dist mat time: {:2.2f}'.format(dist_mat_time))
        print('Total comp map time: {:2.2f}'.format(comp_map_time))
    # Return 'complete' MAP result
    # Get per-class MAP
    if per_class:
        _, idx = probe_class_tsr.cpu().sort() 
        class_view_tsr = avg_precision_tsr.view(probe_class_tsr.max()+1, -1)
        class_mean_tsr = class_view_tsr.mean(dim=1) 
        return amap.item(), class_mean_tsr
    else:
        return amap.item()

def top_n(emb_dict, path_dict, min_gallery=6, top_n=10, num_probes=3):
    """
    Random MAP numpy
    """

    # Seed RNG
    #np.random.seed(0)

    avg_precision_list = []

    # Compute distance matrix
    num_class = 0
    probe_emb_list = []
    gallery_emb_list = []
    probe_class_list = []
    gallery_class_list = []
    probe_path_list = []
    gallery_path_list = []
    for c in sorted(emb_dict):
        Ni = len(emb_dict[c])
        rand_idx = np.arange(Ni)
        np.random.shuffle(rand_idx)
        if len(emb_dict[c]) > min_gallery:
            probe_idx = rand_idx[-1]
            probe_emb_list.append(emb_dict[c][probe_idx])
            probe_class_list.append(c)
            probe_path_list.append(path_dict[c][probe_idx])
            num_class += 1
        gallery_idx_list = rand_idx[:max(min_gallery, Ni-1)]
        for gallery_idx in gallery_idx_list:
            gallery_emb_list.append(emb_dict[c][gallery_idx])
            gallery_class_list.append(c)
            gallery_path_list.append(path_dict[c][gallery_idx])
    probe_emb_arr = np.array(probe_emb_list).T
    gallery_emb_arr = np.array(gallery_emb_list).T
    probe_class_arr = np.array(probe_class_list)
    gallery_class_arr = np.array(gallery_class_list)
    probe_path_arr = np.array(probe_path_list)
    gallery_path_arr = np.array(gallery_path_list)

    # Reshape emb arrs
    sp = probe_emb_arr.shape
    sg = gallery_emb_arr.shape
    probe_emb_arr = probe_emb_arr.reshape((sp[0], 1, sp[1]))
    gallery_emb_arr = gallery_emb_arr.reshape((sg[0], 1, sg[1]))

    # Select random probes to be used
    rand_idx = np.random.choice(range(probe_class_arr.shape[0]), num_probes, replace=False)
    probe_emb_arr_part = probe_emb_arr[:, :, rand_idx]
    probe_class_arr_part = probe_class_arr[rand_idx]

    # Compute squared differences for all embedding pairs
    dist_arr = ((gallery_emb_arr - probe_emb_arr_part.swapaxes(1, 2))**2).sum(axis=0)

    probe_paths = probe_path_arr[rand_idx]

    # Vectorize MAP
    match_idx_arr = []
    gallery_paths = []
    gallery_dists = []
    for j, dist_row in enumerate(dist_arr):
        # Sort dist row, take all but first element (the identity)
        gallery_idx = np.argsort(dist_row)
        gallery_dist = dist_row[gallery_idx][:top_n]
        gallery_dists.append(gallery_dist)
        # Get the class indeces corresponding to the sorted distances
        probe_class = probe_class_arr_part[j]
        sorted_class_row = gallery_class_arr[gallery_idx]
        # Produce binary array by comparing the sorted class row to the probe class
        binary_class_row = sorted_class_row==probe_class
        match_idx_arr.append(binary_class_row[:top_n])
        gallery_paths.append(gallery_path_arr[gallery_idx][:top_n])

    return probe_paths, gallery_paths, gallery_dists, match_idx_arr

def map_from_dist(dist_arr, batch_size=80, use_gpu=False, print_time=False):
    """
    Complete MAP given an array of distances.
    """
    # Convert embedding dictionary to torch tensors
    probe_emb_list = []
    gallery_emb_list = []
    probe_class_list = []
    gallery_class_list = []
    for i,c in enumerate(sorted(emb_dict)):
        if len(emb_dict[c]) > min_gallery:
            for emb in emb_dict[c]:
                probe_emb_list.append(emb.unsqueeze(0))
                probe_class_list.append(i)
        for emb in emb_dict[c]:
            gallery_emb_list.append(emb.unsqueeze(0))
            gallery_class_list.append(i)

    # Convert the embedding list to a torch tensor
    probe_emb_tsr = torch.cat(probe_emb_list)
    gallery_emb_tsr = torch.cat(gallery_emb_list)
    m, _ = probe_emb_tsr.size()
    n, _ = gallery_emb_tsr.size()

    # Convert the list of classes corresponding to the embedding list
    # to a torch tensor
    probe_class_tsr = torch.LongTensor(probe_class_list)
    gallery_class_tsr = torch.LongTensor(gallery_class_list)

    # Keep track of the average precision for each probe/gallery
    avg_precision_tsr = torch.zeros(m)

    # Precomputed torch.range tensor to speed up the computation
    range_tsr = torch.arange(0, n-1)

    # Keep track of time spent in different parts of function for profiling
    dist_mat_time = 0.0
    comp_map_time = 0.0

    # Index for the current probe/gallery
    j = 0

    # The class tensor, full embedding tensor, and a precomputed torch.range
    # tensor are converted to CUDA tensors which use GPU.
    if use_gpu:
        probe_class_tsr = probe_class_tsr.cuda()
        gallery_class_tsr = gallery_class_tsr.cuda()
        probe_emb_tsr = probe_emb_tsr.cuda()
        gallery_emb_tsr = gallery_emb_tsr.cuda()
        range_tsr = range_tsr.cuda()

    # Batchify the computation
    for i in range(0, m, batch_size):
        st1 = time.time()

        # Carve out the mini-batch from the full embedding tensor
        probe_emb_tsr_part = probe_emb_tsr[i:i+batch_size, :]
        if use_gpu:
            probe_emb_tsr_part = probe_emb_tsr_part.cuda()

        # Compute squared differences for all embedding pairs
        dist_tsr = ((gallery_emb_tsr.unsqueeze(0) - probe_emb_tsr_part.unsqueeze(1))**2).sum(dim=2)

        dt1 = time.time()
        dist_mat_time += (dt1 - st1)

        # Vectorize MAP
        st2 = time.time()
        for dist_row in dist_tsr:
            # Sort dist row, take all but first element (the identity)
            _, sorted_idx = torch.sort(dist_row)
            gallery_idx = sorted_idx[1:]
            # Get the class indeces corresponding to the sorted distances
            probe_class = probe_class_tsr[j]
            sorted_class_row = gallery_class_tsr[gallery_idx]
            # Produce binary array by comparing the sorted class row to the probe class
            binary_class_row = sorted_class_row==probe_class
            # Get indeces of matches
            match_idx_row = range_tsr[binary_class_row == True]
            # Get match counts
            match_count_row = range_tsr[:len(match_idx_row)]+1
            # Get total counts up to each match
            tot_count_row = match_idx_row+1
            # Divide element-wise to get precision
            precision_arr = match_count_row/tot_count_row
            # Take mean of precision array to get average precision
            avg_precision = torch.mean(precision_arr)
            # Accumulate average precision
            avg_precision_tsr[j] = avg_precision
            # Increment index for probe/gallery
            j += 1
        dt2 = time.time()
        comp_map_time += (dt2 - st2)
    # Take the mean of the average precision tensor to get the 'complete' MAP
    amap = torch.mean(avg_precision_tsr)
    # Print time profiling info
    if print_time:
        print('Total dist mat time: {:2.2f}'.format(dist_mat_time))
        print('Total comp map time: {:2.2f}'.format(comp_map_time))
    # Return 'complete' MAP result
    return amap
