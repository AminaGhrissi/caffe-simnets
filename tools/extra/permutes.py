import numpy as np



def swap_locs(mat,src_loc,dst_loc):
    src_h,src_w = src_loc
    dst_h,dst_w = dst_loc
    tmp = mat[src_h,src_w]
    mat[src_h,src_w] = mat[dst_h,dst_w]
    mat[dst_h,dst_w] = tmp

"""
Given dimensions of input (2D) and a maximum radius R, returns a string 
'x0,x1,...' defining a permutation such that location x_0 will move to location
0, location x_i to location i (row major indexing). Each location will move at
most R locations from original spot (grid distances, diagonal counts as 1). 
"""
def radius_permute(in_size, R):    
    step_size = 1
    H,W = in_size    
    N = np.prod(in_size)
    move_count_mat = np.zeros(in_size)    
    source_idxs = np.arange(N).reshape(in_size)
    out_mat = np.arange(N).reshape(in_size)
    per_order = range(N)
    for step in range(R):  
        np.random.shuffle(per_order)
        for i,k in enumerate(per_order):
            src_h,src_w = np.unravel_index(k,in_size)
            # selected source location already moved maximum distance
            if move_count_mat[src_h,src_w] == R:  
                continue
            # available destination locations to swap with
            usage_mat = move_count_mat < R 
            radius_mask = usage_mat[max(0,src_h - step_size):min(src_h + step_size + 1,H),
            max(0,src_w - step_size):min(src_w + step_size + 1,W)]
            radius_idxs = source_idxs[max(0,src_h - step_size):min(src_h + step_size + 1,H),
            max(0,src_w - step_size):min(src_w + step_size + 1,W)]
            el_idxs = radius_idxs[radius_mask]
            sel_idx = np.random.choice(el_idxs,1)
            dst_h,dst_w = np.unravel_index(sel_idx,in_size)
            assert( not (move_count_mat[src_h,src_w] > R or move_count_mat[dst_h,dst_w] > R ))
            if src_h != dst_h or src_w != dst_w:
                move_count_mat[src_h,src_w] += 1
                move_count_mat[dst_h,dst_w] += 1
            swap_locs(out_mat,(src_h,src_w),(dst_h,dst_w))
            swap_locs(move_count_mat,(src_h,src_w),(dst_h,dst_w))
            
            
    # return locations as comma-separated string in row major order.
    out_str = ','.join(map(str, out_mat.ravel())) 
    return out_str


"""
Given dimensions of input (2D) and a shift sizes s_x, s_y, returns a string 
 defining a permutation such that each row will be shifted down by 
s_y (negative means shift up) and each column will be shifted s_x right 
(left if s_x < 0)
"""
def cyclic_shift_permute(in_size, s_x, s_y):    
    H,W = in_size    
    N = np.prod(in_size)
    source_idxs = np.arange(N).reshape(in_size)
    res = np.roll(source_idxs,s_y,axis=0)
    out_mat = np.roll(res,s_x,axis=1)
        
    # return locations as comma-separated string in row major order.
    out_str = ','.join(map(str, out_mat.ravel())) 
    return out_str
    
    
    
"""
Utility function to generate lookup table and inverse lookup table defining
the permutation, given the input permutation string (in the format 
corresponding to the Caffe PermutationLayer).
"""
def lt_from_str(permute_string):
    i_lt = np.array([int(x) for x in permute_string.split(",")])
    lt = np.ones_like(i_lt) * -1
    for j in range(len(i_lt)):
        lt[i_lt[j]] = j
    return lt,i_lt
       
"""
Utility function to test operation of the radius_permute method
"""
def test_validity(src_mat, dst_mat, R):
    H,W = src_mat.shape
    for h in range(H):
        for w in range(W):
            src_h,src_w = np.unravel_index(src_mat[h,w],(H,W))
            dst_h,dst_w = np.unravel_index(dst_mat[h,w],(H,W))
            if np.abs(src_h-dst_h) > R or np.abs(src_w-dst_w) > R:
                print "Radius %d exceeded, (%d,%d) moved to (%d,%d) " % (R,src_h,src_w,dst_h,dst_w)
                return False
    return True
            
    
    
if __name__ == '__main__':
    in_size = (3,5)
    out_str = cyclic_shift_permute(in_size ,s_x=-1,s_y=1)
    lt,ilt = lt_from_str(out_str)
    out_mat = ilt.reshape(in_size)