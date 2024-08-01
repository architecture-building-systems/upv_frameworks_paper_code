
def change_tree(tree_array, coverage_factor=0.25, seed=42):
    np.random.seed(seed)
    original_shape = tree_array.shape
    flat_tree_arr = tree_array.flatten()
    tree_arr_idx = np.arange(len(flat_tree_arr))
    total_cells = len(flat_tree_arr)
    
    # get the indices in the larger array where there are not leaves
    no_leaf_array_idx = tree_arr_idx[flat_tree_arr==0]
    non_leaf_cell_count = len(no_leaf_array_idx)
    print("Currently, there are", non_leaf_cell_count, "non leaves")
    
    # get the indices in the larger array where there are leaves
    leaf_array_idx = tree_arr_idx[flat_tree_arr==1]
    leaf_cell_count = len(leaf_array_idx)
    print("Currently, there are", leaf_cell_count, "leaves")
    
    current_coverage = leaf_cell_count / total_cells
    print("Current coverage is", round(current_coverage,3)*100, "%")
    
    if current_coverage<coverage_factor:
        print("Adding leaves")
        
        # calculate the number of leaves that must remain
        n_cells_cover = int(total_cells * coverage_factor)
        print(n_cells_cover, "cells must be covered.")
        
        # calculate the number of cells that must be added
        n_cells_add = n_cells_cover - int(leaf_cell_count)
        
        # generate a random list of indices that is the length of the non_leaf_array_idx    
        rng = np.random.default_rng()
        cell_add_idx = rng.choice(non_leaf_cell_count-1, 
                                     size=n_cells_add, 
                                     replace=False)
        
        # grab the idx numbers from the larger array that need to be deleted
        # leaf_array_idx is the list of idx in flat_tree_arr where there are leaves 
        add_idx = no_leaf_array_idx[cell_add_idx]
        
        # add leaves
        flat_tree_arr[add_idx] = 1
        
        return flat_tree_arr.reshape(original_shape)
        
        
    elif current_coverage>coverage_factor:
        print("Removing leaves")
        
        # calculate the number of leaves that must remain
        n_cells_cover = int(total_cells * coverage_factor)
        print(n_cells_cover, "cells must remain covered.")
        
        # calculate the number of leaves that must be removed
        n_cells_delete = int(leaf_cell_count - n_cells_cover)
        print("Therefore delete", n_cells_delete, "cells")
        
        # generate a random list of indices that is the length of the leaf_array_idx    
        rng = np.random.default_rng()
        cell_delete_idx = rng.choice(leaf_cell_count-1, 
                                     size=n_cells_delete, 
                                     replace=False)
        
        print("Len of delete idx arr", len(cell_delete_idx))
        
        # grab the idx numbers from the larger array that need to be deleted
        # leaf_array_idx is the list of idx in flat_tree_arr where there are leaves 
        delete_idx = leaf_array_idx[cell_delete_idx]
        print("Len of delete idx arr 2 is", len(delete_idx))
        
        # set a 0 into the flat_tree_arr where delete_idx tells it to
        print("Shape of array is", flat_tree_arr.shape)
        print("Sum of array is", np.sum(flat_tree_arr))
        flat_tree_arr[delete_idx] = 0
        print("Shape of array is", flat_tree_arr.shape)
        print("Sum of array is", np.sum(flat_tree_arr))

        return flat_tree_arr.reshape(original_shape)
    
    else:
        print("Tree already masks input coverage factor. No change made")
        return tree_array
    
    

def change_tree_blobs(tree_array, n_clusters=5000, coverage_factor=0.25, seed=42):
    np.random.seed(seed)
    original_shape = tree_array.shape
    flat_tree_arr = tree_array.flatten()
    tree_arr_idx = np.arange(len(flat_tree_arr))
    total_cells = len(flat_tree_arr)
    
    # get the indices in the larger array where there are not leaves
    no_leaf_array_idx = tree_arr_idx[flat_tree_arr==0]
    non_leaf_cell_count = len(no_leaf_array_idx)
    print("Currently, there are", non_leaf_cell_count, "non leaves")
    
    # get the indices in the larger array where there are leaves
    leaf_array_idx = tree_arr_idx[flat_tree_arr==1]
    leaf_cell_count = len(leaf_array_idx)
    print("Currently, there are", leaf_cell_count, "leaves")
    
    current_coverage = leaf_cell_count / total_cells
    print("Current coverage is", round(current_coverage,3)*100, "%")
    
    if current_coverage<coverage_factor:
        print("Adding leaves")
        
        # calculate the number of leaves that must remain
        n_cells_cover = int(total_cells * coverage_factor)
        print(n_cells_cover, "cells must be covered.")
        
        # calculate the number of cells that must be added
        n_cells_add = n_cells_cover - int(leaf_cell_count)

        # calculate the cluster size necessary
        cluster_size = int(n_cells_add / n_clusters)
        
        # select the center point of the clusters
        rng = np.random.default_rng()
        cell_add_idx = rng.choice(non_leaf_cell_count-1, 
                                  size=n_clusters, 
                                  replace=False)
        expanded_cell_add_idx = []
        
        for i in cell_add_idx:
            for n in np.arange(int(i-(cluster_size/2)),int(i+(cluster_size/2))):
                expanded_cell_add_idx.append(n)
                
        expanded_cell_add_idx = np.clip(np.array(expanded_cell_add_idx),0,non_leaf_cell_count-1)
        
        add_idx = no_leaf_array_idx[expanded_cell_add_idx]
        flat_tree_arr[add_idx] = 1
    
        flat_tree_arr = change_tree(flat_tree_arr, coverage_factor=coverage_factor, seed=seed)
        return flat_tree_arr.reshape(original_shape)
    
    elif current_coverage>coverage_factor:
        print("Removing leaves")
        # calculate the number of leaves that must remain
        n_cells_cover = int(total_cells * coverage_factor)
        print(n_cells_cover, "cells must remain covered.")
        
        # calculate the number of leaves that must be removed
        n_cells_delete = int(leaf_cell_count - n_cells_cover)
        print("Therefore delete", n_cells_delete, "cells")
        
        cluster_size = int(n_cells_delete / n_clusters)
        
        rng = np.random.default_rng()
        cell_delete_idx = rng.choice(leaf_cell_count-1, 
                                        size=n_clusters, 
                                        replace=False)
        expanded_cell_delete_idx = []
        
        for i in cell_delete_idx:
            for n in np.arange(int(i-(cluster_size/2)),int(i+(cluster_size/2))):
                expanded_cell_delete_idx.append(n)
        
        expanded_cell_delete_idx = np.clip(np.array(expanded_cell_delete_idx),0,leaf_cell_count-1)
        
        delete_idx = leaf_array_idx[expanded_cell_delete_idx]
        flat_tree_arr[delete_idx] = 0
    
    
        flat_tree_arr = change_tree(flat_tree_arr, coverage_factor=coverage_factor, seed=seed)
        return flat_tree_arr.reshape(original_shape)
    
    else:
        print("Tree already masks input coverage factor. No change made")
        return tree_array