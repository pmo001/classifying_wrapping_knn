import pandas as pd
import numpy as np
import heapq
import copy #to allow deepcopy

#header=None prevents first row from becoming header
###data = pd.read_csv('small_test_data.txt', delim_whitespace=True, header=None)
#data2 = pd.read_csv('large_test_data.txt', delim_whitespace=True, header=None)
data = pd.read_csv('vsmall_data.txt', delim_whitespace=True, header=None)
#print(data)

#for each feature, normalize with z-score normalization
#returns number of standard deviation w.r.t mean
#purpose: for euclidean distance
def z_normalize_per_feature(data):    
    
    dict_mean_std = dict() #init empty dict, keys will be tuple of: (mean, std)
    #print(dict_mean_std.get(2))

    #for each col // shape returns a tuple of dimensions
    for x in range(1, data.shape[1]): #i wrt cols; starts from 1 to ignore class col
        sum = 0 #resets sum for next col
        for y in range(data.shape[0]):
            sum += data.iloc[y,x] #sums an entire col 
        #appends mean of each col into dict
        mean = sum / data.shape[0]
        dict_mean_std.update({x : mean})
        #print(np.format_float_scientific(sum)) #convert to sci not

            #col by col
            #data.iloc[:,x]
    #computing standard deviation == sqrt(variance)
    for x in range(1, data.shape[1]): #for each col from 1:
        sum = 0 #reset sum for each col
        for y in range(data.shape[0]):
            #(each el in col - col's mean)**2
             sum += (data.iloc[y,x] - dict_mean_std.get(x))**2
        #ref: https://www.graphpad.com/support/faqid/1383/
        # divide by (n-1) instead of n to account for fact that we have sample mean,
        # which is generalization of the population mean
        variance = sum / (data.shape[0] - 1)
        std_dev = np.sqrt(variance)
        #update dict with std_dev as [1] of tuple
        dict_mean_std[x] = (dict_mean_std.get(x), std_dev)

    #print("my std vals: ", dict_mean_std)
    #print("panda's std vals: ", data.std(axis=0)) #one line for std

    #print("df before alteration: ", data)
    #compute z-score for each element
    #ref fixme analyticsvidhya 8 ways to deal continuous
    #(each el - mean) / std
    for x in range(1, data.shape[1]): #for each col from 1:
        for y in range(data.shape[0]):
            z_score = (data.iloc[y,x] - dict_mean_std.get(x)[0]) / dict_mean_std.get(x)[1]
            #.at permanently alters df
            data.at[y,x] = z_score
    
    #print("df after change: ", data)
    print("> The data has been normalized to improve nearest neighbor classification")
    return


# dim/numFeatures = 3; rep: [x, y, z]  etc...
# calc euclid for ALL pts
#input: 1.a subset df(training set) w.r.t selected features
#       2. the row of comparison (the validation df)
#process: 1. finds euclid dist row by row
#       2. inserts each euclid dist into minheap as a tuple: (dist, class) //comparison w.r.t first el
#       3. pop out min dist. k times 
#       4. vote
#           a.if popped[1] == 1, count_one += 1; b.
#       5. if count_one > count_two: return 1
#output: the class val: 1 or 2
def kNearestNeigh(training_df, validation_df):
    #fixme can del below
    #print("  >>>within kNN, training_df: ")
    #print(training_df)
    #print("  >>>within kNN, valid_df: ")
    #print(validation_df)

    minheap = [] #init min heap
    #ref: https://saravananthirumuruganathan.wordpress.com/2010/05/17/a-detailed-introduction-to-k-nearest-neighbor-knn-algorithm/
    #k = sqrt(num of observations(rows))
    k_val = np.floor(np.sqrt(training_df.shape[0])) #floor ensures is_integer 
    if k_val % 2 == 0:
        k_val += 1 #makes sure is odd since numClass=2
    print("  >>>k val: ", k_val)
    #toss each of these euclid dists into minheap 
    for i in range(training_df.shape[0]): #get euclid dist: each row to the valid_pt
        if training_df.shape[1] == 2: #1 feature/dim (1 class col + 1 feature)
                # abs(training[] - valid[])
            euclid_dist = np.absolute(training_df.iat[i,1] - validation_df.iat[0,1])
            #pushes into minheap and compares euclid_dist, tuple[1]=class num
 #fixme can del           #print("before insertion into minheap; class num: ", training_df.iat[i,0])
            heapq.heappush(minheap, (euclid_dist, training_df.iat[i,0]))
 #fixme can del           #print("after insertion into minheap; minheap: ", minheap)
        elif training_df.shape[1] > 2: # >1 feature
            sum = 0
            for j in range(1, training_df.shape[1]): #for each feature(thus exclude class col)
                #ref: https://hlab.stanford.edu/brian/euclidean_distance_in.html
                #sum += (point1[i] - point2[i])**2
                sum += (training_df.iat[i,j] - validation_df.iat[0,j])**2
            euclid_dist = np.sqrt(sum)
            heapq.heappush(minheap, (euclid_dist, training_df.iat[i,0])) #tuple[1]=class num
    
    #pop out k_val rows from minheap and vote on the class of validation
    vote_class1 = 0
    vote_class2 = 0
    #print("size of minheap: ", len(minheap))
    for i in range(int(k_val.item())): #.item() converts numpy back to native python
        #pops min val while maintaining min heap invariant
        #pop will return tuple(dist, class_num) 
        if validation_df.iat[0,0] == heapq.heappop(minheap)[1]: #if valid_class = min_class
            vote_class1 += 1
        else:
            vote_class2 += 1
    
    if vote_class1 > vote_class2:
        return 1
    else:
        return 2

#TODO
#(a.)process:
#   1. CV the base df: i.e. 10 folds: 9 folds for training, 1 fold for validation
#   2. 10 training sets: within each: apply subsetFeatures(list)
#return accuracy of model w.r.t selected features
def leave_one_out_CV(base_df, subset_features_list):
    #move (2) up here?
    #1.call training_valid_split to training/valid split the base_df
    # a.the return of the call: 10 diff sets of[ training set(9) and valid set(1)]
    #2.to each of these 10: training set(9): apply variable selection which...
    # //this subsets all 10 training sets to only have the selected cols
    # a.then apply kNN to this subset of a subset
    if 0 not in subset_features_list:
        subset_features_list.append(0)
        #modifies orig
        subset_features_list.sort() #sorts so that class col is first
    print(" >within LOOCV; subset_features_list: ", subset_features_list)

    subset_df = base_df[subset_features_list] #features + class col
    #each row will become sole el of validation set at some point
    num_correct = 0
    for i in range(base_df.shape[0]):
        validation_df = subset_df.loc[subset_df.index == i] #each i will become its own validation set eventually
        training_df = subset_df.loc[subset_df.index != i] # df w select features - i
        #kNN returns the vote: either 1 or 2 for class
        # if vote is same as valid's class, ^correct
        #each forloop/t_v_block produces either a 1 or 0 in correctness
        if kNearestNeigh(training_df, validation_df) == validation_df.iat[0,0]:
            num_correct += 1
    accuracy = num_correct / i
    
    return accuracy

#TODO: test this with sklearn
#leave-one-out-CV:
#process: i observation becomes validation set
#       the rest become training set (this def(t_v_split) stops at this point)
#2. apply kNN:
#       if validation set is correctly classified by the kNN training set: +1? #FIXME
#       elif incorrectly classified, 0 / move on
#       a. take all the +1 / n to get percent of correctness?
# do #2 again for the rest of the loop where the next observation is left out for validation set
# ?: add all of the percent of correct from 2a / n ? to get final percent of correctness?
def training_valid_split(base_df, subset_df): #fixme? need subsetdf?

    #each row will become sole el of validation set at some point
    for i in range(base_df.shape[0]):
        validation_df = base_df.loc[base_df.index == i] #each i will become its own validation set eventually
        training_df = base_df.loc[base_df.index != i] #base df - i

    ###split_pool = (i, (training_df, validation_df)) #FIXME
    return 

#exclude a row from base df based on row's index
#return: a df with the row excluded
#doesn't modify baseDF
def exclude_row_from_df(base_df, idx_row_to_exclude):
    new_df = base_df.loc[base_df.index != idx_row_to_exclude]
    #print("new df with the row excluded: ", new_df)

    #a df with only row2:
    ###row_df = base_df.loc[base_df.index == 2]
    #print("basedf but only with row 2: ", row_df)

    return new_df

# selects features and forms a df with those features
# returns a df 
def init_one_feature(base_df, feature_num):
    #init a new df with a column from oldDF
    post_df = pd.DataFrame(base_df.iloc[:, feature_num])
    return post_df

#take: the df that will gain the feature
#give: the base df that has the feature
def append_feature_to_df(take, give, feature_num):
    #give2 is just another pointer (not deep copy)
    give2 = give.iloc[:, feature_num]
    df_concat = pd.concat([take, give2], axis=1)
    return df_concat

def subset_certain_features(base, feature_list):
    return base[feature_list]

#TODO complete this
#ref: Prof. Eamonn's Project_2_Briefing slides
def forward_selection(df):
    desired_features = [] #init empty list
    maxheap = []

    for i in range(1, df.shape[1]):
        #clear this variable at start of every lvl
        feature_to_add_at_this_level = None
        best_so_far_accuracy = 0
        
        #deletes this list at every start of lvl
        #del feature_to_add_at_this_lvl[:]

        print(">> On level: {} of the search tree".format(i))
        for j in range(1, df.shape[1]):
            #only continue forloop if not already added
            if j not in desired_features: 
                print("   -Considering adding feature {}...".format(j))
                list2 = [j] #convert feature to list to add to a list
                features_to_test = desired_features + list2
                #subset_df = data[features_to_test] #done in LOOCV def
                #print(subset_df)                   #done in LOOCV def
                accuracy = leave_one_out_CV(data, features_to_test)
                print("feature {} has an accuracy of: {}".format(j, accuracy))
                if accuracy > best_so_far_accuracy: 
                    best_so_far_accuracy = accuracy;
                    feature_to_add_at_this_level = j

        desired_features.append(feature_to_add_at_this_level)
        print("    << On level{}, feature>{}< was added to desired features with accuracy {}".format(i, feature_to_add_at_this_level, best_so_far_accuracy))
        print("       >>>>current desired features<<<<<: ", desired_features)  
        #deepcopy b/c minheap's desired_features kept updating
        #prob was just a pointer
        tmp_deep_copy = copy.deepcopy(desired_features)
        heapq.heappush(maxheap, (best_so_far_accuracy, tmp_deep_copy))
        #print("maxheap while changing: ", maxheap)
    pop_largest = heapq.nlargest(1, maxheap)
    best_accuracy = pop_largest[0][0]
    best_set = pop_largest[0][1]
    #print("pop_largest: ", pop_largest)
    print("Overall, the best set is {}, with an accuracy of: {}".format(best_set, best_accuracy))
    return




def main():
    #applies normalization col-wise so that each col is a z-score with 0 as the mean
    #and el vals 1 or -1 representing one std from the mean
    #improves kNN
    #print(data)
    print(">>>>>>>>>>>>>")

    z_normalize_per_feature(data)

    #print(data)
    print(">>>>>>>>>>>>>")

    forward_selection(data)

    

    return
main()
