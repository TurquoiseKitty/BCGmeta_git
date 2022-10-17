import numpy as np
import random

# for given 0 < lam < 1, given N candidate products, generated assortment will contain averagely lam * N products
def GenAssortment_Even(N_prod = 10, lam=1/2, **kwargs):
    potential_vec = np.random.uniform(low=0., high=1, size=N_prod)
    assortment_vec = np.zeros(N_prod)
    assortment_vec[potential_vec <= lam] = 1
    return assortment_vec


# generate assortment containing fixed number of products
def GenAssortment_Fixed(N_prod = 10, fixed_num = 6, **kwargs):
    positions = random.sample(list(range(N_prod)),k=fixed_num)
    assortment_vec = np.zeros(N_prod)
    assortment_vec[positions] = 1
    return assortment_vec


def GenAssortment_Abundant(N_prod = 10, **kwargs):
    fixied = random.sample(list(range(1,N_prod+1)),k=1)[0]
    return GenAssortment_Fixed(N_prod, fixied)


def GenAssortment_Cut(N_prod = 10, cut = 1/2, **kwargs):

    assortment_vec = np.zeros(N_prod)
    up_down = np.random.randint(2)
    if up_down == 0:
        assortment_vec[ : int(N_prod * cut)] = GenAssortment_Abundant(int(N_prod * cut))
    else:
        
        assortment_vec[int(N_prod * cut) : ] = GenAssortment_Abundant(N_prod - int(N_prod * cut))


    return assortment_vec


def Gen_NNplus(N_prod = 10, fixed_num = 6, **kwargs):

    
    up_down = np.random.randint(2)

    if up_down == 0:
        return GenAssortment_Fixed(N_prod, fixed_num)

    else:
        return GenAssortment_Fixed(N_prod, fixed_num + 1)
    
def Gen_Mix(N_prod, NNplus_fixed_num, Cut, Even_lam, **kwargs):

    up_down = np.random.randint(4)

    if up_down == 0:
        return GenAssortment_Abundant(N_prod)

    elif up_down == 1:
        return GenAssortment_Cut(N_prod, Cut)

    elif up_down == 2:
        return GenAssortment_Even(N_prod, Even_lam)

    elif up_down == 3:
        return Gen_NNplus(N_prod, NNplus_fixed_num)



GenAssortment = {
    "Even" : GenAssortment_Even,
    "Fixed": GenAssortment_Fixed,
    "Abundant": GenAssortment_Abundant,
    "Cut": GenAssortment_Cut,
    "NNplus": Gen_NNplus,
    "Mix": Gen_Mix
}
