import numpy as np
import os


class MC:

    def __init__(

        self, 
        N_prod, 

        Sigma = 1,

        **kwargs

    ):

        self.ModelName = "MC"

        self.N_prod = N_prod

        self.Lam = np.zeros(N_prod)

        self.M = np.zeros((N_prod, N_prod))

        self.self_gen_instance(Sigma)


    def self_gen_instance(self, Sigma):

        Lam = np.random.normal(0, Sigma, self.N_prod)

        
        Lam = np.exp(Lam)

        self.Lam = Lam / sum(Lam)

        
        M = np.random.normal(0, Sigma, (self.N_prod, self.N_prod))

        M = np.exp(M)

        self.M = M / M.sum(axis=1, keepdims = True)


    def prob_for_assortment(self, prod_assort):

        Assort = prod_assort
        S_plus = np.squeeze(np.argwhere(Assort == 1),axis=1)
        S_bar = np.squeeze(np.argwhere(Assort == 0),axis=1)
        B = self.M[np.expand_dims(S_bar, axis=1), S_plus]
        C = self.M[np.expand_dims(S_bar, axis=1), S_bar]


    
        distri = np.zeros(self.N_prod)
    
        addi = np.matmul(np.matmul(np.expand_dims(self.Lam[S_bar], axis=0), np.linalg.inv(np.identity(len(C)) - C)), B)
    
        count = 0
        for i in S_plus:
            distri[i] = self.Lam[i] + addi[0,count]
            count += 1
    
        return distri
        

    # we gen so that the prob calculation can be verified
    def gen_final_choice_NONPRACTICAL(self, prod_assort):

        Assort = prod_assort

        bundle = np.array([i for i in range(self.N_prod) if Assort[i] == 1 ])
        
        favorate =  np.random.choice(self.N_prod, 1, p=self.Lam)[0]

        while favorate not in bundle:

            new_prob = self.M[favorate]

            favorate =  np.random.choice(self.N_prod, 1, p=new_prob)[0]

        choice = np.zeros(self.N_prod)
        choice[favorate] = 1

        return choice

    def gen_final_choice(self, prod_assort):

        fin = np.zeros(self.N_prod)
        
        probs = self.prob_for_assortment(prod_assort)

        choice = np.random.choice(self.N_prod, 1, p=probs)[0]

        fin[choice] = 1

        return fin


    def save_para(self, folder_path):

        path = folder_path

        
        with open(folder_path+"/TrueModel_description.txt", 'w') as f:
            f.write("model type : "+self.ModelName)
            f.write("\n")
            f.write("product number : "+str(self.N_prod))
            f.write("\n")
            

        np.save(folder_path+"/Lam", self.Lam)
        np.save(folder_path+"/M", self.M)



