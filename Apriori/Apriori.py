#coding=utf-8
from collections import Counter
from itertools import combinations

class Apriori:
    """
    数据格式：形如[['1','2','3'],
            ['2','3','4','5'],
            ...
            ]的矩阵

    """
    def __init__(self,data,min_sup = 0.4):
        self.data = data 
        self.min_sup = min_sup
        self.size = len(data) 
        self.min_sup_val = self.min_sup*self.size
    def find_l1_itemsets(self):
        """
        计算１-项频繁集
        """
        item_freq = Counter()
        for row in self.data:
            for col in row:
                item_freq[col] += 1
        c1 = filter(lambda x:x[1]>self.min_sup_val,item_freq.iteritems())
        l1 = []
        for item in c1:
            l1.append([item[0]])
        l1.sort()
        return  l1

    def has_infreq_subset(self,c,l_last,k):
        """
        检查子集是否为频繁集
        """
        subsets = list(combinations(c,k-1))
        for subset in subsets:
            subset = list(subset)
            if subset not in l_last:
                return True
            return False

    def apriori_gen(self,l_last):
        """
        由k-1频繁集生成k候选集
        """
        k = len(l_last[0]) + 1 
        ck = []
        for itemset1 in l_last:
            for itemset2 in l_last:
                flag = 0
                for i in range(k-2):  ##字典序的排列方式，生成k项连接只需考虑前k-2项相同的项
                    if itemset1[i] != itemset2[i]:
                        flag = 1
                        break 
                if flag == 1 : continue
                if itemset1[k-2] < itemset2[k-2]:
                    c = itemset1 + [itemset2[k-2]]
                else:
                    continue
                if self.has_infreq_subset(c,l_last,k):
                    continue
                else:
                    ck.append(c)
        return ck 
    def apriori(self):
        """
        运行算法，简单打印从２-项频繁集开始的所有频繁集，返回的是所有频繁集
        """
        l = []
        l_last = self.find_l1_itemsets()
        l=l_last
        while l_last != []:
            ck = self.apriori_gen(l_last)
            freq_dic = {}
            for row in self.data: 
                for c in ck:
                    if set(c) <= set(row):
                        freq_dic[tuple(c)] = freq_dic.get(tuple(c),0) + 1
            lk = []
            for key,val in freq_dic.iteritems():
                if val > self.min_sup_val:
                    lk.append(list(key))
                    print 'k={}'.format(len(key)),list(key),val 
            l_last = lk 
            l+=lk
        return l 


if __name__ == '__main__':
   data = [['1','2','3'],['2','3','4'],['3','4','5'],['1','3','4'],['2','5','10','11'],['1','2','5'],['3','5','7'],['1','2','3','4','5'],['1','2','3']]
   apriori = Apriori(data)
   print apriori.apriori()
 