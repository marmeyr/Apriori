#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__="Alexandra MILLOT"
__date__="24.01.23"
__update__ = "31.01.23 17:48"
__usage__ = """
TID Apriori
Project 2022-2023 IA
"""

import pandas as pd
import numpy as np


class Arules:
    """ build association rules 
    expect the list of itemsets and the support of each itemsets
    """
    def __init__(self, list_of_itemsets:list, dict_support:dict):
        """ constructor simple storage """
        self.list_itemsets = list_of_itemsets
        self.support_itemsets = dict_support
        self.reset()

    def reset(self):
        """ set rules to empty list """
        self.rules = []

    def support(self, lhs:tuple, rhs:tuple) -> float:
        """ support lhs U rhs """
        _ = tuple(sorted(lhs+rhs))
        return self.support_itemsets[_]
    
    def confidence(self, lhs:tuple, rhs:tuple) -> float:
        """ support(lhs U rhs)/support(lhs) """
        _ = tuple(sorted(lhs+rhs))
        return self.support_itemsets[_]/self.support_itemsets[lhs]

    def lift(self, lhs:tuple, rhs:tuple) -> float:
        """ support(lhs U rhs) / (support(lhs) * support(rhs)) """
        _ = tuple(sorted(lhs+rhs))
        return self.support_itemsets[_]/(self.support_itemsets[lhs]*
                                         self.support_itemsets[rhs])

    def leverage(self, lhs:tuple, rhs:tuple) -> float:
        """ support(lhs U rhs) - (support(lhs) * support(rhs)) """
        _ = tuple(sorted(lhs+rhs))
        return self.support_itemsets[_] - (self.support_itemsets[lhs]*
                                           self.support_itemsets[rhs])
    
    def conviction(self, lhs:tuple, rhs:tuple) -> float:
        """ (1 - supp(rhs)) / (1 - confidence(lhs, rhs)) """
        _ = self.confidence(lhs, rhs)
        if _ == 1: return None
        return (1 - self.support_itemsets[rhs]) / (1 - _)

    def lift_diag(self, lhs:tuple, rhs:tuple) -> str:
        """ simple diagnostic for lift computation """
        _lift = self.lift(lhs, rhs)
        if _lift == 1:
            _msg = "ne pas utiliser {} -> {}"
        elif _lift < 1:
            _msg = "{} et {} ne peuvent pas co-exister dans une règle"
        else:
            _msg = "{} -> {} est prédictive"
        return _msg.format(lhs, rhs)

    def cross_product(self, L:list, k:int) -> list:
        """ L: a list of k-itemsets
            return a list of k+1 itemsets
        """
        _bag = []
        _sz = len(L)
        for i in range(_sz -1):
            j = i+1
            while j < _sz and L[i][:-1] == L[j][:-1]:
                _0 = L[i]+L[j][-1:]
                # pruning
                if all(_0[:p]+_0[p+1:] in L
                       for p in range(k-1)): _bag.append(_0)
                j += 1
        return _bag

    def validation_rules(self, itemset:tuple, rhsLst:list,
                         threshold:float) -> list:
        """ return the list of rhs belonging to rhsLst
            such that (itemset - rhs) -> rhs has confidence
            above seuil
        """
        _prune = []
        for rhs in rhsLst:
            lhs = tuple(sorted(set(itemset)-set(rhs)))
            if self.confidence(lhs, rhs) >= threshold:
                _prune.append(rhs)
                self.rules.append( (lhs, rhs) )
        return _prune

    def build_rules(self, itemset:tuple, rhsLst:list,
                    threshold:float):
        """ return nothing, given itemset = (x1, .. xn)
            build all rules from the itemset
        """
        _working = rhsLst
        _item_sz = len(itemset)
        _rhs_sz = 1
        while len(_working) > 1 and _item_sz > _rhs_sz+1:
            _rhs_sz += 1
            _working = self.cross_product(_working, _rhs_sz)
            _working = self.validation_rules(itemset, _working, threshold)

    def generate_rules(self, minConf:float):
        """ build the rules """
        self.reset()
        for l in self.list_itemsets[1:]: # 0 are 1-tuples
            _sz = len(l[0])
            for its in l:
                rhsLst = [tuple([x]) for x in its]
                if _sz == 2:
                    self.validation_rules(its, rhsLst, minConf)
                else:
                    self.build_rules(its, rhsLst, minConf)

    def main(self, minConf:float) -> pd.DataFrame:
        """ call generate_rules and provides a DataFrame
            with 9 columns """
        self.generate_rules(minConf)
        keys = "lhs rhs lhs_support rhs_support"
        keys += " support confidence lift leverage conviction"
        _d = {x:[] for x in keys.split()}
        for a,b in self.rules:
            _d['lhs'].append(a)
            _d['rhs'].append(b)
            _d['lhs_support'].append(self.support_itemsets[a])
            _d['rhs_support'].append(self.support_itemsets[b])
            for k in keys.split()[4:]: #call metrics
                _d[k].append(getattr(self,k)(a,b))
        return pd.DataFrame(_d).fillna(np.inf)
    
