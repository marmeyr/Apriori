#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__="Alexandra MILLOT"
__date__="24.01.23"
__usage__ = """
TID Apriori
Project 2022-2023 IA
"""

class Apriori:
    """ simplistic implementation of TID Apriori """
    def __init__(self, dbase:dict):
        """ save data in dbase, call reset """
        self.dbase = dbase
        self.reset()
        
    def reset(self):
        """ provides some values """
        self.candidates_sz = 1
        self.support_history = dict()
        self.candidates = {tid: list(map(lambda x:tuple([x]), self.dbase[tid]))
                           for tid in self.dbase}
        self.current = {}
        for tid in self.candidates:
            for v in self.candidates[tid]:
                _ = self.current.get(v, set())
                _.add(tid)
                self.current[v] = _
                
    def support(self, minsupp:float) -> dict:
        """ return the current itemsets ge minsupp """
        return {k:len(v)/len(self.dbase)
                for k,v in self.current.items()
                if len(v)>=minsupp*len(self.dbase)}
                
    def scan_dbase(self, minsupp:float):
        """ stores information in support_history
            changes values of current

            for a true implementation, this should be private
        """
        _keep = self.support(minsupp)
        self.support_history.update(_keep)
        self.current = {k:self.current[k]
                        for k in _keep}
        
    def Lk(self) -> list:
        """ ordered itemsets with good support """
        return sorted(self.current.keys())
        
    def cross_product(self):
        """ given a list of candidates, build the next generation 

            for a true implementation, this should be private
        """
        _current = {}
        _candidates = {}
        _L = self.Lk()
        _sz = len(_L)
        for i in range(_sz -1):
            j = i+1
            while j < _sz and _L[i][:-1] == _L[j][:-1]:
                _itset = _L[i]+_L[j][-1:]
                _one = self.current[_L[i]]
                _two = self.current[_L[j]]
                # pruning step
                if all([_itset[:p]+_itset[p+1:] in _L
                        for p in range(self.candidates_sz)]):
                    _current[_itset] = _one.intersection(_two)
                    for tid in _current[_itset]:
                        _ = _candidates.get(tid, list())
                        _.append(_itset)
                        _candidates[tid] = _
                j += 1
        self.current = _current
        self.candidates = _candidates
        self.candidates_sz += 1
        
    def main(self, minsupp:float) -> list:
        """ provides all the itemsets with minimal support """
        self.reset()
        self.scan_dbase(minsupp) # only keep supp(itemsets)>=minsupp
        _out = [ self.Lk() ]
        while len(_out[-1]) > 1:
            self.cross_product()
            self.scan_dbase(minsupp)
            _out.append(self.Lk())
        # last list of itemsets might be empty
        return _out[:-1] if _out[-1] == [] else _out
