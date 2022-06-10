import os
from astexNativeActiveSiteRes_AGP_CysCys_ResM_85 import flexRes
names = ['1g9v', '1ia1', '1jla', '1l7f', '1mmv', '1n46', '1oq5', '1pmn', '1r55', 
         '1sg0', '1t46', '1u1c', '1v0p', '1y6b', '1ywr', '1gkc', '1hq2', '1ig3', 
         '1k3u', '1lpz', '1mzc', '1nav', '1owe', '1q1g', '1r58', '1sj0', '1t9b', '1u4d', 
         '1v48', '1x8x', '1ygc', '1z95', '1gm8', '1hvy', '1j3j', '1ke5', '1lrh', '1n1m', 
         '1of1', '1oyt', '1q41', '1r9o', '1sq5', '1tow', '1uml', '1v4s', '1xm6', '1yqy', 
         '2bm2', '1gpk', '1hwi', '1kzk', '1m2z', '1n2j', '1of6', '1p2y', '1q4g', 
         '1s19', '1sqn', '1tt1', '1unl', '1vcj', '1xoq', '1yv3', '2br1', '1hnn', '1hww',  
         '1jje', '1l2s', '1meh', '1n2v', '1opk', '1p62', '1r1h', '1s3v', '1t40', '1tz8',  
         '1uou', '1w1p', '1xoz', '1yvf', '2bsm', '1hp0', '1w2g']#'1jd0',
    
names = [  '1g9v']#'1v0p']
for num, name in enumerate(names):
    flexResSelectionString = flexRes[name]
    os.system("pythonsh ~/links/dev/ADFR-FRlite-Aug2014-stable/AutoDockFR/bin/randomizeReceptor-chichk.py ~/links/dev/Astex-jan2014/receptorsuniq-85/%s_rec.pdbqt %s ~/links/dev/Astex-jan2014/receptorsuniq-85_randomized-clashfree/%s_rec_random.pdbqt"%(name,flexResSelectionString,name))
