########################################################################
#
# Date: May 2005 Author: Michel Sanner, Pradeep Ravindranath
#
#    sanner@scripps.edu
#
#       The Scripps Research Institute (TSRI)
#       Molecular Graphics Lab
#       La Jolla, CA 92037, USA
#
# Copyright: TSRI
#
#########################################################################
#
# $Header: /mnt/raid/services/cvs/AutoDockFR/bin/randomizeReceptor.py,v 1.3.2.1 2015/08/26 21:02:58 sanner Exp $
#
# $Id: randomizeReceptor.py,v 1.3.2.1 2015/08/26 21:02:58 sanner Exp $
#

"""
This module implements a utility to randomize rotameric side chains in the receptor

"""

def randomizeSoftRot(mol, flexResSelectionString):
    from AutoDockFR.utils import RecXML4SoftRotamSC

    # make sure the molecule has bonds. Needed to find Hn atom
    assert len(mol.allAtoms.bonds[0]) > len(mol.allAtoms)
    FTgenerator = RecXML4SoftRotamSC(mol, flexResSelectionString)
    ft = FTgenerator.getTree()

    from random import uniform

    # loop over motion objects and for FTMotion_SoftRotamer set random CHI angles
    from FlexTree.FTMotions import FTMotion_SoftRotamer
    allAngles = []
    for m in ft.getAllMotion():
        print 'Random angles for residue', m.name,
        if isinstance(m, FTMotion_SoftRotamer):
            angles = []
            for i in range(len(m.angDef)):
                angles.append( uniform(0., 360.) )
                print angles[-1],
            print
            m.setAngles(angles)
            m.apply(m.node().molecularFragment)
            allAngles.append(angles)
            
    return allAngles

if __name__=='__main__':
    
    import sys
    from MolKit import Read
    from astexNativeActiveSiteRes_AGP_CysCys_ResM_85 import flexRes
    names = ['1g9v', '1ia1', '1jla', '1l7f', '1mmv', '1n46', '1oq5', '1pmn', '1r55', 
               '1sg0', '1t46', '1u1c', '1v0p', '1y6b', '1ywr', '1gkc', '1hq2', '1ig3', 
	       '1k3u', '1lpz', '1mzc', '1nav', '1owe', '1q1g', '1r58', '1sj0', '1t9b', '1u4d', 
	       '1v48', '1x8x', '1ygc', '1z95', '1gm8', '1hvy', '1j3j', '1ke5', '1lrh', '1n1m', 
	       '1of1', '1oyt', '1q41', '1r9o', '1sq5', '1tow', '1uml', '1v4s', '1xm6', '1yqy', 
	       '2bm2', '1gpk', '1hwi', '1jd0',]
    names = [  '1kzk', '1m2z', '1n2j', '1of6', '1p2y', '1q4g', 
	       '1s19', '1sqn', '1tt1', '1unl', '1vcj', '1xoq', '1yv3', '2br1', '1hnn', '1hww',  
	       '1jje', '1l2s', '1meh', '1n2v', '1opk', '1p62', '1r1h', '1s3v', '1t40', '1tz8',  
	       '1uou', '1w1p', '1xoz', '1yvf', '2bsm', '1hp0', '1w2g']
    names = [  '1v0p']
    
    for name in names:
    
        molfile = "/localhome/export2/people/pradeep/Astex-jan2014/receptorsuniq-85/%s_rec.pdbqt"%name
        flexResSelectionString = flexRes[name]
        outFilename = "/localhome/export2/people/pradeep/Astex-jan2014/receptorsuniq-85_randomized/%s_rec_random.pdbqt"%name

        
        mol = Read(molfile)[0]
        mol.buildBondsByDistance()
        #angles.append(randomizeSoftRot(mol, flexResSelectionString))
        angles=randomizeSoftRot(mol, flexResSelectionString)
        ## chi = [[], [], [], []]
        ## for i in range(5000):
        ##     print '---->', i
        ##     angsPerCall = randomizeSoftRot(mol, flexResSelectionString)
        ##     for angsPerRot in angsPerCall:
        ##         for i, ang in enumerate(angsPerRot):
        ##             chi[i].append(ang)

        # write the molecule
        comments = [
            "*********************************************************\n",
            "receptor with randomized soft rotameric side chains\n",
            '*********************************************************'
            ]
        mol.parser.write_with_new_coords(
            mol.allAtoms.coords, filename=outFilename, comments=comments,
            withBondsFor=mol)

        ## from Vision import runVision
        ## runVision()
        ## Vision.ed.loadNetwork('/home/sanner/plotAngles_net.py')
    
