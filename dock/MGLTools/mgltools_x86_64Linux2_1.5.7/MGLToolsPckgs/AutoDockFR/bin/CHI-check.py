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
# $Header: /mnt/raid/services/cvs/AutoDockFR/bin/CHI-check.py,v 1.1.2.1 2015/08/26 21:02:58 sanner Exp $
#
# $Id: CHI-check.py,v 1.1.2.1 2015/08/26 21:02:58 sanner Exp $
#

"""
This module implements a utility to randomize rotameric side chains in the receptor

"""
from bhtree import bhtreelib

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
    flexAtoms = []
    ## for m in ft.getAllMotion():
    ##     print 'Random angles for residue', m.name,
    ##     if isinstance(m, FTMotion_SoftRotamer):
    ##         angles = []
    ##         for i in range(len(m.angDef)):
    ##             angles.append( uniform(0., 360.) )
    ##             print angles[-1],
    ##         print
    ##         m.setAngles(angles)
    ##         m.apply(m.node().molecularFragment)
    ##         allAngles.append(angles)
    ##         flexAtoms.extend(m.node().molecularFragment)
    ##         for i in range(len(m.angDef)):
    ##             angles.append( uniform(0., 360.) )
    ##             m.node().molecularFragment.coords

    # try to remove clashes
    #
    # build BHTree or rigid coords
    rigidAtoms = mol.allAtoms - (FTgenerator.interfaceAtoms+ft.getMovingAtoms())
    bht = bhtreelib.BHtree( rigidAtoms.coords, None, 10)

    import numpy
    results = numpy.zeros(5000, 'i')
    dist2 = numpy.zeros(5000, 'f')
    maxbo = max([a.bondOrderRadius for a in rigidAtoms])
    # loop over flexible residues and find a conformation with no clash
    for m in ft.getAllMotion():
        #print 'Random angles for residue', m.name,
        found = False
        if isinstance(m, FTMotion_SoftRotamer):
            ct = 0
            #print m.angleList
            for angles in m.angleList:
                ct += 1
                m.setAngles(angles)
                m.apply(m.node().molecularFragment)
                newFrag = m.node().molecularFragment
                #added by pradeep for clash free rotamers
                #rigidAtoms = mol.allAtoms - (FTgenerator.interfaceAtoms+ft.getMovingAtoms())#-newFrag)
                #bht = bhtreelib.BHtree( rigidAtoms.coords, None, 10)
                #results = numpy.zeros(5000, 'i')
                #dist2 = numpy.zeros(5000, 'f')
                #maxbo = max([a.bondOrderRadius for a in rigidAtoms])
                clash = False
                newFrag.reverse()
                maxbo = max(maxbo, max([a.bondOrderRadius for a in newFrag]))
                cutoff = maxbo+maxbo
                #for i,pt in enumerate(m.node().molecularFragment.coords):
                for i,pt in enumerate(newFrag.coords):
                    #m.node().molecularFragment[i].radius
                    nb = bht.closePointsDist2(tuple(pt), cutoff, results, dist2)
                    bo1 = newFrag[i].bondOrderRadius
                    #print dist2[:nb]
                    #print cutoff
                    for ind, d2 in zip(results[:nb], dist2[:nb]):
                        bo = bo1 + rigidAtoms[ind].bondOrderRadius
                        if d2 < bo*bo*0.81:#6.0:
                            clash = True
                            #break
                    #if clash:
                    #    break
                if not clash:
                    print 'motion %s without clash'%m.name, "try: ", ct
                    rigidAtoms += m.rotamer.atoms
                    bht = bhtreelib.BHtree( rigidAtoms.coords, None, 10)
                    found = True
                    #break
            print "Try: ", ct
            #if not found:
            #    raise RuntimeError, 'no rotamer without clash for %s'%m.name
                
    return allAngles, flexAtoms



if __name__=='__main__':
    
    import sys
    molfile = sys.argv[1]
    flexResSelectionString = sys.argv[2]
    outFilename = sys.argv[3]

    from MolKit import Read
    mol = Read(molfile)[0]
    mol.buildBondsByDistance()
    randomizeSoftRot(mol, flexResSelectionString)
    
    # write the molecule
    comments = [
        "*********************************************************\n",
        "receptor with randomized soft rotameric side chains\n",
        '*********************************************************'
        ]
    mol.parser.write_with_new_coords(
        mol.allAtoms.coords, filename=outFilename, comments=comments,
        withBondsFor=mol)

