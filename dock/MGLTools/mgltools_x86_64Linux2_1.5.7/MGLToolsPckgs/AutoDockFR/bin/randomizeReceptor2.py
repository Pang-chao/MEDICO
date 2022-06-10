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
# $Header: /mnt/raid/services/cvs/AutoDockFR/bin/randomizeReceptor2.py,v 1.1.2.1 2015/08/26 21:02:58 sanner Exp $
#
# $Id: randomizeReceptor2.py,v 1.1.2.1 2015/08/26 21:02:58 sanner Exp $
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
    flexAtoms = []
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
            flexAtoms.extend(m.node().molecularFragment)
            
    return allAngles, flexAtoms

def getBondedInteractions(atoms, tree=None, include_1_4_interactions = True):
    #include_1_4_interactions == True
    # build a list of 0-based indices of bonded interactions
    # 1-1, 1-2, 1-3, 1-4
    # bonds might point to atoms not in the set. but a 1-3 interaction
    # between atoms in the set can me mediated through an atom outside
    # the set. hence we can not stop when the bonded atom is outside the set
    # but we on only add pairs where both atoms in the pair belong to the set

    #MLD_8_16_12: Turning this off for single atom energy analysis
    #assert len(atoms.bonds[0]) > 0, "ERROR: atoms in molecule %s have no bonds"%atoms[0].top.name
    interactions = []

    if tree:
        # tage rotatable bonds
        bonds, noBonds = atoms.bonds
        bonds._rotatable = False

        def _traverse(atoms, node):
            if node.bond[0] is not None:
                a1 = atoms[node.bond[0]]
                a2 = atoms[node.bond[1]]
                for b in a1.bonds:
                    if b.atom1==a2 or b.atom2==a2:
                        b._rotatable = True
            for c in node.children:
                _traverse(atoms, c)

        _traverse(atoms, tree.rootNode)

    ## create a sequential number
    atoms.uniqInterNum = range(len(atoms))
    ## outer loop over atoms
    for a in atoms:
        # loop over the bonds of a
        n1 = a.uniqInterNum
        interactions.append( (n1,n1) ) # 1-1

        ## loop over atoms bonded to a
        for b in a.bonds:
            # find a2 which is bonded to a through b
            a2 = b.atom1
            if a2==a: a2=b.atom2
            if hasattr(a2, 'uniqInterNum'):
                # only exlude pair if both atoms are in the set atoms
                interactions.append( (n1,a2.uniqInterNum) ) # 1-2

            ## loop over bonds of at2 to find 1-3 interactions
            for b1 in a2.bonds:
                # find a2 which is bonded to a2 through b1
                a3 = b1.atom1
                if a3==a2: a3=b1.atom2
                if a3 == a : continue
                if hasattr(a3, 'uniqInterNum'):
                    interactions.append( (n1,a3.uniqInterNum) ) # 1-3

                ## loop over bonds of at3 to find 1-3 interactions
                if (include_1_4_interactions == True):
                    # we only remove the 1-4 if a2-a3 bond is not rotatable
                    if tree is None or not b1._rotatable:
                        for b2 in a3.bonds:
                            # find a4 which is bonded to a3 through b2
                            a4 = b2.atom1
                            if a4==a3: a4=b2.atom2
                            if a4 == a2 : continue
                            if hasattr(a4, 'uniqInterNum'):
                                interactions.append( (n1,a4.uniqInterNum) ) # 1-4

                else: # 1-4 interaction have to be removed, so we append them to the list
                    for b2 in a3.bonds:
                        # find a4 which is bonded to a3 through b2
                        a4 = b2.atom1
                        if a4==a3: a4=b2.atom2
                        if a4 == a2 : continue
                        if hasattr(a4, 'uniqInterNum'):
                            interactions.append( (n1,a4.uniqInterNum) ) # 1-4


    for a in atoms:
        del a.uniqInterNum

    return interactions


def pyMolToCAtomVect(mol):
    """
    pyAtomVect <- pyMolToCAtomVect()

    convert Protein or AtomSet to AtomVector
    """

    className = mol.__class__.__name__
    if className == 'Protein':
        pyAtoms = mol.getAtoms()
    elif className == 'AtomSet':
        pyAtoms = mol
    else:
        print 'Warning: Need a AtomSet or Protein'
        raise ValueError
        return None
    pyAtomVect = AtomVector()

    confNB = pyAtoms[0].conformation
    pyAtoms.setConformation(0)
    for atm in pyAtoms:
        a = Atom()
        a.set_name(atm.full_name())
        #if atm.autodock_element == "Cl":
        #    print "Error: Cl should be c, as autodock_element"
        #    raise ValueError
        #if atm.autodock_element == "Br":
        #    print "Error: Br should be b, as autodock_element"
        #    raise ValueError                
        a.set_element(atm.autodock_element)# aromatic type 'A', vs 'C'
        coords = atm.coords
        a.set_coords( Coords(coords[0],coords[1],coords[2]))
        a.set_charge( atm.charge)
        if hasattr(atm, 'AtVol'):
            a.set_atvol( atm.AtVol)

        if hasattr(atm, 'AtSolPar'):
            a.set_atsolpar( atm.AtSolPar)

        if hasattr(atm, 'atInterface'): # an interface atom?
            a.set_atInterface(atm.atInterface)

        a.set_bond_ord_rad( atm.bondOrderRadius)
        a.set_atom_type( atm.autodock_element) # this is used for DS & setting vdw correctly in cAutoDockDist code
        #a.set_charge( atm.charge)
        pyAtomVect.append(a)

    if confNB != 0:
        pyAtoms.setConformation(confNB)

    return pyAtomVect

## class NamedWeightedMultiTerm(WeightedMultiTerm):
##     """
##     subclass WeightedMultiTerm in order to keep track of scoring componet names

##     """

##     def __init__(self,*args):
##         WeightedMultiTerm.__init__(self, *args)
##         self.names = []

##     def add_term(self, term, weight, name):
##         self.names.append(name)
##         WeightedMultiTerm.add_term(self, term, weight)



if __name__=='__main__':
    
    import sys
    import numpy
    from MolKit import Read
    from cAutoDock.scorer import CoordsVector, Coords, MolecularSystem,\
     isNAN, AtomVector, Atom, Coords, InternalEnergy
    from cAutoDock._scorer import updateCoords
    from memoryobject.memobject import return_share_mem_ptr, allocate_shared_mem, FLOAT
    from cAutoDock.scorer import  Electrostatics, \
     HydrogenBonding, VanDerWaals, Desolvation4, WeightedMultiTerm
    import AutoDockFR.ScoringFunction as scoreFn
    import AutoDockFR.ADCscorer as ADC
    from astexNativeActiveSiteRes_AGP_CysCys_ResM_85 import flexRes
    #from MolKit.molecule import Atom, AtomSet

    FE_coeff_vdW_42		= 0.1662 # van der waals
    FE_coeff_hbond_42	= 0.1209 # hydrogen bonding
    FE_coeff_estat_42	= 0.1406 # electrostatics
    FE_coeff_desolv_42	= 0.1322 # desolvation
    FE_coeff_tors_42	= 0.2983 # torsional 

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
        flexMol = Read('1v0p/AllFlexSC.pdbqt')[0]
        flexMol.buildBondsByDistance()
        flexRecAtoms= flexMol.allAtoms
        
        #angles.append(randomizeSoftRot(mol, flexResSelectionString))
        sf = scoreFn.ScoringFunction()
        molSyst = sf.addMolecularSystem('FlexReceptor-FlexReceptor',
                                      internal=True)
        FRFRmolSyst = molSyst
        flexRecCatoms=pyMolToCAtomVect(flexRecAtoms)
        msentity1 = molSyst.add_entities(flexRecCatoms)
        msentity2 = molSyst.add_entities(flexRecCatoms)
        molSyst.atomSetIndices = {'set1':msentity1, 'set2':msentity2}
        molSyst.set_use_mask(1)
        bondedInter = getBondedInteractions(flexRecAtoms,
                                                 )
        for i,j in bondedInter:
            molSyst.set_mask( i, j, 0)
        estat = Electrostatics(molSyst)
        hBond = HydrogenBonding(molSyst)
        vdw = VanDerWaals(molSyst)
        vdw.set_ad_version("4.0")
        ds = Desolvation4(molSyst)
        FRFRscorer = ADC.NamedWeightedMultiTerm(molSyst)
        vdw.set_symmetric(True)
        #hBond.set_symmetric(True)
        hBond.set_directional(False)
        hBond.set_NA_HDfactor(False)
        estat.set_symmetric(True)
        ds.set_symmetric(True)
        FRFRscorer.add_term(estat, FE_coeff_estat_42, 'electrostatics')
        FRFRscorer.add_term(hBond, FE_coeff_hbond_42, 'hBonds')
        FRFRscorer.add_term(vdw,   FE_coeff_vdW_42, 'vdw')
        FRFRmolSyst.scorer = FRFRscorer
        molSyst.terms = [vdw, estat, hBond, ds]

        msLen = 2 * len(flexRecCatoms)
        FRFRscorer.sharedMem = allocate_shared_mem([msLen, 3],'FRFRSharedMemory', FLOAT)
        FRFRscorer.sharedMemPtr = return_share_mem_ptr('FRFRSharedMemory')[0]
        FRFRscorer.sharedMemLen = msLen
        FRFRscorer.sharedMemSplit = len(flexRecCatoms)

        # The bonds have to be computed AFTER the pairwaise scorers are created
        # else the bond list in C++ is corrupted
        molSyst.build_bonds(msentity1)
        molSyst.build_bonds(msentity2)
        DEBUG = 0
        FRScore = 9999.0
        while FRScore >3:


            angles,flexAtoms=randomizeSoftRot(mol, flexResSelectionString)
            from MolKit.molecule import Atom, AtomSet
            flexAtoms2 = AtomSet(flexAtoms)
            FR_coords= flexAtoms2.coords+flexAtoms2.coords
            #import pdb
            #pdb.set_trace()
            #assert len(flexMol.allAtoms) is not len(flexAtoms2), 'Check flexAtoms'
            scorer = FRFRmolSyst.scorer
            scorer.sharedMem[:] = numpy.array(FR_coords, 'f')
	    updateCoords(scorer.sharedMemSplit, scorer.sharedMemLen, FRFRmolSyst,\
                         scorer.sharedMemPtr)
            # if distances are > cutoff, NAN is return.
            # else: the offending distance is returned. 
	    # The distance matrice might only be partially populated as we 
	    # break out as soon as a distance smaller than cutoff is seen. 
	    # Scoring should not occur in this case as the clash is too severe
	    mini = FRFRmolSyst.check_distance_cutoff(0, 1, 1.0)
	    nan = isNAN(mini)
	    if not nan:
	        if DEBUG:
		    print "Not scoring because Flex-Flex rec clash =", mini
	        scoreFRFR = min(9999999.9, 9999.9/mini) #larger than 10K and smaller than 10M
	        # Return a large negative value (not favorable interaction)
	        # This is due to the GA fitness/performance looking for a maximum
                FRScore = scoreFRFR
		
	    else:
		scoreFRFR = scorer.get_score() * FRFRmolSyst.factor
                FRScore = scoreFRFR
            print FRScore
            
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
    
