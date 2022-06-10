########################################################################
#
# Date: April 2006 Authors: Guillaume Vareille, Michel Sanner
#
#    sanner@scripps.edu
#    vareille@scripps.edu
#
#       The Scripps Research Institute (TSRI)
#       Molecular Graphics Lab
#       La Jolla, CA 92037, USA
#
# Copyright: Guillaume Vareille, Michel Sanner and TSRI
#
# revision:
#
#########################################################################
#
# $Header: /mnt/raid/services/cvs/python/packages/share1.5/Pmv/VisionInterface/PmvTypes.py,v 1.3 2006/11/01 21:03:55 vareille Exp $
#
# $Id: PmvTypes.py,v 1.3 2006/11/01 21:03:55 vareille Exp $
#

from NetworkEditor.datatypes import AnyArrayType

class PmvType(AnyArrayType):

    from Pmv.moleculeViewer import MoleculeViewer
    def __init__(self, name='PmvInstance', color='#7A7AFF', shape='circle',
                 klass=MoleculeViewer):
      
        AnyArrayType.__init__(self, name=name, color=color, shape=shape, 
                              klass=klass)



class PmvCmdType(AnyArrayType):

    from ViewerFramework.VFCommand import Command
    def __init__(self, name='PmvCmd', color='#7A7AFF', shape='rect',
                 klass=Command):
      
        AnyArrayType.__init__(self, name=name, color=color, shape=shape, 
                              klass=klass)
